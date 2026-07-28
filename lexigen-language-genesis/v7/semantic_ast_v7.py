from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

Grid = tuple[tuple[int, ...], ...]
Point = tuple[int, int]


class SemanticRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Region:
    points: frozenset[Point]
    colour: int

    @property
    def anchor(self) -> Point:
        return min(self.points)

    @property
    def area(self) -> int:
        return len(self.points)


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise SemanticRuntimeError("grid must be a non-empty rectangle")
    return grid


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def inferred_background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def border_mode_colours(grid: Grid) -> set[int]:
    height, width = grid_shape(grid)
    border = [grid[r][c] for r in range(height) for c in range(width) if r in (0, height - 1) or c in (0, width - 1)]
    counts = Counter(border)
    maximum = max(counts.values())
    return {colour for colour, count in counts.items() if count == maximum}


def neighbours(point: Point, height: int, width: int) -> Iterable[Point]:
    row, col = point
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = row + dr, col + dc
        if 0 <= nxt[0] < height and 0 <= nxt[1] < width:
            yield nxt


def components_for_colour(grid: Grid, colour: int) -> list[Region]:
    height, width = grid_shape(grid)
    unseen = {(r, c) for r in range(height) for c in range(width) if grid[r][c] == colour}
    regions: list[Region] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        points = {start}
        while queue:
            current = queue.popleft()
            for nxt in neighbours(current, height, width):
                if nxt in unseen:
                    unseen.remove(nxt)
                    points.add(nxt)
                    queue.append(nxt)
        regions.append(Region(frozenset(points), colour))
    return sorted(regions, key=lambda region: (region.anchor, region.area))


def touches_border(region: Region, height: int, width: int) -> bool:
    return any(r in (0, height - 1) or c in (0, width - 1) for r, c in region.points)


def boundary_colours(grid: Grid, region: Region) -> tuple[int, ...]:
    height, width = grid_shape(grid)
    boundary: set[Point] = set()
    for point in region.points:
        for nxt in neighbours(point, height, width):
            if nxt not in region.points:
                boundary.add(nxt)
    return tuple(sorted(grid[r][c] for r, c in boundary))


def normalise(points: Iterable[Point]) -> frozenset[Point]:
    values = list(points)
    if not values:
        return frozenset()
    min_row = min(row for row, _ in values)
    min_col = min(col for _, col in values)
    return frozenset((row - min_row, col - min_col) for row, col in values)


def dihedral_variants(points: Iterable[Point]) -> set[frozenset[Point]]:
    current = list(normalise(points))
    variants: set[frozenset[Point]] = set()
    for _ in range(4):
        variants.add(normalise(current))
        variants.add(normalise((row, -col) for row, col in current))
        current = [(col, -row) for row, col in current]
    return variants


def bbox_shape(region: Region) -> tuple[int, int]:
    rows = [row for row, _ in region.points]
    cols = [col for _, col in region.points]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def extract_holes(grid: Grid, background: int, frame_colour: int, boundary_mode: str) -> list[Region]:
    height, width = grid_shape(grid)
    holes: list[Region] = []
    for region in components_for_colour(grid, background):
        if touches_border(region, height, width):
            continue
        boundary = boundary_colours(grid, region)
        if not boundary:
            continue
        if boundary_mode == "all" and all(value == frame_colour for value in boundary):
            holes.append(region)
        elif boundary_mode == "any" and any(value == frame_colour for value in boundary):
            holes.append(region)
    return sorted(holes, key=lambda region: (region.anchor, region.area))


def extract_objects(grid: Grid, background: int, frame_colour: int, exclude_frame: bool, colour_role: str = "any") -> list[Region]:
    colours = sorted({cell for row in grid for cell in row})
    excluded = {background}
    if exclude_frame:
        excluded.add(frame_colour)
    objects: list[Region] = []
    for colour in colours:
        if colour in excluded:
            continue
        components = components_for_colour(grid, colour)
        if colour_role == "single_component" and len(components) != 1:
            continue
        objects.extend(components)
    return sorted(objects, key=lambda region: (region.area, region.colour, region.anchor))


def regions_match(left: Region, right: Region, predicate: dict[str, Any]) -> bool:
    feature = predicate["feature"]
    if feature == "area":
        return left.area == right.area
    if feature == "bbox":
        return bbox_shape(left) == bbox_shape(right)
    if feature == "normalised_points":
        symmetry = predicate["symmetry"]
        if symmetry == "identity":
            return normalise(left.points) == normalise(right.points)
        if symmetry == "dihedral":
            return normalise(right.points) in dihedral_variants(left.points)
    raise SemanticRuntimeError(f"unsupported predicate: {predicate!r}")


def stable_matching(
    objects: list[Region],
    holes: list[Region],
    predicate: dict[str, Any],
) -> list[tuple[Region, Region]]:
    used: set[int] = set()
    matches: list[tuple[Region, Region]] = []
    for source in objects:
        candidates = [
            index
            for index, hole in enumerate(holes)
            if index not in used and regions_match(source, hole, predicate)
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda index: (holes[index].anchor, holes[index].area))
        used.add(chosen)
        matches.append((source, holes[chosen]))
    return matches


def execute_ast(ast: dict[str, Any], grid: Grid) -> Grid:
    if ast.get("schema") != "lexigen-arc-relational-ast-v1":
        raise SemanticRuntimeError("unsupported semantic AST schema")
    background = int(ast["scene"]["background_colour"])
    frame_colour = int(ast["scene"]["frame_colour"])
    boundary_mode = str(ast["scene"]["hole_boundary"])
    exclude_frame = bool(ast["scene"]["exclude_frame_objects"])
    colour_role = str(ast["scene"].get("object_colour_role", "any"))
    holes = extract_holes(grid, background, frame_colour, boundary_mode)
    objects = extract_objects(grid, background, frame_colour, exclude_frame, colour_role)
    matches = stable_matching(objects, holes, ast["match"])

    values = [list(row) for row in grid]
    erase_source = bool(ast["render"]["erase_source"])
    for source, destination in matches:
        if erase_source:
            for row, col in source.points:
                values[row][col] = background
        for row, col in destination.points:
            values[row][col] = source.colour
    return tuple(tuple(row) for row in values)


def ast_description_length(ast: dict[str, Any]) -> int:
    feature_cost = {"area": 1, "bbox": 2, "normalised_points": 3}[ast["match"]["feature"]]
    symmetry_cost = 0 if ast["match"].get("symmetry", "identity") == "identity" else 2
    boundary_cost = 0 if ast["scene"]["hole_boundary"] == "all" else 2
    exclusion_cost = 0 if ast["scene"]["exclude_frame_objects"] else 1
    erasure_cost = 1 if ast["render"]["erase_source"] else 0
    role_cost = 1 if ast["scene"].get("object_colour_role") == "single_component" else 0
    return 9 + feature_cost + role_cost + symmetry_cost + boundary_cost + exclusion_cost + erasure_cost


def candidate_asts(examples: Sequence[tuple[Grid, Grid]]) -> list[dict[str, Any]]:
    colours = sorted({cell for pair in examples for grid in pair for row in grid for cell in row})
    background_sets = [border_mode_colours(source) for source, _ in examples]
    backgrounds = sorted(set.intersection(*background_sets))
    if not backgrounds:
        backgrounds = sorted(set.union(*background_sets))
    predicates = [
        {"feature": "area", "symmetry": "identity"},
        {"feature": "bbox", "symmetry": "identity"},
        {"feature": "normalised_points", "symmetry": "identity"},
        {"feature": "normalised_points", "symmetry": "dihedral"},
    ]
    candidates: list[dict[str, Any]] = []
    for background_colour, frame_colour, boundary_mode, exclude_frame, colour_role, predicate, erase_source in itertools.product(
        backgrounds,
        colours,
        ("all", "any"),
        (True, False),
        ("any", "single_component"),
        predicates,
        (True, False),
    ):
        if background_colour == frame_colour:
            continue
        candidates.append(
            {
                "schema": "lexigen-arc-relational-ast-v1",
                "scene": {
                    "background_colour": background_colour,
                    "frame_colour": frame_colour,
                    "hole_boundary": boundary_mode,
                    "exclude_frame_objects": exclude_frame,
                    "object_colour_role": colour_role,
                },
                "match": predicate,
                "render": {
                    "op": "paint_destination_from_source_colour",
                    "erase_source": erase_source,
                    "preserve_unmatched": True,
                },
            }
        )
    return sorted(
        candidates,
        key=lambda ast: (
            ast_description_length(ast),
            hashlib.sha256(canonical_json(ast).encode("utf-8")).digest(),
        ),
    )


@dataclass(frozen=True)
class SynthesisResult:
    ast: dict[str, Any] | None
    candidates_tested: int
    exact_candidate_count: int
    exact_candidates: tuple[dict[str, Any], ...]
    ambiguous: bool


def synthesize_ast(examples: Sequence[tuple[Grid, Grid]]) -> SynthesisResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    exact: list[dict[str, Any]] = []
    tested = 0
    for candidate in candidate_asts(examples):
        tested += 1
        if all(execute_ast(candidate, source) == target for source, target in examples):
            exact.append(candidate)
    if not exact:
        return SynthesisResult(None, tested, 0, tuple(), False)
    minimum_length = min(ast_description_length(ast) for ast in exact)
    minimal = [ast for ast in exact if ast_description_length(ast) == minimum_length]
    semantic_keys = {
        (
            ast["scene"]["background_colour"],
            ast["scene"]["frame_colour"],
            ast["scene"]["hole_boundary"],
            ast["scene"]["exclude_frame_objects"],
            ast["scene"].get("object_colour_role"),
            ast["match"]["feature"],
            ast["match"].get("symmetry"),
            ast["render"]["erase_source"],
        )
        for ast in minimal
    }
    chosen = min(minimal, key=lambda ast: hashlib.sha256(canonical_json(ast).encode()).digest())
    return SynthesisResult(chosen, tested, len(exact), tuple(exact), len(semantic_keys) > 1)


def demonstrations_sha256(examples: Sequence[tuple[Grid, Grid]]) -> str:
    payload = [
        {"input": [list(row) for row in source], "output": [list(row) for row in target]}
        for source, target in examples
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_certificate(ast: dict[str, Any], examples: Sequence[tuple[Grid, Grid]], portable_agreement: bool) -> dict[str, Any]:
    ast_sha = hashlib.sha256(canonical_json(ast).encode("utf-8")).hexdigest()
    exact = all(execute_ast(ast, source) == target for source, target in examples)
    return {
        "schema": "lexigen-arc-semantic-certificate-v1",
        "ast_sha256": ast_sha,
        "demonstrations_sha256": demonstrations_sha256(examples),
        "demonstration_count": len(examples),
        "exact_reconstruction": exact,
        "portable_runtime_agreement": portable_agreement,
        "deterministic": all(execute_ast(ast, source) == execute_ast(ast, source) for source, _ in examples),
        "human_supplied_meta_grammar": True,
    }
