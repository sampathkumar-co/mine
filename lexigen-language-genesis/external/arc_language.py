from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

Grid = tuple[tuple[int, ...], ...]
Program = tuple[dict[str, Any], ...]


class ArcLanguageError(RuntimeError):
    pass


def as_grid(value: Sequence[Sequence[int]]) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ArcLanguageError("grid must be a non-empty rectangle")
    if any(cell < 0 or cell > 9 for row in grid for cell in row):
        raise ArcLanguageError("ARC colours must be integers from 0 through 9")
    return grid


def to_json_grid(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def background(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda colour: (-counts[colour], colour))


def rotate90(grid: Grid) -> Grid:
    return tuple(tuple(row[col] for row in reversed(grid)) for col in range(len(grid[0])))


def transpose(grid: Grid) -> Grid:
    return tuple(tuple(grid[row][col] for row in range(len(grid))) for col in range(len(grid[0])))


def crop_bounds(grid: Grid, cells: list[tuple[int, int]]) -> Grid:
    if not cells:
        return grid
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return tuple(tuple(grid[row][col] for col in range(c0, c1 + 1)) for row in range(r0, r1 + 1))


def component_cells(grid: Grid, *, colour: int | None, diagonal: bool) -> list[list[tuple[int, int]]]:
    h, w = shape(grid)
    bg = background(grid)
    allowed = {
        (row, col)
        for row in range(h)
        for col in range(w)
        if (grid[row][col] != bg if colour is None else grid[row][col] == colour)
    }
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        steps += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    components: list[list[tuple[int, int]]] = []
    while allowed:
        start = min(allowed)
        allowed.remove(start)
        queue = deque([start])
        component = [start]
        while queue:
            row, col = queue.popleft()
            for dr, dc in steps:
                nxt = (row + dr, col + dc)
                if nxt in allowed:
                    allowed.remove(nxt)
                    queue.append(nxt)
                    component.append(nxt)
        components.append(sorted(component))
    return components


def apply_primitive(grid: Grid, primitive: dict[str, Any]) -> Grid:
    op = primitive["op"]
    if op == "identity":
        return grid
    if op == "rot90":
        return rotate90(grid)
    if op == "rot180":
        return rotate90(rotate90(grid))
    if op == "rot270":
        return rotate90(rotate90(rotate90(grid)))
    if op == "flip_h":
        return tuple(tuple(reversed(row)) for row in grid)
    if op == "flip_v":
        return tuple(reversed(grid))
    if op == "transpose":
        return transpose(grid)
    if op == "anti_transpose":
        return rotate90(rotate90(transpose(grid)))
    if op == "crop_nonbackground":
        bg = background(grid)
        cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value != bg]
        return crop_bounds(grid, cells)
    if op == "crop_colour":
        colour = int(primitive["colour"])
        cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == colour]
        return crop_bounds(grid, cells)
    if op == "compress_blank_rows":
        bg = background(grid)
        rows = [row for row in grid if any(value != bg for value in row)]
        return tuple(rows) if rows else grid
    if op == "compress_blank_cols":
        bg = background(grid)
        keep = [col for col in range(len(grid[0])) if any(grid[row][col] != bg for row in range(len(grid)))]
        return tuple(tuple(row[col] for col in keep) for row in grid) if keep else grid
    if op == "recolour_nonbackground":
        target = int(primitive["target"])
        bg = background(grid)
        return tuple(tuple(bg if value == bg else target for value in row) for row in grid)
    if op == "replace_colour":
        source, target = int(primitive["source"]), int(primitive["target"])
        return tuple(tuple(target if value == source else value for value in row) for row in grid)
    if op == "swap_colours":
        first, second = int(primitive["first"]), int(primitive["second"])
        return tuple(
            tuple(second if value == first else first if value == second else value for value in row)
            for row in grid
        )
    if op == "map_colours":
        mapping = {int(key): int(value) for key, value in primitive["mapping"].items()}
        return tuple(tuple(mapping.get(value, value) for value in row) for row in grid)
    if op == "upscale":
        factor = int(primitive["factor"])
        return tuple(
            tuple(value for value in row for _ in range(factor))
            for row in grid
            for _ in range(factor)
        )
    if op == "tile_h":
        count = int(primitive["count"])
        return tuple(tuple(row) * count for row in grid)
    if op == "tile_v":
        count = int(primitive["count"])
        return tuple(grid for _ in range(count)) if False else tuple(row for _ in range(count) for row in grid)
    if op == "isolate_colour":
        colour = int(primitive["colour"])
        bg = background(grid)
        return tuple(tuple(value if value == colour else bg for value in row) for row in grid)
    if op == "fill_bbox":
        colour = int(primitive["colour"])
        bg = background(grid)
        cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value != bg]
        if not cells:
            return grid
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        return tuple(
            tuple(colour if r0 <= r <= r1 and c0 <= c <= c1 else grid[r][c] for c in range(len(grid[0])))
            for r in range(len(grid))
        )
    if op == "largest_component_crop":
        components = component_cells(
            grid,
            colour=primitive.get("colour"),
            diagonal=bool(primitive.get("diagonal", False)),
        )
        if not components:
            return grid
        component = min(components, key=lambda cells: (-len(cells), cells))
        return crop_bounds(grid, component)
    if op == "smallest_component_crop":
        components = component_cells(
            grid,
            colour=primitive.get("colour"),
            diagonal=bool(primitive.get("diagonal", False)),
        )
        if not components:
            return grid
        component = min(components, key=lambda cells: (len(cells), cells))
        return crop_bounds(grid, component)
    raise ArcLanguageError(f"unknown primitive: {op}")


def execute_program(program: Program, grid: Grid) -> Grid:
    current = grid
    for primitive in program:
        current = apply_primitive(current, primitive)
        if len(current) > 60 or len(current[0]) > 60:
            raise ArcLanguageError("intermediate grid exceeds frozen size budget")
    return current


def infer_colour_map(examples: Sequence[tuple[Grid, Grid]]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for source, target in examples:
        if shape(source) != shape(target):
            return None
        for source_row, target_row in zip(source, target):
            for before, after in zip(source_row, target_row):
                prior = mapping.setdefault(before, after)
                if prior != after:
                    return None
    return mapping


def primitive_inventory(examples: Sequence[tuple[Grid, Grid]]) -> list[dict[str, Any]]:
    colours = sorted({value for pair in examples for grid in pair for row in grid for value in row})
    primitives: list[dict[str, Any]] = [
        {"op": "identity"},
        {"op": "rot90"},
        {"op": "rot180"},
        {"op": "rot270"},
        {"op": "flip_h"},
        {"op": "flip_v"},
        {"op": "transpose"},
        {"op": "anti_transpose"},
        {"op": "crop_nonbackground"},
        {"op": "compress_blank_rows"},
        {"op": "compress_blank_cols"},
    ]
    for colour in colours:
        primitives.extend(
            [
                {"op": "crop_colour", "colour": colour},
                {"op": "isolate_colour", "colour": colour},
                {"op": "recolour_nonbackground", "target": colour},
                {"op": "fill_bbox", "colour": colour},
                {"op": "largest_component_crop", "colour": colour, "diagonal": False},
                {"op": "largest_component_crop", "colour": colour, "diagonal": True},
                {"op": "smallest_component_crop", "colour": colour, "diagonal": False},
                {"op": "smallest_component_crop", "colour": colour, "diagonal": True},
            ]
        )
    primitives.extend(
        [
            {"op": "largest_component_crop", "colour": None, "diagonal": False},
            {"op": "largest_component_crop", "colour": None, "diagonal": True},
            {"op": "smallest_component_crop", "colour": None, "diagonal": False},
            {"op": "smallest_component_crop", "colour": None, "diagonal": True},
        ]
    )
    for source, target in itertools.permutations(colours, 2):
        primitives.append({"op": "replace_colour", "source": source, "target": target})
    for first, second in itertools.combinations(colours, 2):
        primitives.append({"op": "swap_colours", "first": first, "second": second})
    for factor in (2, 3, 4):
        primitives.extend(
            [
                {"op": "upscale", "factor": factor},
                {"op": "tile_h", "count": factor},
                {"op": "tile_v", "count": factor},
            ]
        )
    inferred = infer_colour_map(examples)
    if inferred is not None and any(key != value for key, value in inferred.items()):
        primitives.append({"op": "map_colours", "mapping": {str(k): v for k, v in sorted(inferred.items())}})
    unique = {canonical_json(primitive): primitive for primitive in primitives}
    return [unique[key] for key in sorted(unique, key=lambda text: hashlib.sha256(text.encode()).digest())]


@dataclass(frozen=True)
class SynthesisResult:
    program: Program | None
    baseline_program: Program | None
    candidates_tested: int
    signatures_seen: int
    inventory_size: int


def synthesize(
    examples: Sequence[tuple[Grid, Grid]],
    *,
    max_depth: int = 3,
    candidate_budget: int = 75_000,
) -> SynthesisResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    inputs = tuple(source for source, _ in examples)
    targets = tuple(target for _, target in examples)
    inventory = primitive_inventory(examples)
    baseline: Program | None = None
    tested = 0

    frontier: dict[tuple[Grid, ...], Program] = {inputs: tuple()}
    visited: dict[tuple[Grid, ...], Program] = {inputs: tuple()}
    solution: Program | None = tuple() if inputs == targets else None

    for depth in range(1, max_depth + 1):
        next_frontier: dict[tuple[Grid, ...], Program] = {}
        for signature, program in sorted(
            frontier.items(),
            key=lambda item: hashlib.sha256(canonical_json(item[1]).encode()).digest(),
        ):
            for primitive in inventory:
                tested += 1
                if tested > candidate_budget:
                    return SynthesisResult(solution, baseline, tested - 1, len(visited), len(inventory))
                try:
                    transformed = tuple(apply_primitive(grid, primitive) for grid in signature)
                except (ArcLanguageError, ValueError, IndexError):
                    continue
                candidate = program + (primitive,)
                if depth == 1 and transformed == targets and baseline is None:
                    baseline = candidate
                if transformed == targets:
                    solution = candidate
                    return SynthesisResult(solution, baseline, tested, len(visited), len(inventory))
                if transformed not in visited:
                    visited[transformed] = candidate
                    next_frontier[transformed] = candidate
        frontier = next_frontier
        if not frontier:
            break
    return SynthesisResult(solution, baseline, tested, len(visited), len(inventory))


def language_artifact(program: Program, examples: Sequence[tuple[Grid, Grid]]) -> dict[str, Any]:
    semantics = [dict(primitive) for primitive in program]
    evidence = [
        {"input": to_json_grid(source), "output": to_json_grid(target)}
        for source, target in examples
    ]
    semantics_bytes = canonical_json(semantics).encode()
    evidence_bytes = canonical_json(evidence).encode()
    return {
        "schema": "lexigen-arc-language-artifact-v1",
        "name": "arc_macro_" + hashlib.sha256(semantics_bytes).hexdigest()[:12],
        "signature": "Grid -> Grid",
        "operational_semantics": semantics,
        "semantics_sha256": hashlib.sha256(semantics_bytes).hexdigest(),
        "demonstration_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "portable": True,
    }
