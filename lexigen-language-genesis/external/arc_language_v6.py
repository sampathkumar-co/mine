from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from arc_language import ArcLanguageError, Grid, Program, canonical_json, shape
from arc_language_v5 import apply_primitive as apply_v5
from arc_language_v5 import primitive_inventory as inventory_v5

Point = tuple[int, int]


def _components(grid: Grid, predicate) -> list[set[Point]]:
    h, w = shape(grid)
    unseen = {(r, c) for r in range(h) for c in range(w) if predicate(r, c, grid[r][c])}
    result: list[set[Point]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in unseen:
                    unseen.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
        result.append(component)
    return result


def _normalise(points: Iterable[Point]) -> frozenset[Point]:
    pts = list(points)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return frozenset((r - min_r, c - min_c) for r, c in pts)


def _variants(points: Iterable[Point]) -> set[frozenset[Point]]:
    base = list(_normalise(points))
    variants: set[frozenset[Point]] = set()
    current = base
    for _ in range(4):
        variants.add(_normalise(current))
        variants.add(_normalise((r, -c) for r, c in current))
        current = [(c, -r) for r, c in current]
    return variants


def transplant_matching_components_into_gray_holes(
    grid: Grid,
    *,
    gray: int = 5,
    allow_dihedral: bool = True,
) -> Grid:
    """Move coloured components into same-shaped holes enclosed by gray objects.

    This semantic primitive was diagnosed only after permanent failure on the
    untouched ARC-GEN task `228f6490`. It is ineligible for rescoring that gate.
    External coloured components are erased after their colour is transferred
    into a matching enclosed background component.
    """
    h, w = shape(grid)
    values = [list(row) for row in grid]

    background_components = _components(grid, lambda _r, _c, value: value == 0)
    holes = [
        component
        for component in background_components
        if all(r not in (0, h - 1) and c not in (0, w - 1) for r, c in component)
        and any(
            0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr][c + dc] == gray
            for r, c in component
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
    ]

    coloured_components: list[tuple[int, set[Point]]] = []
    for colour in sorted({value for row in grid for value in row} - {0, gray}):
        for component in _components(grid, lambda _r, _c, value, colour=colour: value == colour):
            coloured_components.append((colour, component))

    used_holes: set[int] = set()
    for colour, component in sorted(coloured_components, key=lambda item: (len(item[1]), item[0], min(item[1]))):
        source_shapes = _variants(component) if allow_dihedral else {_normalise(component)}
        matches = [
            index
            for index, hole in enumerate(holes)
            if index not in used_holes and _normalise(hole) in source_shapes
        ]
        if not matches:
            continue
        hole_index = min(matches, key=lambda index: min(holes[index]))
        for r, c in component:
            values[r][c] = 0
        for r, c in holes[hole_index]:
            values[r][c] = colour
        used_holes.add(hole_index)

    return tuple(tuple(row) for row in values)


def apply_primitive(grid: Grid, primitive: dict[str, Any]) -> Grid:
    if primitive["op"] == "transplant_matching_components_into_gray_holes":
        return transplant_matching_components_into_gray_holes(
            grid,
            gray=int(primitive["gray"]),
            allow_dihedral=bool(primitive["allow_dihedral"]),
        )
    return apply_v5(grid, primitive)


def primitive_inventory(examples: Sequence[tuple[Grid, Grid]]) -> list[dict[str, Any]]:
    primitives = inventory_v5(examples)
    colours = sorted({value for pair in examples for grid in pair for row in grid for value in row})
    for gray in colours:
        for allow_dihedral in (False, True):
            primitives.append(
                {
                    "op": "transplant_matching_components_into_gray_holes",
                    "gray": gray,
                    "allow_dihedral": allow_dihedral,
                }
            )
    unique = {canonical_json(primitive): primitive for primitive in primitives}
    return [unique[key] for key in sorted(unique, key=lambda text: hashlib.sha256(text.encode()).digest())]


@dataclass(frozen=True)
class SynthesisResult:
    program: Program | None
    baseline_program: Program | None
    candidates_tested: int
    signatures_seen: int
    inventory_size: int


def execute_program(program: Program, grid: Grid) -> Grid:
    current = grid
    for primitive in program:
        current = apply_primitive(current, primitive)
        if len(current) > 60 or len(current[0]) > 60:
            raise ArcLanguageError("intermediate grid exceeds frozen size budget")
    return current


def synthesize(examples: Sequence[tuple[Grid, Grid]], *, max_depth: int = 3, candidate_budget: int = 75_000) -> SynthesisResult:
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
        for signature, program in sorted(frontier.items(), key=lambda item: hashlib.sha256(canonical_json(item[1]).encode()).digest()):
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
                    return SynthesisResult(candidate, baseline, tested, len(visited), len(inventory))
                if transformed not in visited:
                    visited[transformed] = candidate
                    next_frontier[transformed] = candidate
        frontier = next_frontier
        if not frontier:
            break
    return SynthesisResult(solution, baseline, tested, len(visited), len(inventory))
