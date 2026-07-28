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
    height, width = shape(grid)
    unseen = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if predicate(row, col, grid[row][col])
    }
    result: list[set[Point]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = row + delta_row, col + delta_col
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        result.append(component)
    return sorted(result, key=lambda component: (min(component), len(component)))


def _normalise(points: Iterable[Point]) -> frozenset[Point]:
    points_list = list(points)
    minimum_row = min(row for row, _ in points_list)
    minimum_col = min(col for _, col in points_list)
    return frozenset(
        (row - minimum_row, col - minimum_col)
        for row, col in points_list
    )


def _variants(points: Iterable[Point]) -> set[frozenset[Point]]:
    base = list(_normalise(points))
    variants: set[frozenset[Point]] = set()
    current = base
    for _ in range(4):
        variants.add(_normalise(current))
        variants.add(_normalise((row, -col) for row, col in current))
        current = [(col, -row) for row, col in current]
    return variants


def _orthogonal_neighbours(point: Point, height: int, width: int) -> Iterable[Point]:
    row, col = point
    for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbour = row + delta_row, col + delta_col
        if 0 <= neighbour[0] < height and 0 <= neighbour[1] < width:
            yield neighbour


def transplant_matching_components_into_gray_holes(
    grid: Grid,
    *,
    gray: int = 5,
    allow_dihedral: bool = False,
) -> Grid:
    """Move coloured components into same-shaped holes enclosed by one frame colour.

    A valid hole is a background component that does not touch the grid boundary
    and whose entire non-background orthogonal boundary is exactly `gray`.
    Identity matching is the conservative default. Rotation/reflection remains
    executable but is not offered to the frozen synthesizer until external
    evidence requires that extra invariance.
    """
    if gray == 0:
        raise ArcLanguageError("the enclosing frame colour must be non-background")

    height, width = shape(grid)
    values = [list(row) for row in grid]
    background_components = _components(
        grid,
        lambda _row, _col, value: value == 0,
    )

    holes: list[set[Point]] = []
    for component in background_components:
        if any(
            row in (0, height - 1) or col in (0, width - 1)
            for row, col in component
        ):
            continue
        boundary_colours = {
            grid[neighbour_row][neighbour_col]
            for point in component
            for neighbour_row, neighbour_col in _orthogonal_neighbours(
                point,
                height,
                width,
            )
            if (neighbour_row, neighbour_col) not in component
            and grid[neighbour_row][neighbour_col] != 0
        }
        if boundary_colours == {gray}:
            holes.append(component)

    coloured_components: list[tuple[int, set[Point]]] = []
    for colour in sorted({value for row in grid for value in row} - {0, gray}):
        for component in _components(
            grid,
            lambda _row, _col, value, colour=colour: value == colour,
        ):
            coloured_components.append((colour, component))

    used_holes: set[int] = set()
    for colour, component in sorted(
        coloured_components,
        key=lambda item: (len(item[1]), item[0], min(item[1])),
    ):
        source_shapes = (
            _variants(component)
            if allow_dihedral
            else {_normalise(component)}
        )
        matches = [
            index
            for index, hole in enumerate(holes)
            if index not in used_holes
            and _normalise(hole) in source_shapes
        ]
        if not matches:
            continue
        hole_index = min(matches, key=lambda index: min(holes[index]))
        for row, col in component:
            values[row][col] = 0
        for row, col in holes[hole_index]:
            values[row][col] = colour
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


def primitive_inventory(
    examples: Sequence[tuple[Grid, Grid]],
) -> list[dict[str, Any]]:
    primitives = inventory_v5(examples)
    non_background_colours = sorted(
        {
            value
            for pair in examples
            for grid in pair
            for row in grid
            for value in row
            if value != 0
        }
    )
    for gray in non_background_colours:
        primitives.append(
            {
                "op": "transplant_matching_components_into_gray_holes",
                "gray": gray,
                "allow_dihedral": False,
            }
        )
    unique = {
        canonical_json(primitive): primitive
        for primitive in primitives
    }
    return [
        unique[key]
        for key in sorted(
            unique,
            key=lambda text: hashlib.sha256(text.encode()).digest(),
        )
    ]


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
            raise ArcLanguageError(
                "intermediate grid exceeds frozen size budget"
            )
    return current


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
            key=lambda item: hashlib.sha256(
                canonical_json(item[1]).encode()
            ).digest(),
        ):
            for primitive in inventory:
                tested += 1
                if tested > candidate_budget:
                    return SynthesisResult(
                        solution,
                        baseline,
                        tested - 1,
                        len(visited),
                        len(inventory),
                    )
                try:
                    transformed = tuple(
                        apply_primitive(grid, primitive)
                        for grid in signature
                    )
                except (ArcLanguageError, ValueError, IndexError):
                    continue
                candidate = program + (primitive,)
                if depth == 1 and transformed == targets and baseline is None:
                    baseline = candidate
                if transformed == targets:
                    return SynthesisResult(
                        candidate,
                        baseline,
                        tested,
                        len(visited),
                        len(inventory),
                    )
                if transformed not in visited:
                    visited[transformed] = candidate
                    next_frontier[transformed] = candidate
        frontier = next_frontier
        if not frontier:
            break
    return SynthesisResult(
        solution,
        baseline,
        tested,
        len(visited),
        len(inventory),
    )
