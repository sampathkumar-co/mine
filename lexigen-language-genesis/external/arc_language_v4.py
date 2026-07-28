from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from arc_language import ArcLanguageError, Grid, Program, background, canonical_json, shape
from arc_language_v3 import apply_primitive as apply_v3
from arc_language_v3 import primitive_inventory as inventory_v3


def trace_reflected_diagonal(
    grid: Grid,
    *,
    source: int,
    vertical_direction: int,
    horizontal_direction: int,
) -> Grid:
    """Trace a diagonal ray and reflect its horizontal direction at boundaries."""
    height, width = shape(grid)
    points = [(row, col) for row in range(height) for col in range(width) if grid[row][col] == source]
    if len(points) != 1:
        return grid
    row, col = points[0]
    dr = 1 if vertical_direction > 0 else -1
    dc = 1 if horizontal_direction > 0 else -1
    values = [list(line) for line in grid]
    bg = background(grid)

    while True:
        next_row = row + dr
        if next_row < 0 or next_row >= height:
            break
        next_col = col + dc
        if next_col < 0 or next_col >= width:
            dc *= -1
            next_col = col + dc
        row, col = next_row, next_col
        if values[row][col] == bg:
            values[row][col] = source
    return tuple(tuple(line) for line in values)


def apply_primitive(grid: Grid, primitive: dict[str, Any]) -> Grid:
    if primitive["op"] == "trace_reflected_diagonal":
        return trace_reflected_diagonal(
            grid,
            source=int(primitive["source"]),
            vertical_direction=int(primitive["vertical_direction"]),
            horizontal_direction=int(primitive["horizontal_direction"]),
        )
    return apply_v3(grid, primitive)


def primitive_inventory(examples: Sequence[tuple[Grid, Grid]]) -> list[dict[str, Any]]:
    primitives = inventory_v3(examples)
    input_colours = sorted({value for source, _ in examples for row in source for value in row})
    for source in input_colours:
        for vertical_direction in (-1, 1):
            for horizontal_direction in (-1, 1):
                primitives.append(
                    {
                        "op": "trace_reflected_diagonal",
                        "source": source,
                        "vertical_direction": vertical_direction,
                        "horizontal_direction": horizontal_direction,
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


def synthesize(
    examples: Sequence[tuple[Grid, Grid]], *, max_depth: int = 3, candidate_budget: int = 75_000
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
            frontier.items(), key=lambda item: hashlib.sha256(canonical_json(item[1]).encode()).digest()
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
                    return SynthesisResult(candidate, baseline, tested, len(visited), len(inventory))
                if transformed not in visited:
                    visited[transformed] = candidate
                    next_frontier[transformed] = candidate
        frontier = next_frontier
        if not frontier:
            break
    return SynthesisResult(solution, baseline, tested, len(visited), len(inventory))
