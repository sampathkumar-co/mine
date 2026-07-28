from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from arc_language import ArcLanguageError, Grid, Program, canonical_json, shape
from arc_language_v4 import apply_primitive as apply_v4
from arc_language_v4 import primitive_inventory as inventory_v4


def parse_top_left_legend(grid: Grid) -> tuple[list[tuple[int, int]], Grid]:
    """Parse consecutive `source, fill, 0...` rows and erase the legend."""
    values = [list(row) for row in grid]
    mappings: list[tuple[int, int]] = []
    row = 0
    while (
        row < len(values)
        and len(values[row]) >= 2
        and values[row][0] != 0
        and values[row][1] != 0
        and all(value == 0 for value in values[row][2:])
    ):
        mappings.append((values[row][0], values[row][1]))
        values[row][0] = 0
        values[row][1] = 0
        row += 1
    return mappings, tuple(tuple(line) for line in values)


def expand_legend_bounding_fields(grid: Grid, *, radius: int = 1) -> Grid:
    """Expand each legend-mapped source bounding box in legend order.

    Source locations are frozen after legend removal. Each expanded rectangle
    paints only currently empty cells, so earlier legend entries win overlaps.
    This also remains well-defined when a fill colour is a later source colour.
    """
    mappings, cleaned = parse_top_left_legend(grid)
    if not mappings:
        return grid
    height, width = shape(cleaned)
    original = cleaned
    output = [list(row) for row in cleaned]

    frozen_points = {
        source: [
            (row, col)
            for row in range(height)
            for col in range(width)
            if original[row][col] == source
        ]
        for source, _ in mappings
    }

    for source, fill in mappings:
        points = frozen_points[source]
        if not points:
            continue
        rows = [row for row, _ in points]
        cols = [col for _, col in points]
        row_start = max(0, min(rows) - radius)
        row_end = min(height - 1, max(rows) + radius)
        col_start = max(0, min(cols) - radius)
        col_end = min(width - 1, max(cols) + radius)
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                if output[row][col] == 0:
                    output[row][col] = fill
    return tuple(tuple(row) for row in output)


def apply_primitive(grid: Grid, primitive: dict[str, Any]) -> Grid:
    if primitive["op"] == "expand_legend_bounding_fields":
        return expand_legend_bounding_fields(grid, radius=int(primitive["radius"]))
    return apply_v4(grid, primitive)


def primitive_inventory(examples: Sequence[tuple[Grid, Grid]]) -> list[dict[str, Any]]:
    primitives = inventory_v4(examples)
    primitives.extend(
        {"op": "expand_legend_bounding_fields", "radius": radius}
        for radius in (1, 2)
    )
    unique = {canonical_json(primitive): primitive for primitive in primitives}
    return [
        unique[key]
        for key in sorted(unique, key=lambda text: hashlib.sha256(text.encode()).digest())
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
            raise ArcLanguageError("intermediate grid exceeds frozen size budget")
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
                    return SynthesisResult(candidate, baseline, tested, len(visited), len(inventory))
                if transformed not in visited:
                    visited[transformed] = candidate
                    next_frontier[transformed] = candidate
        frontier = next_frontier
        if not frontier:
            break
    return SynthesisResult(solution, baseline, tested, len(visited), len(inventory))
