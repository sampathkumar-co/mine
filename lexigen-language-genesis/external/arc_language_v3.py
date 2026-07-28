from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from arc_language import ArcLanguageError, Grid, Program, canonical_json, shape
from arc_language_v2 import apply_primitive as apply_v2
from arc_language_v2 import primitive_inventory as inventory_v2


def connect_anchor_spine(grid: Grid, source: int, fill: int) -> Grid:
    """Connect anchors through the unique aligned-pair spine.

    The longest source-colour pair sharing a row or column defines a backbone.
    Remaining source anchors are projected perpendicularly onto that backbone.
    Source endpoints and non-background coloured cells are preserved.
    """
    height, width = shape(grid)
    points = [(row, col) for row in range(height) for col in range(width) if grid[row][col] == source]
    aligned: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for index, first in enumerate(points):
        for second in points[index + 1 :]:
            if first[0] == second[0]:
                aligned.append((abs(first[1] - second[1]), first, second))
            elif first[1] == second[1]:
                aligned.append((abs(first[0] - second[0]), first, second))
    if not aligned:
        return grid
    _, first, second = max(aligned, key=lambda item: (item[0], tuple(sorted((item[1], item[2])))))
    values = [list(row) for row in grid]

    def paint(row: int, col: int) -> None:
        if values[row][col] == 0:
            values[row][col] = fill

    if first[0] == second[0]:
        spine_row = first[0]
        left, right = sorted((first[1], second[1]))
        for col in range(left + 1, right):
            paint(spine_row, col)
        spine_points = {first, second}
        for row, col in points:
            if (row, col) in spine_points:
                continue
            start, end = sorted((row, spine_row))
            for current_row in range(start + 1, end):
                paint(current_row, col)
            paint(spine_row, col)
    else:
        spine_col = first[1]
        top, bottom = sorted((first[0], second[0]))
        for row in range(top + 1, bottom):
            paint(row, spine_col)
        spine_points = {first, second}
        for row, col in points:
            if (row, col) in spine_points:
                continue
            start, end = sorted((col, spine_col))
            for current_col in range(start + 1, end):
                paint(row, current_col)
            paint(row, spine_col)
    return tuple(tuple(row) for row in values)


def apply_primitive(grid: Grid, primitive: dict[str, Any]) -> Grid:
    if primitive["op"] == "connect_anchor_spine":
        return connect_anchor_spine(grid, int(primitive["source"]), int(primitive["fill"]))
    return apply_v2(grid, primitive)


def primitive_inventory(examples: Sequence[tuple[Grid, Grid]]) -> list[dict[str, Any]]:
    primitives = inventory_v2(examples)
    input_colours = sorted({value for source, _ in examples for row in source for value in row})
    output_colours = sorted({value for _, target in examples for row in target for value in row})
    introduced = [colour for colour in output_colours if colour not in input_colours]
    fill_colours = introduced or output_colours
    for source in input_colours:
        for fill in fill_colours:
            if source != fill:
                primitives.append({"op": "connect_anchor_spine", "source": source, "fill": fill})
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
