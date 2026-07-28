from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from arc_language import (
    ArcLanguageError,
    Grid,
    Program,
    apply_primitive as apply_v1,
    background,
    canonical_json,
    primitive_inventory as inventory_v1,
    shape,
)


def connect_aligned(grid: Grid, source: int, fill: int) -> Grid:
    """Connect every source-colour pair sharing a row or column.

    Endpoints are preserved. Interior background cells become `fill`; other
    coloured cells are never overwritten. This primitive was added only after
    ARC-GEN gate 1 had irreversibly failed and is ineligible as evidence there.
    """
    h, w = shape(grid)
    bg = background(grid)
    points = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == source]
    values = [list(row) for row in grid]
    for index, (r1, c1) in enumerate(points):
        for r2, c2 in points[index + 1 :]:
            if r1 == r2:
                for col in range(min(c1, c2) + 1, max(c1, c2)):
                    if values[r1][col] == bg:
                        values[r1][col] = fill
            elif c1 == c2:
                for row in range(min(r1, r2) + 1, max(r1, r2)):
                    if values[row][c1] == bg:
                        values[row][c1] = fill
    return tuple(tuple(row) for row in values)


def apply_primitive(grid: Grid, primitive: dict[str, Any]) -> Grid:
    if primitive["op"] == "connect_aligned":
        return connect_aligned(grid, int(primitive["source"]), int(primitive["fill"]))
    return apply_v1(grid, primitive)


def primitive_inventory(examples: Sequence[tuple[Grid, Grid]]) -> list[dict[str, Any]]:
    primitives = inventory_v1(examples)
    colours = sorted({value for pair in examples for grid in pair for row in grid for value in row})
    for source in colours:
        for fill in colours:
            if source != fill:
                primitives.append({"op": "connect_aligned", "source": source, "fill": fill})
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
