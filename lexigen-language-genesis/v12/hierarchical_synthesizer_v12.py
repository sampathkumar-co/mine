from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from hierarchical_runtime_v12 import (
    Grid,
    HierarchicalRuntimeError,
    canonical_json,
    execute_program,
)


@dataclass(frozen=True)
class HierarchicalSynthesisResult:
    program: dict[str, Any] | None
    candidates_tested: int
    exact_candidate_count: int


def _colours(examples: Sequence[tuple[Grid, Grid]]) -> list[int]:
    return sorted({value for pair in examples for grid in pair for row in grid for value in row})


def _full_lines(grid: Grid, colour: int) -> tuple[int, int]:
    rows = sum(all(value == colour for value in row) for row in grid)
    cols = sum(all(row[col] == colour for row in grid) for col in range(len(grid[0])))
    return rows, cols


def separator_candidates(examples: Sequence[tuple[Grid, Grid]]) -> list[int]:
    colours = _colours(examples)
    candidates = [
        colour
        for colour in colours
        if all(sum(_full_lines(source, colour)) > 0 for source, _ in examples)
    ]
    return candidates or colours


def blank_column_background_candidates(
    examples: Sequence[tuple[Grid, Grid]],
) -> list[int]:
    colours = _colours(examples)
    candidates = []
    for colour in colours:
        if all(
            any(all(row[col] == colour for row in source) for col in range(len(source[0])))
            for source, _ in examples
        ):
            candidates.append(colour)
    return candidates or colours


def marker_candidates(
    examples: Sequence[tuple[Grid, Grid]],
    background: int,
) -> list[int]:
    colours = _colours(examples)
    return [colour for colour in colours if colour != background]


def description_length(program: dict[str, Any]) -> int:
    partition, transform, assemble = program["stages"]
    costs = {
        "separator_lines": 1,
        "marker_gap_chain": 2,
        "complete_local_midpoints": 2,
        "reduce_mode": 1,
        "reduce_min": 3,
        "reduce_max": 3,
        "align_local_frames": 3,
        "preserve_canvas": 0,
        "summary_grid": 1,
        "concatenate_segments": 2,
        "min_shift": 1,
        "lexicographic_offsets": 2,
    }
    score = 12 + costs.get(partition["mode"], 0)
    score += costs.get(transform["mode"], 0) + costs.get(assemble["mode"], 0)
    score += costs.get(str(transform.get("rank_mode", "")), 0)
    score += int(bool(transform.get("require_same_colour", False)))
    score += int(assemble.get("border", 0))
    return score


def _completion_programs(examples: Sequence[tuple[Grid, Grid]]):
    colours = _colours(examples)
    for separator, background, same in itertools.product(
        separator_candidates(examples), colours, (True, False)
    ):
        if background == separator:
            continue
        yield {
            "schema": "lexigen-hierarchical-scene-v1",
            "types": ["Grid", "Containers", "LocalRelations", "Grid"],
            "stages": [
                {"kind": "partition", "mode": "separator_lines", "separator_colour": separator},
                {"kind": "transform", "mode": "complete_local_midpoints", "cell_background": background, "require_same_colour": same},
                {"kind": "assemble", "mode": "preserve_canvas"},
            ],
        }


def _reduction_programs(examples: Sequence[tuple[Grid, Grid]]):
    colours = _colours(examples)
    target_shapes = {(len(target), len(target[0])) for _, target in examples}
    for separator, reducer, border, canvas in itertools.product(
        separator_candidates(examples), ("mode", "min", "max"), range(3), colours
    ):
        if any(height <= 2 * border or width <= 2 * border for height, width in target_shapes):
            continue
        yield {
            "schema": "lexigen-hierarchical-scene-v1",
            "types": ["Grid", "Containers", "Statistics", "Grid"],
            "stages": [
                {"kind": "partition", "mode": "separator_lines", "separator_colour": separator},
                {"kind": "transform", "mode": f"reduce_{reducer}"},
                {"kind": "assemble", "mode": "summary_grid", "border": border, "canvas_colour": canvas},
            ],
        }


def _alignment_programs(examples: Sequence[tuple[Grid, Grid]]):
    for background in blank_column_background_candidates(examples):
        for marker, rank_mode in itertools.product(
            marker_candidates(examples, background),
            ("min_shift", "lexicographic_offsets"),
        ):
            yield {
                "schema": "lexigen-hierarchical-scene-v1",
                "types": ["Grid", "Segments", "LocalFrames", "Grid"],
                "stages": [
                    {"kind": "partition", "mode": "marker_gap_chain", "background": background, "marker": marker},
                    {"kind": "transform", "mode": "align_local_frames", "rank_mode": rank_mode},
                    {"kind": "assemble", "mode": "concatenate_segments"},
                ],
            }


def candidate_programs(examples: Sequence[tuple[Grid, Grid]]):
    same_shapes = all(
        (len(source), len(source[0])) == (len(target), len(target[0]))
        for source, target in examples
    )
    same_heights = all(len(source) == len(target) for source, target in examples)
    if same_shapes:
        yield from _completion_programs(examples)
    if not same_shapes:
        yield from _reduction_programs(examples)
    if same_heights and any(len(source[0]) > len(target[0]) for source, target in examples):
        yield from _alignment_programs(examples)


def synthesize_hierarchical(
    examples: Sequence[tuple[Grid, Grid]],
) -> HierarchicalSynthesisResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    exact: list[dict[str, Any]] = []
    tested = 0
    for program in candidate_programs(examples):
        tested += 1
        try:
            if all(execute_program(program, source) == target for source, target in examples):
                exact.append(program)
        except (HierarchicalRuntimeError, ValueError, IndexError, KeyError):
            continue
    if not exact:
        return HierarchicalSynthesisResult(None, tested, 0)
    chosen = min(
        exact,
        key=lambda program: (
            description_length(program),
            hashlib.sha256(canonical_json(program).encode()).digest(),
        ),
    )
    program = dict(chosen)
    digest = hashlib.sha256(canonical_json(chosen).encode()).hexdigest()
    program["name"] = "generated_hierarchical_program_" + digest[:12]
    program["provenance"] = {
        "method": "typed hierarchical scene grammar synthesis",
        "candidates_tested": tested,
        "exact_candidate_count": len(exact),
        "human_supplied_finished_task_operator": False,
        "human_supplied_generic_hierarchical_substrate": True,
    }
    return HierarchicalSynthesisResult(program, tested, len(exact))
