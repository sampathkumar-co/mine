from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Sequence

from compositional_runtime_v11 import (
    CompositionalRuntimeError,
    Grid,
    canonical_json,
    execute_pipeline,
)


def border_mode_colours(grid: Grid) -> list[int]:
    border = list(grid[0]) + list(grid[-1])
    border.extend(row[0] for row in grid[1:-1])
    border.extend(row[-1] for row in grid[1:-1])
    counts = {colour: border.count(colour) for colour in set(border)}
    maximum = max(counts.values())
    return sorted(colour for colour, count in counts.items() if count == maximum)


def background_candidates(examples: Sequence[tuple[Grid, Grid]]) -> list[int]:
    sets = [set(border_mode_colours(source)) for source, _ in examples]
    shared = set.intersection(*sets)
    return sorted(shared or set.union(*sets))


def output_shape_candidates(examples: Sequence[tuple[Grid, Grid]]) -> list[tuple[int, int]]:
    return sorted({(len(target), len(target[0])) for _, target in examples})


def description_length(program: dict[str, Any]) -> int:
    costs = {
        "compress_blank_axes": 1,
        "sample_regular_stride": 3,
        "equal_nonbackground": 1,
        "equal": 2,
        "both_nonbackground": 2,
        "horizontal_then_vertical_unclaimed": 1,
        "vertical_then_horizontal_unclaimed": 2,
        "all_edges": 0,
    }
    stages = program["stages"]
    return 16 + sum(
        costs.get(value, 0)
        for value in (
            stages[0]["mode"],
            stages[2]["predicate"],
            stages[3]["mode"],
        )
    ) + int(stages[4]["skip_background_tiles"])


@dataclass(frozen=True)
class PipelineSynthesisResult:
    program: dict[str, Any] | None
    candidates_tested: int
    exact_candidate_count: int


def candidate_programs(examples: Sequence[tuple[Grid, Grid]]):
    backgrounds = background_candidates(examples)
    output_shapes = output_shape_candidates(examples)
    all_colours = sorted({value for pair in examples for grid in pair for row in grid for value in row})
    for (
        background,
        extraction,
        output_shape,
        margin,
        gap,
        predicate,
        precedence,
        canvas_background,
        skip_background,
    ) in itertools.product(
        backgrounds,
        ("compress_blank_axes", "sample_regular_stride"),
        output_shapes,
        range(0, 4),
        range(0, 4),
        ("equal_nonbackground", "equal", "both_nonbackground"),
        ("all_edges", "horizontal_then_vertical_unclaimed", "vertical_then_horizontal_unclaimed"),
        all_colours,
        (True, False),
    ):
        extraction_stage: dict[str, Any] = {
            "kind": "extract_lattice",
            "mode": extraction,
            "background": background,
        }
        if extraction == "sample_regular_stride":
            extraction_stage.update(
                {
                    "row_offset": 1,
                    "col_offset": 1,
                    "row_stride": 2,
                    "col_stride": 2,
                }
            )
        yield {
            "schema": "lexigen-compositional-pipeline-v1",
            "types": ["Grid", "Lattice", "Relations", "Plan", "Grid"],
            "stages": [
                extraction_stage,
                {
                    "kind": "allocate_layout",
                    "output_height": output_shape[0],
                    "output_width": output_shape[1],
                    "margin": margin,
                    "gap": gap,
                },
                {
                    "kind": "build_relations",
                    "predicate": predicate,
                },
                {
                    "kind": "apply_precedence",
                    "mode": precedence,
                },
                {
                    "kind": "render",
                    "canvas_background": canvas_background,
                    "skip_background_tiles": skip_background,
                },
            ],
        }


def synthesize_pipeline(examples: Sequence[tuple[Grid, Grid]]) -> PipelineSynthesisResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    exact: list[dict[str, Any]] = []
    tested = 0
    for program in candidate_programs(examples):
        tested += 1
        try:
            if all(execute_pipeline(program, source) == target for source, target in examples):
                exact.append(program)
        except (CompositionalRuntimeError, ValueError, IndexError, KeyError):
            continue
    if not exact:
        return PipelineSynthesisResult(None, tested, 0)
    chosen = min(
        exact,
        key=lambda program: (
            description_length(program),
            hashlib.sha256(canonical_json(program).encode()).digest(),
        ),
    )
    program = dict(chosen)
    digest = hashlib.sha256(canonical_json(chosen).encode()).hexdigest()
    program["name"] = "generated_compositional_pipeline_" + digest[:12]
    program["provenance"] = {
        "method": "typed multi-stage pipeline synthesis",
        "candidates_tested": tested,
        "exact_candidate_count": len(exact),
        "human_supplied_finished_task_operator": False,
        "human_supplied_generic_stage_substrates": True,
    }
    return PipelineSynthesisResult(program, tested, len(exact))
