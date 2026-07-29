from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from scene_runtime_v14 import Grid, SceneRuntimeError, background, execute_stage, shape

Pipeline = tuple[dict[str, Any], ...]
Signature = tuple[Grid, ...]
TRANSFORMS = ("identity", "flip_h", "flip_v", "transpose", "rotate_180")


@dataclass(frozen=True)
class SceneSynthesisResult:
    pipeline: Pipeline | None
    candidates_tested: int
    signatures_seen: int
    inventory_size: int
    exact_pipeline_count: int


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def colours(grids: Iterable[Grid]) -> list[int]:
    return sorted({cell for grid in grids for row in grid for cell in row})


def infer_recolour(examples: Sequence[tuple[Grid, Grid]]) -> dict[str, Any] | None:
    if any(shape(source) != shape(target) for source, target in examples):
        return None
    mapping: dict[int, int] = {}
    for source_colour in colours(source for source, _ in examples):
        observed = {
            target[r][c]
            for source, target in examples
            for r in range(len(source))
            for c in range(len(source[0]))
            if source[r][c] == source_colour
        }
        if len(observed) != 1:
            return None
        target_colour = next(iter(observed))
        if target_colour != source_colour:
            mapping[source_colour] = target_colour
    if not mapping:
        return None
    return {
        "op": "recolour",
        "mapping": {str(key): value for key, value in sorted(mapping.items())},
    }


def singleton_colours(grid: Grid) -> set[int]:
    counts: dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return {value for value, count in counts.items() if count == 1 and value != background(grid)}


def candidate_stages(examples: Sequence[tuple[Grid, Grid]]) -> tuple[dict[str, Any], ...]:
    stages: list[dict[str, Any]] = []
    recolour = infer_recolour(examples)
    if recolour is not None:
        stages.append(recolour)

    singleton_sets = [singleton_colours(source) for source, _ in examples]
    common_singletons = set.intersection(*singleton_sets) if singleton_sets else set()
    for source_colour in sorted(common_singletons):
        for target_colour in sorted(common_singletons - {source_colour}):
            stages.append({
                "op": "move_singleton_towards",
                "source_colour": source_colour,
                "target_colour": target_colour,
            })

    if all(
        shape(target) == (shape(source)[0] + 2, shape(source)[1] + 2)
        for source, target in examples
    ):
        corner_values = {
            value
            for _, target in examples
            for value in (target[0][0], target[0][-1], target[-1][0], target[-1][-1])
        }
        for fill_colour in sorted(corner_values):
            stages.append({"op": "edge_project", "radius": 1, "fill_colour": fill_colour})

    all_colours = colours(grid for pair in examples for grid in pair)
    for line_colour in [*all_colours, "structural"]:
        for transform_name in TRANSFORMS:
            stages.append({
                "op": "decode_regular_linegrid",
                "line_colour": line_colour,
                "transform": transform_name,
            })

    first_source, first_target = examples[0]
    in_height, in_width = shape(first_source)
    out_height, out_width = shape(first_target)
    if out_height and out_width and in_height % out_height == 0 and in_width % out_width == 0:
        tile_rows = in_height // out_height
        tile_cols = in_width // out_width
        tile_count = tile_rows * tile_cols
        if 2 <= tile_count <= 6 and all(
            shape(source) == (out_height * tile_rows, out_width * tile_cols)
            and shape(target) == (out_height, out_width)
            for source, target in examples
        ):
            for order in itertools.permutations(range(tile_count)):
                stages.append({
                    "op": "overlay_equal_tiles",
                    "tile_rows": tile_rows,
                    "tile_cols": tile_cols,
                    "order": list(order),
                })

    for object_mode in ("components", "colours"):
        stages.append({"op": "canonical_rectangular_layers", "object_mode": object_mode})

    for fill_colour in colours(target for _, target in examples):
        stages.append({"op": "fill_internal_blank_axis", "fill_colour": fill_colour})

    stages.append({"op": "extend_corner_marked_rays"})
    for transform_name in TRANSFORMS[1:]:
        stages.append({"op": "transform", "name": transform_name})

    unique = {canonical(stage): stage for stage in stages}
    return tuple(
        unique[key]
        for key in sorted(unique, key=lambda text: hashlib.sha256(text.encode()).digest())
    )


def stage_cost(stage: dict[str, Any]) -> int:
    costs = {
        "recolour": 2,
        "move_singleton_towards": 4,
        "edge_project": 3,
        "decode_regular_linegrid": 5,
        "overlay_equal_tiles": 5,
        "canonical_rectangular_layers": 6,
        "fill_internal_blank_axis": 4,
        "extend_corner_marked_rays": 5,
        "transform": 1,
    }
    return costs.get(stage["op"], 8) + len(canonical(stage)) // 24


def pipeline_key(pipeline: Pipeline) -> tuple[int, bytes]:
    text = canonical(list(pipeline))
    return (
        sum(stage_cost(stage) for stage in pipeline),
        hashlib.sha256(text.encode()).digest(),
    )


def synthesize_scene(
    examples: Sequence[tuple[Grid, Grid]],
    *,
    max_depth: int = 2,
    candidate_budget: int = 200_000,
) -> SceneSynthesisResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    inputs: Signature = tuple(source for source, _ in examples)
    targets: Signature = tuple(target for _, target in examples)
    inventory = candidate_stages(examples)
    tested = 0
    visited: dict[Signature, Pipeline] = {inputs: tuple()}
    frontier: dict[Signature, Pipeline] = {inputs: tuple()}

    for _depth in range(1, max_depth + 1):
        exact: list[Pipeline] = []
        next_frontier: dict[Signature, Pipeline] = {}
        for signature, pipeline in sorted(frontier.items(), key=lambda item: pipeline_key(item[1])):
            for stage in inventory:
                tested += 1
                if tested > candidate_budget:
                    chosen = min(exact, key=pipeline_key) if exact else None
                    return SceneSynthesisResult(
                        chosen,
                        tested - 1,
                        len(visited),
                        len(inventory),
                        len(exact),
                    )
                try:
                    transformed = tuple(execute_stage(stage, grid) for grid in signature)
                except (SceneRuntimeError, ValueError, IndexError, KeyError):
                    continue
                candidate = pipeline + (stage,)
                if transformed == targets:
                    exact.append(candidate)
                    continue
                previous = visited.get(transformed)
                if previous is None or pipeline_key(candidate) < pipeline_key(previous):
                    visited[transformed] = candidate
                    next_frontier[transformed] = candidate
        if exact:
            return SceneSynthesisResult(
                min(exact, key=pipeline_key),
                tested,
                len(visited),
                len(inventory),
                len(exact),
            )
        frontier = next_frontier
        if not frontier:
            break
    return SceneSynthesisResult(None, tested, len(visited), len(inventory), 0)
