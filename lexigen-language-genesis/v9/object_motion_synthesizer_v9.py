from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Sequence

from object_motion_runtime_v9 import Grid, ObjectMotionError, canonical_json, execute_extension


def border_mode_colours(grid: Grid) -> set[int]:
    border = list(grid[0]) + list(grid[-1])
    border.extend(row[0] for row in grid[1:-1])
    border.extend(row[-1] for row in grid[1:-1])
    counts = {colour: border.count(colour) for colour in set(border)}
    maximum = max(counts.values())
    return {colour for colour, count in counts.items() if count == maximum}


def candidate_backgrounds(examples: Sequence[tuple[Grid, Grid]]) -> list[int]:
    sets = [border_mode_colours(source) for source, _ in examples]
    shared = set.intersection(*sets)
    return sorted(shared or set.union(*sets))


def disappearing_colours(examples: Sequence[tuple[Grid, Grid]]) -> list[int]:
    result = set()
    for source, target in examples:
        source_colours = {value for row in source for value in row}
        target_colours = {value for row in target for value in row}
        result.update(source_colours - target_colours)
    return sorted(result)


def extension_length(extension: dict[str, Any]) -> int:
    costs = {
        "inside_bbox": 1,
        "touches_component": 2,
        "inside_or_touches": 3,
        "bbox_fill": 1,
        "component_plus_marker": 2,
        "component_only": 1,
        "outward": 1,
        "inward": 1,
        "zero": 0,
    }
    return 10 + sum(
        costs[value]
        for value in (
            extension["association"]["mode"],
            extension["shape"]["mode"],
            extension["displacement"]["row"],
            extension["displacement"]["col"],
        )
    ) + int(extension["render"]["erase_source"])


@dataclass(frozen=True)
class ObjectMotionSynthesisResult:
    extension: dict[str, Any] | None
    candidates_tested: int
    exact_candidate_count: int


def candidate_extensions(examples: Sequence[tuple[Grid, Grid]]):
    colours = sorted({value for pair in examples for grid in pair for row in grid for value in row})
    backgrounds = candidate_backgrounds(examples)
    markers = disappearing_colours(examples) or colours
    for background, marker, association, shape_mode, row_mode, col_mode, erase_source in itertools.product(
        backgrounds,
        markers,
        ("inside_bbox", "touches_component", "inside_or_touches"),
        ("bbox_fill", "component_plus_marker", "component_only"),
        ("outward", "inward", "zero"),
        ("outward", "inward", "zero"),
        (True, False),
    ):
        if marker == background:
            continue
        yield {
            "schema": "lexigen-object-motion-extension-v1",
            "types": {
                "input": "Grid",
                "object": "ConnectedRegion",
                "marker": "Point",
                "completed_shape": "Set[Point]",
                "displacement": "Vector[Int,Int]",
                "output": "Grid",
            },
            "scene": {
                "background_colour": background,
                "marker_colour": marker,
            },
            "association": {
                "op": "select_unique_marker_by_relation",
                "mode": association,
            },
            "shape": {
                "op": "construct_point_set",
                "mode": shape_mode,
            },
            "displacement": {
                "op": "derive_axis_vector_from_marker_extreme",
                "row": row_mode,
                "col": col_mode,
            },
            "render": {
                "op": "translate_and_paint",
                "erase_source": erase_source,
            },
        }


def synthesize_object_motion(examples: Sequence[tuple[Grid, Grid]]) -> ObjectMotionSynthesisResult:
    exact: list[dict[str, Any]] = []
    tested = 0
    for extension in candidate_extensions(examples):
        tested += 1
        try:
            if all(execute_extension(extension, source) == target for source, target in examples):
                exact.append(extension)
        except (ObjectMotionError, ValueError, IndexError, KeyError):
            continue
    if not exact:
        return ObjectMotionSynthesisResult(None, tested, 0)
    chosen = min(
        exact,
        key=lambda extension: (
            extension_length(extension),
            hashlib.sha256(canonical_json(extension).encode()).digest(),
        ),
    )
    extension = dict(chosen)
    digest = hashlib.sha256(canonical_json(chosen).encode()).hexdigest()
    extension["name"] = "generated_motion_production_" + digest[:12]
    extension["provenance"] = {
        "method": "typed object-marker-vector grammar synthesis",
        "candidates_tested": tested,
        "exact_candidate_count": len(exact),
        "human_supplied_finished_task_operator": False,
        "human_supplied_generic_object_substrate": True,
    }
    return ObjectMotionSynthesisResult(extension, tested, len(exact))
