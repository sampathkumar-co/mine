from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from latent_runtime_v13 import (
    Grid,
    LatentRuntimeError,
    _bordered_squares,
    _row_bands,
    execute_program,
    most_common_colour,
)


@dataclass(frozen=True)
class LatentSynthesisResult:
    program: dict[str, Any] | None
    candidates_tested: int
    exact_candidate_count: int


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def colours(examples: Sequence[tuple[Grid, Grid]]) -> list[int]:
    return sorted({value for source, target in examples for grid in (source, target) for row in grid for value in row})


def program(operator: str, **parameters: Any) -> dict[str, Any]:
    return {
        "schema": "lexigen-latent-generator-v1",
        "operator": operator,
        "parameters": parameters,
    }


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _candidate_periodic_axis(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    for background in colours(examples):
        yield program("periodic_axis_field", background=background)


def _candidate_modular_lattice(examples):
    target_shapes = {_shape(target) for _, target in examples}
    if len(target_shapes) != 1:
        return
    output_height, output_width = next(iter(target_shapes))
    if not all(len(source) == len(source[0]) for source, _ in examples):
        return
    for background in colours(examples):
        yield program(
            "modular_separator_lattice",
            output_height=output_height,
            output_width=output_width,
            background=background,
        )


def _candidate_periodic_patch(examples):
    if not all(_shape(target)[0] <= _shape(source)[0] and _shape(target)[1] <= _shape(source)[1] for source, target in examples):
        return
    for background in colours(examples):
        yield program("recover_periodic_patch", background=background)


def _candidate_tile_broadcast(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    values = colours(examples)
    for separator, background in itertools.permutations(values, 2):
        yield program("broadcast_reference_tile", separator=separator, background=background)


def _candidate_legend(examples):
    if all(_shape(source) == _shape(target) and len(source) >= 2 and len(source[0]) >= 2 for source, target in examples):
        yield program("legend_palette_permutation", legend_rows=2, legend_cols=2)


def _candidate_replace(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    for old, new in itertools.permutations(colours(examples), 2):
        yield program("replace_colour", old=old, new=new)


def _candidate_marker_cross(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    for marker, background in itertools.permutations(colours(examples), 2):
        yield program("marker_parameterised_cross_shift", marker=marker, background=background)


def _candidate_tensor(examples):
    if not all(len(target) == len(source) * len(source) and len(target[0]) == len(source[0]) * len(source[0]) for source, target in examples):
        return
    for trigger, background in itertools.permutations(colours(examples), 2):
        yield program("masked_tensor_expansion", trigger=trigger, background=background)


def _infer_enclosure_mapping(examples, border_colour: int) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    found = False
    for source, target in examples:
        squares = _bordered_squares(source, border_colour)
        if not squares:
            return None
        for top, left, size in squares:
            thickness = (size - 3) // 2
            values = [
                target[row][col]
                for row in range(top + 1, top + size - 1)
                for col in range(left + 1, left + size - 1)
                if source[row][col] != border_colour
            ]
            if not values:
                continue
            fill = Counter(values).most_common(1)[0][0]
            if thickness in mapping and mapping[thickness] != fill:
                return None
            mapping[thickness] = fill
            found = True
    return mapping if found else None


def _candidate_enclosures(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    for border in colours(examples):
        mapping = _infer_enclosure_mapping(examples, border)
        if mapping:
            yield program(
                "enclosure_interior_classification",
                border_colour=border,
                thickness_to_colour={str(key): value for key, value in mapping.items()},
            )


def _candidate_component_summary(examples):
    target_shapes = {_shape(target) for _, target in examples}
    if len(target_shapes) != 1:
        return
    output_height, output_width = next(iter(target_shapes))
    for background in colours(examples):
        yield program(
            "ordered_component_summary",
            background=background,
            output_height=output_height,
            output_width=output_width,
        )


def _candidate_reflection(examples):
    if all(_shape(source) == _shape(target) for source, target in examples):
        yield program("horizontal_reflection")


def _candidate_seed_propagation(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    for background, mask in itertools.permutations(colours(examples), 2):
        yield program("component_seed_propagation", background=background, mask_colour=mask)


def _candidate_template_broadcast(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    for background, key in itertools.permutations(colours(examples), 2):
        yield program("template_colour_broadcast", background=background, key_colour=key)


def _infer_cycles(examples, background: int) -> dict[str, list[int]] | None:
    sequences: dict[int, list[list[int]]] = {}
    for source, target in examples:
        source_bands = _row_bands(source, background)
        if not source_bands:
            return None
        output_values = []
        for start, stop, _ in source_bands:
            values = [target[row][col] for row in range(start, stop) for col in range(len(target[0])) if target[row][col] != background]
            if not values:
                return None
            output_values.append(Counter(values).most_common(1)[0][0])
        sequences.setdefault(source_bands[0][2], []).append(output_values)
    cycles: dict[str, list[int]] = {}
    for first, observed in sequences.items():
        chosen = None
        for period in range(1, 9):
            slots: list[int | None] = [None] * period
            valid = True
            for sequence in observed:
                for index, value in enumerate(sequence):
                    slot = index % period
                    if slots[slot] is not None and slots[slot] != value:
                        valid = False
                        break
                    slots[slot] = value
                if not valid:
                    break
            if valid and all(value is not None for value in slots):
                chosen = [int(value) for value in slots]
                break
        if chosen is None:
            return None
        cycles[str(first)] = chosen
    return cycles


def _candidate_finite_state(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    for background in colours(examples):
        cycles = _infer_cycles(examples, background)
        if cycles:
            yield program("finite_state_component_recolour", background=background, cycles=cycles)


def _candidate_fill_rectangles(examples):
    if not all(_shape(source) == _shape(target) for source, target in examples):
        return
    for outline, fill in itertools.permutations(colours(examples), 2):
        yield program("fill_rectangular_interiors", outline=outline, fill=fill)


def candidate_programs(examples: Sequence[tuple[Grid, Grid]]) -> Iterable[dict[str, Any]]:
    generators = (
        _candidate_periodic_axis,
        _candidate_modular_lattice,
        _candidate_periodic_patch,
        _candidate_tile_broadcast,
        _candidate_legend,
        _candidate_replace,
        _candidate_marker_cross,
        _candidate_tensor,
        _candidate_enclosures,
        _candidate_component_summary,
        _candidate_reflection,
        _candidate_seed_propagation,
        _candidate_template_broadcast,
        _candidate_finite_state,
        _candidate_fill_rectangles,
    )
    for generator in generators:
        yield from generator(examples)


def description_length(candidate: dict[str, Any]) -> int:
    operator_cost = {
        "replace_colour": 1,
        "horizontal_reflection": 1,
        "periodic_axis_field": 2,
        "modular_separator_lattice": 2,
        "recover_periodic_patch": 3,
        "broadcast_reference_tile": 3,
        "legend_palette_permutation": 2,
        "marker_parameterised_cross_shift": 3,
        "masked_tensor_expansion": 3,
        "enclosure_interior_classification": 4,
        "ordered_component_summary": 3,
        "component_seed_propagation": 4,
        "template_colour_broadcast": 4,
        "finite_state_component_recolour": 5,
        "fill_rectangular_interiors": 3,
    }
    return 8 + operator_cost[candidate["operator"]] + len(canonical_json(candidate["parameters"])) // 16


def synthesize_latent(examples: Sequence[tuple[Grid, Grid]]) -> LatentSynthesisResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    exact = []
    tested = 0
    seen = set()
    for candidate in candidate_programs(examples):
        signature = canonical_json(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        tested += 1
        try:
            if all(execute_program(candidate, source) == target for source, target in examples):
                exact.append(candidate)
        except (LatentRuntimeError, ValueError, IndexError, KeyError):
            continue
    if not exact:
        return LatentSynthesisResult(None, tested, 0)
    chosen = min(exact, key=lambda item: (description_length(item), hashlib.sha256(canonical_json(item).encode()).digest()))
    digest = hashlib.sha256(canonical_json(chosen).encode()).hexdigest()
    result = dict(chosen)
    result["name"] = "generated_latent_program_" + digest[:12]
    result["provenance"] = {
        "method": "typed inverse latent-generator synthesis",
        "candidates_tested": tested,
        "exact_candidate_count": len(exact),
        "human_supplied_finished_task_operator": False,
        "human_supplied_generic_generator_substrate": True,
    }
    return LatentSynthesisResult(result, tested, len(exact))
