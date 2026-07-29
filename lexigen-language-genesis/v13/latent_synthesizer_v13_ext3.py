from __future__ import annotations

import hashlib
from typing import Any, Sequence

from latent_runtime_v13 import Grid, LatentRuntimeError
from latent_runtime_v13_ext3 import execute_program
from latent_synthesizer_v13 import LatentSynthesisResult, canonical_json, colours, program
from latent_synthesizer_v13_ext2 import candidate_programs as previous_candidates


def candidate_programs(examples: Sequence[tuple[Grid, Grid]]):
    yield from previous_candidates(examples)
    same_shape = all(
        (len(source), len(source[0])) == (len(target), len(target[0]))
        for source, target in examples
    )
    if same_shape:
        for key_colour in colours(examples):
            for index_stride in range(0, 5):
                yield program(
                    "indexed_legend_template_broadcast",
                    key_colour=key_colour,
                    index_stride=index_stride,
                )


def description_length(candidate: dict[str, Any]) -> int:
    costs = {
        "replace_colour": 1,
        "replace_colour_with_background": 2,
        "horizontal_reflection": 1,
        "reflect_component_positions": 3,
        "periodic_axis_field": 2,
        "modular_separator_lattice": 2,
        "recover_periodic_patch": 3,
        "broadcast_reference_tile": 3,
        "legend_palette_permutation": 2,
        "marker_parameterised_cross_shift": 3,
        "masked_tensor_expansion": 3,
        "enclosure_interior_classification": 4,
        "ordered_component_summary": 3,
        "ordered_colour_summary": 3,
        "component_seed_propagation": 4,
        "template_colour_broadcast": 4,
        "template_colour_broadcast_invariant": 5,
        "template_colour_broadcast_invariant_v2": 5,
        "indexed_legend_template_broadcast": 5,
        "finite_state_component_recolour": 5,
        "fill_rectangular_interiors": 3,
        "reconstruct_periodic_lattice": 5,
    }
    return 8 + costs.get(candidate["operator"], 8) + len(canonical_json(candidate["parameters"])) // 16


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
    chosen = min(
        exact,
        key=lambda item: (
            description_length(item),
            hashlib.sha256(canonical_json(item).encode()).digest(),
        ),
    )
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
