from __future__ import annotations

import hashlib
from typing import Sequence

from latent_runtime_v13 import Grid, LatentRuntimeError
from latent_runtime_v13_ext4 import execute_program
from latent_synthesizer_v13 import LatentSynthesisResult, canonical_json
from latent_synthesizer_v13_ext3 import candidate_programs, description_length

VALIDATED_OPERATORS = {
    "periodic_axis_field",
    "modular_separator_lattice",
    "recover_periodic_patch",
    "masked_tensor_expansion",
    "broadcast_reference_tile",
    "enclosure_interior_classification",
    "legend_palette_permutation",
    "ordered_colour_summary",
    "reflect_component_positions",
    "reconstruct_periodic_lattice",
    "marker_parameterised_cross_shift",
    "indexed_legend_template_broadcast",
    "component_seed_propagation",
    "finite_state_component_recolour",
}


def synthesize_latent_final(
    examples: Sequence[tuple[Grid, Grid]],
) -> LatentSynthesisResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    exact = []
    tested = 0
    seen = set()
    for candidate in candidate_programs(examples):
        if candidate["operator"] not in VALIDATED_OPERATORS:
            continue
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
        "language_inventory": "v13-fourteen-operator-frozen",
        "candidates_tested": tested,
        "exact_candidate_count": len(exact),
        "human_supplied_finished_task_operator": False,
        "human_supplied_generic_generator_substrate": True,
    }
    return LatentSynthesisResult(result, tested, len(exact))
