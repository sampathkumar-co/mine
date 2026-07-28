from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .intervention_genesis_v22 import (
    information_lower_bound,
    intervention_response,
    optimal_worst_case_queries,
    observational_equivalence_certificate,
    synthesize_position_rule,
)
from .outcome_alphabet_v23 import (
    ACCEPT,
    ACTION_NAMES,
    FEATURE_NAMES,
    LEFT,
    RIGHT,
    ClosedLoopEvaluation,
    OutcomeProgram,
    decoder_digest,
    evaluate_program,
    hand_program,
    random_codebook_control,
)


@dataclass(frozen=True)
class SemanticState:
    name: str
    left_exists: bool
    right_exists: bool
    left_active: bool
    right_active: bool
    left_greater: bool
    action: int
    derivation: str

    def values(self) -> dict[str, bool]:
        return {
            "left_exists": self.left_exists,
            "right_exists": self.right_exists,
            "left_active": self.left_active,
            "right_active": self.right_active,
            "left_greater": self.left_greater,
        }


@dataclass(frozen=True)
class ResponseExample:
    state: str
    left: float | None
    right: float | None
    action: int


@dataclass(frozen=True)
class ClassificationEvaluation:
    accuracy: float
    macro_accuracy: float
    minimum_state_accuracy: float
    unseen_code_rate: float
    per_state_accuracy: dict[str, float]


def canonical_semantic_universe() -> tuple[SemanticState, ...]:
    """All response states reachable under a lower-midpoint query.

    While an interval contains at least two candidates, its lower midpoint is
    strictly smaller than the interval high endpoint. Therefore the queried
    node always has a right neighbour in the original chain. The only queried
    node without a left neighbour is node zero. These invariants reduce every
    reachable noiseless response to the five cases below.
    """
    return (
        SemanticState(
            "left_boundary_equal",
            False,
            True,
            False,
            True,
            False,
            ACCEPT,
            "query=0 and root=query",
        ),
        SemanticState(
            "left_boundary_right",
            False,
            True,
            False,
            False,
            False,
            RIGHT,
            "query=0 and root>query",
        ),
        SemanticState(
            "interior_left",
            True,
            True,
            False,
            True,
            False,
            LEFT,
            "query>0 and root<query",
        ),
        SemanticState(
            "interior_equal",
            True,
            True,
            True,
            True,
            False,
            ACCEPT,
            "query>0 and root=query",
        ),
        SemanticState(
            "interior_right",
            True,
            True,
            True,
            False,
            True,
            RIGHT,
            "query>0 and root>query",
        ),
    )


def encode_values(values: dict[str, bool], features: tuple[str, ...]) -> int:
    code = 0
    for index, feature in enumerate(features):
        code |= int(values[feature]) << index
    return code


def semantic_lookup(
    features: tuple[str, ...],
) -> tuple[dict[int, int] | None, dict[str, object] | None]:
    mapping: dict[int, int] = {}
    witnesses: dict[int, SemanticState] = {}
    for state in canonical_semantic_universe():
        code = encode_values(state.values(), features)
        previous = mapping.get(code)
        if previous is not None and previous != state.action:
            first = witnesses[code]
            return None, {
                "code": code,
                "first_state": first.name,
                "first_action": ACTION_NAMES[first.action],
                "second_state": state.name,
                "second_action": ACTION_NAMES[state.action],
            }
        mapping[code] = state.action
        witnesses[code] = state
    return mapping, None


def semantic_lower_bound_certificate() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    consistent: list[tuple[str, ...]] = []
    for feature_count in range(1, len(FEATURE_NAMES) + 1):
        for features in itertools.combinations(FEATURE_NAMES, feature_count):
            mapping, conflict = semantic_lookup(features)
            row = {
                "features": list(features),
                "feature_count": feature_count,
                "consistent": mapping is not None,
                "mapping": (
                    {
                        str(code): ACTION_NAMES[action]
                        for code, action in sorted(mapping.items())
                    }
                    if mapping is not None
                    else None
                ),
                "conflict_witness": conflict,
            }
            rows.append(row)
            if mapping is not None:
                consistent.append(features)

    minimum = min(len(features) for features in consistent)
    minimal = sorted(features for features in consistent if len(features) == minimum)
    selected = minimal[0]
    all_smaller_refuted = all(
        row["conflict_witness"] is not None
        for row in rows
        if int(row["feature_count"]) < minimum
    )
    selected_mapping, selected_conflict = semantic_lookup(selected)
    assert selected_mapping is not None and selected_conflict is None

    semantic_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "name": state.name,
                    "features": state.values(),
                    "action": ACTION_NAMES[state.action],
                    "derivation": state.derivation,
                }
                for state in canonical_semantic_universe()
            ],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "semantic_states": [
            {
                "name": state.name,
                "features": state.values(),
                "action": ACTION_NAMES[state.action],
                "derivation": state.derivation,
            }
            for state in canonical_semantic_universe()
        ],
        "semantic_state_count": len(canonical_semantic_universe()),
        "feature_grammar": list(FEATURE_NAMES),
        "subsets_checked": len(rows),
        "minimum_feature_count": minimum,
        "minimal_feature_sets": [list(features) for features in minimal],
        "unique_minimal_feature_set": len(minimal) == 1,
        "selected_features": list(selected),
        "selected_mapping": {
            str(code): ACTION_NAMES[action]
            for code, action in sorted(selected_mapping.items())
        },
        "all_smaller_subsets_refuted": all_smaller_refuted,
        "semantic_digest": semantic_digest,
        "rows": rows,
    }


def _sample_case(
    rng: np.random.Generator,
    state_name: str,
    dimensions: tuple[int, ...],
) -> tuple[int, int, int]:
    dimension = int(rng.choice(dimensions))
    if state_name == "left_boundary_equal":
        return dimension, 0, 0
    if state_name == "left_boundary_right":
        return dimension, int(rng.integers(1, dimension)), 0

    if dimension < 3:
        raise ValueError("interior semantic states require dimension >= 3")
    query = int(rng.integers(1, dimension - 1))
    if state_name == "interior_left":
        root = int(rng.integers(0, query))
    elif state_name == "interior_equal":
        root = query
    elif state_name == "interior_right":
        root = int(rng.integers(query + 1, dimension))
    else:
        raise ValueError(state_name)
    return dimension, root, query


def generate_balanced_examples(
    seed: int,
    per_state: int,
    dimensions: Iterable[int],
    replicates: int,
    rho_low: float,
    rho_high: float,
) -> list[ResponseExample]:
    rng = np.random.default_rng(seed)
    dimensions_tuple = tuple(dimensions)
    examples: list[ResponseExample] = []
    for state in canonical_semantic_universe():
        for index in range(per_state):
            dimension, root, query = _sample_case(
                rng, state.name, dimensions_tuple
            )
            rho = float(rng.uniform(rho_low, rho_high))
            left, right = intervention_response(
                seed
                + (index + 1) * 104_729
                + (len(examples) + 1) * 65_537,
                dimension,
                root,
                rho,
                query,
                replicates,
            )
            examples.append(
                ResponseExample(
                    state=state.name,
                    left=left,
                    right=right,
                    action=state.action,
                )
            )
    rng.shuffle(examples)
    return examples


def response_feature_values(
    left: float | None,
    right: float | None,
    threshold: float,
) -> dict[str, bool]:
    left_number = left if left is not None else -1e12
    right_number = right if right is not None else -1e12
    return {
        "left_exists": left is not None,
        "right_exists": right is not None,
        "left_active": left is not None and left > threshold,
        "right_active": right is not None and right > threshold,
        "left_greater": left_number > right_number,
    }


def encode_response(
    left: float | None,
    right: float | None,
    threshold: float,
    features: tuple[str, ...],
) -> int:
    return encode_values(
        response_feature_values(left, right, threshold), features
    )


def threshold_candidates(
    examples: list[ResponseExample],
    count: int = 161,
) -> list[float]:
    values = [
        value
        for example in examples
        for value in (example.left, example.right)
        if value is not None
    ]
    quantiles = np.linspace(0.01, 0.99, count)
    candidates = {
        float(value) for value in np.quantile(values, quantiles)
    }
    candidates.update((0.0, 0.10, 0.15, 0.20, 0.25, 0.30))
    return sorted(candidates)


def _lookup_from_examples(
    examples: list[ResponseExample],
    features: tuple[str, ...],
    threshold: float,
) -> tuple[dict[int, int], int]:
    counts: dict[int, list[int]] = {}
    totals = [0, 0, 0]
    for example in examples:
        code = encode_response(
            example.left, example.right, threshold, features
        )
        counts.setdefault(code, [0, 0, 0])[example.action] += 1
        totals[example.action] += 1
    mapping = {
        code: int(np.argmax(action_counts))
        for code, action_counts in counts.items()
    }
    return mapping, int(np.argmax(totals))


def evaluate_classifier(
    program: OutcomeProgram,
    examples: list[ResponseExample],
) -> ClassificationEvaluation:
    mapping = dict(program.mapping)
    correct = 0
    unseen = 0
    state_correct: dict[str, int] = {}
    state_total: dict[str, int] = {}
    for example in examples:
        code = encode_response(
            example.left,
            example.right,
            program.threshold,
            program.features,
        )
        if code not in mapping:
            unseen += 1
        prediction = mapping.get(code, program.default_action)
        correct += int(prediction == example.action)
        state_total[example.state] = state_total.get(example.state, 0) + 1
        state_correct[example.state] = (
            state_correct.get(example.state, 0)
            + int(prediction == example.action)
        )
    per_state = {
        state: state_correct.get(state, 0) / total
        for state, total in sorted(state_total.items())
    }
    return ClassificationEvaluation(
        accuracy=correct / len(examples),
        macro_accuracy=float(np.mean(list(per_state.values()))),
        minimum_state_accuracy=min(per_state.values()),
        unseen_code_rate=unseen / len(examples),
        per_state_accuracy=per_state,
    )


def fit_program(
    training: list[ResponseExample],
    development: list[ResponseExample],
    features: tuple[str, ...],
) -> tuple[OutcomeProgram, ClassificationEvaluation]:
    best: tuple[
        tuple[float, float, float, int, float],
        OutcomeProgram,
        ClassificationEvaluation,
    ] | None = None
    for threshold in threshold_candidates(training):
        mapping, default = _lookup_from_examples(
            training, features, threshold
        )
        program = OutcomeProgram(
            features=features,
            threshold=float(threshold),
            mapping=tuple(sorted(mapping.items())),
            default_action=default,
            training_accuracy=0.0,
            development_accuracy=0.0,
            mapping_entries=len(mapping),
        )
        training_evaluation = evaluate_classifier(program, training)
        development_evaluation = evaluate_classifier(program, development)
        program = OutcomeProgram(
            features=features,
            threshold=program.threshold,
            mapping=program.mapping,
            default_action=program.default_action,
            training_accuracy=training_evaluation.macro_accuracy,
            development_accuracy=development_evaluation.macro_accuracy,
            mapping_entries=program.mapping_entries,
        )
        score = (
            development_evaluation.minimum_state_accuracy,
            development_evaluation.macro_accuracy,
            training_evaluation.macro_accuracy,
            -len(mapping),
            -abs(threshold - 0.20),
        )
        candidate = (score, program, development_evaluation)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2]


def synthesize_programs(
    seed: int,
    certificate: dict[str, object],
) -> tuple[OutcomeProgram, OutcomeProgram, dict[str, object]]:
    training = generate_balanced_examples(
        seed,
        per_state=1_000,
        dimensions=(3, 4, 5, 6, 7, 8),
        replicates=384,
        rho_low=0.30,
        rho_high=0.90,
    )
    development = generate_balanced_examples(
        seed + 1_000_003,
        per_state=500,
        dimensions=(4, 6, 9, 11),
        replicates=384,
        rho_low=0.28,
        rho_high=0.92,
    )
    selected_features = tuple(certificate["selected_features"])
    selected, selected_development = fit_program(
        training, development, selected_features
    )

    reduced_rows = []
    for feature_count in range(1, int(certificate["minimum_feature_count"])):
        for features in itertools.combinations(FEATURE_NAMES, feature_count):
            program, evaluation = fit_program(
                training, development, features
            )
            reduced_rows.append((program, evaluation))
    reduced, reduced_development = max(
        reduced_rows,
        key=lambda item: (
            item[1].minimum_state_accuracy,
            item[1].macro_accuracy,
            item[0].training_accuracy,
            -len(item[0].mapping),
        ),
    )

    evidence = {
        "training_examples": len(training),
        "development_examples": len(development),
        "selected_features": list(selected.features),
        "selected_threshold": selected.threshold,
        "selected_mapping_entries": selected.mapping_entries,
        "selected_training_macro_accuracy": selected.training_accuracy,
        "selected_development": selected_development.__dict__,
        "reduced_features": list(reduced.features),
        "reduced_threshold": reduced.threshold,
        "reduced_mapping_entries": reduced.mapping_entries,
        "reduced_training_macro_accuracy": reduced.training_accuracy,
        "reduced_development": reduced_development.__dict__,
    }
    return selected, reduced, evidence


def run(seed: int = 901) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    query_rule, query_evidence = synthesize_position_rule()
    certificate = semantic_lower_bound_certificate()
    selected, reduced, synthesis = synthesize_programs(
        seed * 10_000 + 73,
        certificate,
    )
    frozen_digest = decoder_digest(query_rule, selected)

    # Hidden semantic states are generated only after the semantic certificate,
    # threshold, lookup table and reduced control have been frozen.
    hidden_examples = generate_balanced_examples(
        seed * 10_000 + 9_000_001,
        per_state=1_000,
        dimensions=(9, 13, 21, 37, 63),
        replicates=512,
        rho_low=0.28,
        rho_high=0.94,
    )
    candidate_semantic = evaluate_classifier(selected, hidden_examples)
    reduced_semantic = evaluate_classifier(reduced, hidden_examples)

    hidden_dimensions = (9, 13, 21, 37, 63)
    candidate_loop = evaluate_program(
        seed * 10_000 + 9_000_003,
        query_rule,
        selected,
        hidden_dimensions,
        trials=3_000,
        replicates=512,
    )
    reduced_loop = evaluate_program(
        seed * 10_000 + 9_000_005,
        query_rule,
        reduced,
        hidden_dimensions,
        trials=3_000,
        replicates=512,
    )
    human_loop = evaluate_program(
        seed * 10_000 + 9_000_007,
        query_rule,
        hand_program(selected.threshold),
        hidden_dimensions,
        trials=3_000,
        replicates=512,
    )
    random_control = random_codebook_control(
        seed * 10_000 + 9_000_009,
        query_rule,
        selected,
        hidden_dimensions,
        trials=48,
    )

    semantic_gap = (
        candidate_semantic.macro_accuracy
        - reduced_semantic.macro_accuracy
    )
    loop_gap = candidate_loop.accuracy - reduced_loop.accuracy
    human_gap = candidate_loop.accuracy - human_loop.accuracy
    random_gap = (
        candidate_loop.accuracy - random_control["median_accuracy"]
    )
    query_optimality = all(
        optimal_worst_case_queries(dimension)
        == information_lower_bound(dimension)
        for dimension in hidden_dimensions
    )

    candidate_gate = (
        bool(equivalence["exact_within_tolerance"])
        and query_rule.name == "lower_midpoint"
        and query_evidence["all_training_depths_meet_lower_bound"]
        and certificate["minimum_feature_count"] == 3
        and certificate["unique_minimal_feature_set"]
        and certificate["all_smaller_subsets_refuted"]
        and tuple(certificate["selected_features"])
        == selected.features
        and selected.development_accuracy >= 0.985
        and candidate_semantic.macro_accuracy >= 0.985
        and candidate_semantic.minimum_state_accuracy >= 0.97
        and candidate_semantic.unseen_code_rate <= 0.01
        and semantic_gap >= 0.15
        and candidate_loop.accuracy >= 0.985
        and candidate_loop.invalid_transition_rate <= 0.01
        and loop_gap >= 0.02
        and human_gap >= -0.01
        and random_gap >= 0.45
        and query_optimality
    )

    return {
        "status": (
            "semantic_outcome_irreducibility_candidate"
            if candidate_gate
            else "not_yet"
        ),
        "claim_scope": (
            "the system carries a machine-checkable semantic certificate that "
            "every one- and two-feature response alphabet aliases reachable "
            "states requiring different actions, while one unique three-feature "
            "alphabet supports a frozen controller on boundary-balanced hidden "
            "causal chains; this remains a proof-carrying controller-synthesis "
            "milestone around classical ordered search, not a world breakthrough"
        ),
        "seed": seed,
        "candidate_gate": candidate_gate,
        "observational_equivalence": equivalence,
        "query_synthesis": query_evidence,
        "semantic_certificate": certificate,
        "program_synthesis": synthesis,
        "selected_program": {
            "features": list(selected.features),
            "threshold": selected.threshold,
            "mapping": [
                {"code": code, "action": ACTION_NAMES[action]}
                for code, action in selected.mapping
            ],
            "default_action": ACTION_NAMES[selected.default_action],
            "training_macro_accuracy": selected.training_accuracy,
            "development_macro_accuracy": selected.development_accuracy,
        },
        "reduced_program": {
            "features": list(reduced.features),
            "threshold": reduced.threshold,
            "mapping": [
                {"code": code, "action": ACTION_NAMES[action]}
                for code, action in reduced.mapping
            ],
            "default_action": ACTION_NAMES[reduced.default_action],
            "training_macro_accuracy": reduced.training_accuracy,
            "development_macro_accuracy": reduced.development_accuracy,
        },
        "frozen_decoder_digest": frozen_digest,
        "hidden_dimensions": list(hidden_dimensions),
        "hidden_query_optimality": query_optimality,
        "candidate_semantic": candidate_semantic.__dict__,
        "best_reduced_semantic": reduced_semantic.__dict__,
        "candidate_closed_loop": candidate_loop.__dict__,
        "best_reduced_closed_loop": reduced_loop.__dict__,
        "human_closed_loop": human_loop.__dict__,
        "random_codebook_control": random_control,
        "semantic_gap": semantic_gap,
        "closed_loop_gap": loop_gap,
        "human_gap": human_gap,
        "random_gap": random_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "features": report["selected_program"]["features"],
                "semantic_macro_accuracy": report["candidate_semantic"][
                    "macro_accuracy"
                ],
                "minimum_state_accuracy": report["candidate_semantic"][
                    "minimum_state_accuracy"
                ],
                "semantic_gap": report["semantic_gap"],
                "closed_loop_accuracy": report["candidate_closed_loop"][
                    "accuracy"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
