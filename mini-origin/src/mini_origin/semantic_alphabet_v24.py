from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Iterable

import numpy as np

from .intervention_genesis_v22 import (
    fit_activity_threshold,
    information_lower_bound,
    intervention_response,
    observational_equivalence_certificate,
)
from .outcome_alphabet_v23 import (
    ACCEPT,
    ACTION_NAMES,
    FEATURE_NAMES,
    LEFT,
    RIGHT,
    OutcomeProgram,
    decode_action,
    encode_response,
)


@dataclass(frozen=True)
class SemanticState:
    name: str
    left: float | None
    right: float | None
    action: int


@dataclass(frozen=True)
class SemanticCertificate:
    selected_features: tuple[str, ...]
    selected_mapping: tuple[tuple[int, int], ...]
    selected_default: int
    selected_accuracy: float
    selected_feature_count: int
    perfect_smaller_alphabets: int
    best_smaller_accuracy: float
    exhaustive_subsets: int
    state_count: int


@dataclass(frozen=True)
class PolicyEvaluation:
    accuracy: float
    lower_accuracy: float
    upper_accuracy: float
    alternating_accuracy: float
    mean_queries: float
    maximum_queries: int
    invalid_transition_rate: float


def semantic_states() -> tuple[SemanticState, ...]:
    inactive = 0.0
    active = 1.0
    return (
        SemanticState("left_boundary_equal", None, active, ACCEPT),
        SemanticState("left_boundary_root_right", None, inactive, RIGHT),
        SemanticState("right_boundary_equal", active, None, ACCEPT),
        SemanticState("right_boundary_root_left", inactive, None, LEFT),
        SemanticState("interior_root_left", inactive, active, LEFT),
        SemanticState("interior_equal", active, active, ACCEPT),
        SemanticState("interior_root_right", active, inactive, RIGHT),
    )


def optimal_semantic_mapping(
    features: tuple[str, ...],
    threshold: float,
) -> tuple[float, dict[int, int], int]:
    counts: dict[int, list[int]] = {}
    global_counts = [0, 0, 0]
    for state in semantic_states():
        code = encode_response(
            state.left, state.right, threshold, features
        )
        counts.setdefault(code, [0, 0, 0])[state.action] += 1
        global_counts[state.action] += 1
    mapping = {
        code: int(np.argmax(bucket)) for code, bucket in counts.items()
    }
    correct = sum(
        bucket[mapping[code]] for code, bucket in counts.items()
    )
    return correct / len(semantic_states()), mapping, int(np.argmax(global_counts))


def semantic_program(
    features: tuple[str, ...],
    threshold: float,
) -> OutcomeProgram:
    accuracy, mapping, default = optimal_semantic_mapping(
        features, threshold
    )
    return OutcomeProgram(
        features=features,
        threshold=threshold,
        mapping=tuple(sorted(mapping.items())),
        default_action=default,
        training_accuracy=accuracy,
        development_accuracy=0.0,
        mapping_entries=len(mapping),
    )


def exhaustive_semantic_certificate(
    threshold: float,
) -> tuple[OutcomeProgram, list[OutcomeProgram], SemanticCertificate]:
    programs: list[OutcomeProgram] = []
    for size in range(1, len(FEATURE_NAMES) + 1):
        for features in itertools.combinations(FEATURE_NAMES, size):
            programs.append(semantic_program(features, threshold))

    perfect = [
        program for program in programs if program.training_accuracy == 1.0
    ]
    if not perfect:
        raise RuntimeError("no exact semantic alphabet exists in the grammar")
    perfect.sort(
        key=lambda program: (
            len(program.features),
            program.mapping_entries,
            program.features,
        )
    )
    selected = perfect[0]
    smaller = [
        program
        for program in programs
        if len(program.features) < len(selected.features)
    ]
    certificate = SemanticCertificate(
        selected_features=selected.features,
        selected_mapping=selected.mapping,
        selected_default=selected.default_action,
        selected_accuracy=selected.training_accuracy,
        selected_feature_count=len(selected.features),
        perfect_smaller_alphabets=sum(
            program.training_accuracy == 1.0 for program in smaller
        ),
        best_smaller_accuracy=max(
            program.training_accuracy for program in smaller
        ),
        exhaustive_subsets=len(programs),
        state_count=len(semantic_states()),
    )
    return selected, smaller, certificate


def query_position(
    mode: str,
    low: int,
    high: int,
    step: int,
) -> int:
    size = high - low + 1
    lower = low + (size - 1) // 2
    upper = low + size // 2
    if mode == "lower":
        return lower
    if mode == "upper":
        return upper
    if mode == "alternating":
        return lower if step % 2 == 0 else upper
    raise ValueError(mode)


def run_closed_loop(
    seed: int,
    dimension: int,
    root: int,
    rho: float,
    program: OutcomeProgram,
    mode: str,
    replicates: int,
) -> tuple[bool, int, bool]:
    low = 0
    high = dimension - 1
    queries = 0
    budget = information_lower_bound(dimension)
    while low < high and queries < budget:
        query = query_position(mode, low, high, queries)
        left, right = intervention_response(
            seed + queries * 7_919,
            dimension,
            root,
            rho,
            query,
            replicates,
        )
        action = decode_action(program, left, right)
        queries += 1
        if action == ACCEPT:
            return query == root, queries, False
        if action == LEFT:
            high = query - 1
        elif action == RIGHT:
            low = query + 1
        else:
            raise ValueError(action)
        if low > high:
            return False, queries, True
    prediction = low + max(0, high - low) // 2
    return prediction == root, queries, False


def sample_root(
    rng: np.random.Generator,
    dimension: int,
) -> int:
    draw = float(rng.random())
    if draw < 0.25:
        return 0
    if draw < 0.50:
        return dimension - 1
    if draw < 0.625 and dimension > 2:
        return 1
    if draw < 0.75 and dimension > 2:
        return dimension - 2
    return int(rng.integers(0, dimension))


def evaluate_program(
    seed: int,
    program: OutcomeProgram,
    dimensions: tuple[int, ...],
    trials: int,
    replicates: int = 256,
) -> PolicyEvaluation:
    rng = np.random.default_rng(seed)
    modes = ("lower", "upper", "alternating")
    correct: list[bool] = []
    query_counts: list[int] = []
    invalid: list[bool] = []
    per_mode: dict[str, list[bool]] = {mode: [] for mode in modes}
    for index in range(trials):
        dimension = int(rng.choice(dimensions))
        root = sample_root(rng, dimension)
        rho = float(rng.uniform(0.30, 0.90))
        mode = modes[index % len(modes)]
        was_correct, queries, was_invalid = run_closed_loop(
            seed + index * 104_729,
            dimension,
            root,
            rho,
            program,
            mode,
            replicates,
        )
        correct.append(was_correct)
        query_counts.append(queries)
        invalid.append(was_invalid)
        per_mode[mode].append(was_correct)
    return PolicyEvaluation(
        accuracy=float(np.mean(correct)),
        lower_accuracy=float(np.mean(per_mode["lower"])),
        upper_accuracy=float(np.mean(per_mode["upper"])),
        alternating_accuracy=float(np.mean(per_mode["alternating"])),
        mean_queries=float(np.mean(query_counts)),
        maximum_queries=max(query_counts),
        invalid_transition_rate=float(np.mean(invalid)),
    )


def select_strongest_smaller_control(
    seed: int,
    smaller: list[OutcomeProgram],
) -> tuple[OutcomeProgram, PolicyEvaluation]:
    rows = []
    for index, program in enumerate(smaller):
        evaluation = evaluate_program(
            seed + index * 1_000_003,
            program,
            dimensions=(4, 5, 6, 7, 8),
            trials=1_800,
            replicates=256,
        )
        rows.append((evaluation.accuracy, program.training_accuracy, program, evaluation))
    rows.sort(key=lambda row: (row[0], row[1], row[2].features), reverse=True)
    _, _, program, evaluation = rows[0]
    return program, evaluation


def random_codebook_control(
    seed: int,
    template: OutcomeProgram,
    dimensions: tuple[int, ...],
    trials: int = 64,
) -> dict[str, float]:
    rng = random.Random(seed)
    codes = [code for code, _ in template.mapping]
    scores = []
    for index in range(trials):
        program = OutcomeProgram(
            features=template.features,
            threshold=template.threshold,
            mapping=tuple(
                sorted(
                    (code, rng.choice((LEFT, ACCEPT, RIGHT)))
                    for code in codes
                )
            ),
            default_action=rng.choice((LEFT, ACCEPT, RIGHT)),
            training_accuracy=0.0,
            development_accuracy=0.0,
            mapping_entries=len(codes),
        )
        scores.append(
            evaluate_program(
                seed + index * 1_000_003,
                program,
                dimensions,
                trials=480,
                replicates=256,
            ).accuracy
        )
    return {
        "trials": trials,
        "median_accuracy": float(np.median(scores)),
        "maximum_accuracy": max(scores),
    }


def program_digest(program: OutcomeProgram) -> str:
    return hashlib.sha256(program.text().encode("utf-8")).hexdigest()


def run(seed: int = 901) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    threshold = fit_activity_threshold(
        seed * 10_000 + 101,
        samples=1_800,
        replicates=256,
    )
    selected, smaller, certificate = exhaustive_semantic_certificate(
        threshold.threshold
    )
    reduced, reduced_development = select_strongest_smaller_control(
        seed * 10_000 + 103,
        smaller,
    )
    frozen_digest = program_digest(selected)

    # The hidden policy family, dimensions and boundary-heavy root mixture are
    # opened only after the semantic alphabet and strongest smaller control are frozen.
    hidden_dimensions = (10, 15, 26, 41, 70)
    candidate = evaluate_program(
        seed * 10_000 + 9_000_001,
        selected,
        hidden_dimensions,
        trials=3_600,
        replicates=256,
    )
    reduced_hidden = evaluate_program(
        seed * 10_000 + 9_000_003,
        reduced,
        hidden_dimensions,
        trials=3_600,
        replicates=256,
    )
    random_control = random_codebook_control(
        seed * 10_000 + 9_000_005,
        selected,
        hidden_dimensions,
    )

    policy_floor = min(
        candidate.lower_accuracy,
        candidate.upper_accuracy,
        candidate.alternating_accuracy,
    )
    reduced_gap = candidate.accuracy - reduced_hidden.accuracy
    random_gap = candidate.accuracy - random_control["median_accuracy"]
    candidate_gate = (
        bool(equivalence["exact_within_tolerance"])
        and threshold.balanced_accuracy >= 0.995
        and certificate.selected_accuracy == 1.0
        and certificate.selected_feature_count == 4
        and certificate.perfect_smaller_alphabets == 0
        and certificate.best_smaller_accuracy <= 6.0 / 7.0 + 1e-12
        and candidate.accuracy >= 0.985
        and policy_floor >= 0.98
        and candidate.invalid_transition_rate <= 0.01
        and reduced_gap >= 0.06
        and random_gap >= 0.55
    )

    return {
        "status": (
            "proof_carrying_policy_independent_alphabet_candidate"
            if candidate_gate
            else "not_yet"
        ),
        "claim_scope": (
            "the system exhaustively proves a four-bit lower bound for policy-independent causal "
            "intervention semantics, freezes the exact alphabet, and transfers across lower, upper "
            "and alternating optimal query trees on unseen dimensions; this is formal automata and "
            "controller synthesis around classical binary search, not a world breakthrough"
        ),
        "seed": seed,
        "candidate_gate": candidate_gate,
        "observational_equivalence": equivalence,
        "activity_threshold": threshold.__dict__,
        "semantic_certificate": {
            "selected_features": list(certificate.selected_features),
            "selected_mapping": [
                {"code": code, "action": ACTION_NAMES[action]}
                for code, action in certificate.selected_mapping
            ],
            "selected_accuracy": certificate.selected_accuracy,
            "selected_feature_count": certificate.selected_feature_count,
            "perfect_smaller_alphabets": certificate.perfect_smaller_alphabets,
            "best_smaller_accuracy": certificate.best_smaller_accuracy,
            "exhaustive_subsets": certificate.exhaustive_subsets,
            "state_count": certificate.state_count,
        },
        "selected_program": {
            "features": list(selected.features),
            "threshold": selected.threshold,
            "text": selected.text(),
        },
        "strongest_smaller_program": {
            "features": list(reduced.features),
            "semantic_accuracy": reduced.training_accuracy,
            "development": reduced_development.__dict__,
            "text": reduced.text(),
        },
        "frozen_program_digest": frozen_digest,
        "hidden_dimensions": list(hidden_dimensions),
        "candidate": candidate.__dict__,
        "strongest_smaller_hidden": reduced_hidden.__dict__,
        "random_codebook_control": random_control,
        "candidate_smaller_gap": reduced_gap,
        "candidate_random_gap": random_gap,
        "candidate_policy_floor": policy_floor,
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
                "threshold_accuracy": report["activity_threshold"]["balanced_accuracy"],
                "hidden_accuracy": report["candidate"]["accuracy"],
                "policy_floor": report["candidate_policy_floor"],
                "smaller_accuracy": report["strongest_smaller_hidden"]["accuracy"],
                "smaller_gap": report["candidate_smaller_gap"],
                "random_median": report["random_codebook_control"]["median_accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
