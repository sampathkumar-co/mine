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
    PositionRule,
    information_lower_bound,
    intervention_response,
    observational_equivalence_certificate,
    synthesize_position_rule,
)


LEFT = 0
ACCEPT = 1
RIGHT = 2
ACTION_NAMES = {LEFT: "keep_left", ACCEPT: "accept_query", RIGHT: "keep_right"}
FEATURE_NAMES = (
    "left_exists",
    "right_exists",
    "left_active",
    "right_active",
    "left_greater",
)


@dataclass(frozen=True)
class DecoderExample:
    left: float | None
    right: float | None
    action: int


@dataclass(frozen=True)
class OutcomeProgram:
    features: tuple[str, ...]
    threshold: float
    mapping: tuple[tuple[int, int], ...]
    default_action: int
    training_accuracy: float
    development_accuracy: float
    mapping_entries: int

    def mapping_dict(self) -> dict[int, int]:
        return dict(self.mapping)

    def text(self) -> str:
        encoded = ",".join(
            f"{code}:{ACTION_NAMES[action]}" for code, action in self.mapping
        )
        return (
            f"features={'+'.join(self.features)};threshold={self.threshold:.8f};"
            f"default={ACTION_NAMES[self.default_action]};map={encoded}"
        )


@dataclass(frozen=True)
class ClosedLoopEvaluation:
    accuracy: float
    mean_queries: float
    maximum_queries: int
    invalid_transition_rate: float
    mean_remaining_candidates: float


def feature_values(
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
    values = feature_values(left, right, threshold)
    code = 0
    for index, feature in enumerate(features):
        code |= int(values[feature]) << index
    return code


def generate_examples(
    seed: int,
    samples: int = 6_000,
    replicates: int = 256,
) -> list[DecoderExample]:
    rng = np.random.default_rng(seed)
    examples: list[DecoderExample] = []
    for index in range(samples):
        dimension = int(rng.choice((3, 4, 5, 6, 7)))
        root = int(rng.integers(0, dimension))
        query = int(rng.integers(0, dimension))
        rho = float(rng.uniform(0.30, 0.90))
        left, right = intervention_response(
            seed + index * 65_537,
            dimension,
            root,
            rho,
            query,
            replicates,
        )
        action = LEFT if root < query else ACCEPT if root == query else RIGHT
        examples.append(DecoderExample(left, right, action))
    return examples


def threshold_candidates(
    examples: list[DecoderExample],
    count: int = 129,
) -> list[float]:
    values = [
        value
        for example in examples
        for value in (example.left, example.right)
        if value is not None
    ]
    quantiles = np.linspace(0.02, 0.98, count)
    candidates = sorted(
        set(float(value) for value in np.quantile(values, quantiles))
    )
    candidates.extend((0.0, 0.15, 0.20, 0.25, 0.30))
    return sorted(set(candidates))


def optimal_lookup(
    examples: list[DecoderExample],
    features: tuple[str, ...],
    threshold: float,
) -> tuple[float, dict[int, int], int]:
    counts: dict[int, list[int]] = {}
    global_counts = [0, 0, 0]
    for example in examples:
        code = encode_response(
            example.left, example.right, threshold, features
        )
        bucket = counts.setdefault(code, [0, 0, 0])
        bucket[example.action] += 1
        global_counts[example.action] += 1
    mapping = {
        code: int(np.argmax(bucket)) for code, bucket in counts.items()
    }
    correct = sum(
        bucket[mapping[code]] for code, bucket in counts.items()
    )
    default = int(np.argmax(global_counts))
    return correct / len(examples), mapping, default


def fit_subset(
    examples: list[DecoderExample],
    features: tuple[str, ...],
) -> OutcomeProgram:
    best: tuple[tuple[float, int, float], float, dict[int, int], int] | None = None
    for threshold in threshold_candidates(examples):
        accuracy, mapping, default = optimal_lookup(
            examples, features, threshold
        )
        score = (accuracy, -len(mapping), -abs(threshold - 0.20))
        candidate = (score, threshold, mapping, default)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    score, threshold, mapping, default = best
    return OutcomeProgram(
        features=features,
        threshold=float(threshold),
        mapping=tuple(sorted(mapping.items())),
        default_action=default,
        training_accuracy=float(score[0]),
        development_accuracy=0.0,
        mapping_entries=len(mapping),
    )


def decode_action(
    program: OutcomeProgram,
    left: float | None,
    right: float | None,
) -> int:
    code = encode_response(
        left, right, program.threshold, program.features
    )
    return program.mapping_dict().get(code, program.default_action)


def run_closed_loop(
    seed: int,
    dimension: int,
    root: int,
    rho: float,
    query_rule: PositionRule,
    program: OutcomeProgram,
    replicates: int,
    query_budget: int,
) -> tuple[bool, int, bool, int]:
    low = 0
    high = dimension - 1
    queries = 0
    invalid = False
    while low < high and queries < query_budget:
        size = high - low + 1
        query = low + query_rule.offset(size)
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
            return query == root, queries, invalid, 1
        if action == LEFT:
            high = query - 1
        elif action == RIGHT:
            low = query + 1
        else:
            raise ValueError(action)
        if low > high:
            invalid = True
            return False, queries, invalid, 0
    remaining = max(1, high - low + 1)
    prediction = low + (remaining - 1) // 2
    return prediction == root, queries, invalid, remaining


def evaluate_program(
    seed: int,
    query_rule: PositionRule,
    program: OutcomeProgram,
    dimensions: tuple[int, ...],
    trials: int,
    replicates: int = 256,
) -> ClosedLoopEvaluation:
    rng = np.random.default_rng(seed)
    correct: list[bool] = []
    queries: list[int] = []
    invalid: list[bool] = []
    remaining: list[int] = []
    for index in range(trials):
        dimension = int(rng.choice(dimensions))
        root = int(rng.integers(0, dimension))
        rho = float(rng.uniform(0.30, 0.90))
        result = run_closed_loop(
            seed + index * 104_729,
            dimension,
            root,
            rho,
            query_rule,
            program,
            replicates,
            information_lower_bound(dimension),
        )
        was_correct, query_count, was_invalid, candidates = result
        correct.append(was_correct)
        queries.append(query_count)
        invalid.append(was_invalid)
        remaining.append(candidates)
    return ClosedLoopEvaluation(
        accuracy=float(np.mean(correct)),
        mean_queries=float(np.mean(queries)),
        maximum_queries=max(queries),
        invalid_transition_rate=float(np.mean(invalid)),
        mean_remaining_candidates=float(np.mean(remaining)),
    )


def with_development_accuracy(
    program: OutcomeProgram,
    accuracy: float,
) -> OutcomeProgram:
    return OutcomeProgram(
        features=program.features,
        threshold=program.threshold,
        mapping=program.mapping,
        default_action=program.default_action,
        training_accuracy=program.training_accuracy,
        development_accuracy=accuracy,
        mapping_entries=program.mapping_entries,
    )


def synthesize_outcome_program(
    seed: int,
    query_rule: PositionRule,
) -> tuple[OutcomeProgram, OutcomeProgram, dict[str, object]]:
    examples = generate_examples(seed, samples=6_000, replicates=256)
    rows: list[tuple[OutcomeProgram, ClosedLoopEvaluation]] = []
    for size in range(1, len(FEATURE_NAMES) + 1):
        for features in itertools.combinations(FEATURE_NAMES, size):
            fitted = fit_subset(examples, features)
            development = evaluate_program(
                seed + size * 1_000_003 + sum(map(len, features)),
                query_rule,
                fitted,
                dimensions=(3, 4, 5, 6, 7),
                trials=1_600,
                replicates=256,
            )
            rows.append(
                (
                    with_development_accuracy(
                        fitted, development.accuracy
                    ),
                    development,
                )
            )

    best_development = max(
        program.development_accuracy for program, _ in rows
    )
    eligible = [
        (program, evaluation)
        for program, evaluation in rows
        if program.development_accuracy >= 0.99
        and program.development_accuracy >= best_development - 0.005
    ]
    if not eligible:
        raise RuntimeError("no compact response alphabet met the development gate")
    eligible.sort(
        key=lambda item: (
            len(item[0].features),
            item[0].mapping_entries,
            -item[0].development_accuracy,
            -item[0].training_accuracy,
            item[0].features,
        )
    )
    selected = eligible[0][0]

    reduced_rows = [
        (program, evaluation)
        for program, evaluation in rows
        if len(program.features) < len(selected.features)
    ]
    if not reduced_rows:
        reduced = min(
            (program for program, _ in rows),
            key=lambda value: len(value.features),
        )
    else:
        reduced = max(
            reduced_rows,
            key=lambda item: (
                item[0].development_accuracy,
                item[0].training_accuracy,
                -item[0].mapping_entries,
            ),
        )[0]

    best_by_feature_count = {}
    for feature_count in range(1, len(FEATURE_NAMES) + 1):
        candidates = [
            program.development_accuracy
            for program, _ in rows
            if len(program.features) == feature_count
        ]
        best_by_feature_count[feature_count] = max(candidates)

    evidence = {
        "example_count": len(examples),
        "feature_grammar": list(FEATURE_NAMES),
        "feature_subsets_evaluated": len(rows),
        "best_development_accuracy": best_development,
        "selected_features": list(selected.features),
        "selected_feature_count": len(selected.features),
        "selected_training_accuracy": selected.training_accuracy,
        "selected_development_accuracy": selected.development_accuracy,
        "selected_mapping_entries": selected.mapping_entries,
        "selected_program": selected.text(),
        "best_reduced_features": list(reduced.features),
        "best_reduced_development_accuracy": reduced.development_accuracy,
        "best_by_feature_count": best_by_feature_count,
        "minimal_within_tolerance": all(
            accuracy < 0.99
            for count, accuracy in best_by_feature_count.items()
            if count < len(selected.features)
        ),
    }
    return selected, reduced, evidence


def hand_program(threshold: float) -> OutcomeProgram:
    # Explicit upper control corresponding to the human boundary/activity
    # decoder used in v0.22. It is not part of synthesis selection.
    features = (
        "left_exists",
        "right_exists",
        "left_active",
        "right_active",
    )
    mapping = {
        1: LEFT,
        2: RIGHT,
        3: LEFT,
        5: ACCEPT,
        7: RIGHT,
        10: ACCEPT,
        11: LEFT,
        15: ACCEPT,
    }
    return OutcomeProgram(
        features,
        threshold,
        tuple(sorted(mapping.items())),
        ACCEPT,
        1.0,
        1.0,
        len(mapping),
    )


def random_codebook_control(
    seed: int,
    query_rule: PositionRule,
    template: OutcomeProgram,
    dimensions: tuple[int, ...],
    trials: int = 48,
) -> dict[str, float]:
    rng = random.Random(seed)
    scores = []
    keys = [code for code, _ in template.mapping]
    for index in range(trials):
        mapping = tuple(
            sorted((code, rng.choice((LEFT, ACCEPT, RIGHT))) for code in keys)
        )
        program = OutcomeProgram(
            template.features,
            template.threshold,
            mapping,
            rng.choice((LEFT, ACCEPT, RIGHT)),
            0.0,
            0.0,
            len(mapping),
        )
        score = evaluate_program(
            seed + index * 1_000_003,
            query_rule,
            program,
            dimensions,
            trials=420,
            replicates=256,
        ).accuracy
        scores.append(score)
    return {
        "trials": trials,
        "median_accuracy": float(np.median(scores)),
        "maximum_accuracy": max(scores),
    }


def decoder_digest(
    query_rule: PositionRule,
    program: OutcomeProgram,
) -> str:
    return hashlib.sha256(
        f"{query_rule.name}:{program.text()}".encode("utf-8")
    ).hexdigest()


def run(seed: int = 801) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    query_rule, query_evidence = synthesize_position_rule()
    selected, reduced, synthesis = synthesize_outcome_program(
        seed * 10_000 + 71,
        query_rule,
    )
    frozen_digest = decoder_digest(query_rule, selected)

    # Irregular hidden sizes are created only after the query rule, feature
    # alphabet, threshold and action table have been frozen.
    hidden_dimensions = (9, 13, 21, 37, 63)
    candidate = evaluate_program(
        seed * 10_000 + 8_000_001,
        query_rule,
        selected,
        hidden_dimensions,
        trials=2_800,
        replicates=256,
    )
    reduced_control = evaluate_program(
        seed * 10_000 + 8_000_003,
        query_rule,
        reduced,
        hidden_dimensions,
        trials=2_800,
        replicates=256,
    )
    human_control = evaluate_program(
        seed * 10_000 + 8_000_005,
        query_rule,
        hand_program(selected.threshold),
        hidden_dimensions,
        trials=2_800,
        replicates=256,
    )
    random_control = random_codebook_control(
        seed * 10_000 + 8_000_007,
        query_rule,
        selected,
        hidden_dimensions,
    )

    query_optimality = all(
        information_lower_bound(dimension)
        == max(
            0, math.ceil(math.log2(dimension + 1)) - 1
        )
        for dimension in hidden_dimensions
    )
    reduced_gap = candidate.accuracy - reduced_control.accuracy
    random_gap = candidate.accuracy - random_control["median_accuracy"]
    human_gap = candidate.accuracy - human_control.accuracy
    candidate_gate = (
        bool(equivalence["exact_within_tolerance"])
        and query_rule.name == "lower_midpoint"
        and query_evidence["all_training_depths_meet_lower_bound"]
        and synthesis["minimal_within_tolerance"]
        and synthesis["selected_feature_count"] <= 3
        and synthesis["selected_development_accuracy"] >= 0.99
        and query_optimality
        and candidate.accuracy >= 0.985
        and candidate.invalid_transition_rate <= 0.01
        and reduced_gap >= 0.02
        and random_gap >= 0.45
        and human_gap >= -0.01
    )

    return {
        "status": (
            "self_synthesized_outcome_alphabet_candidate"
            if candidate_gate
            else "not_yet"
        ),
        "claim_scope": (
            "the system jointly synthesizes a compact Boolean alphabet over continuous intervention "
            "responses and an action lookup table, freezes it with the midpoint query program, and "
            "transfers to irregular unseen causal-chain sizes; the result is finite-state controller "
            "and decision-tree synthesis around classical binary search, not a world breakthrough"
        ),
        "seed": seed,
        "candidate_gate": candidate_gate,
        "observational_equivalence": equivalence,
        "query_synthesis": query_evidence,
        "outcome_synthesis": synthesis,
        "selected_program": {
            "features": list(selected.features),
            "threshold": selected.threshold,
            "mapping": [
                {"code": code, "action": ACTION_NAMES[action]}
                for code, action in selected.mapping
            ],
            "default_action": ACTION_NAMES[selected.default_action],
            "training_accuracy": selected.training_accuracy,
            "development_accuracy": selected.development_accuracy,
        },
        "reduced_program": {
            "features": list(reduced.features),
            "threshold": reduced.threshold,
            "training_accuracy": reduced.training_accuracy,
            "development_accuracy": reduced.development_accuracy,
        },
        "frozen_decoder_digest": frozen_digest,
        "hidden_dimensions": list(hidden_dimensions),
        "hidden_query_optimality": query_optimality,
        "candidate": candidate.__dict__,
        "best_reduced_control": reduced_control.__dict__,
        "human_decoder_control": human_control.__dict__,
        "random_codebook_control": random_control,
        "candidate_reduced_gap": reduced_gap,
        "candidate_random_gap": random_gap,
        "candidate_human_gap": human_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=801)
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
                "development_accuracy": report["selected_program"][
                    "development_accuracy"
                ],
                "hidden_accuracy": report["candidate"]["accuracy"],
                "reduced_accuracy": report["best_reduced_control"][
                    "accuracy"
                ],
                "human_accuracy": report["human_decoder_control"][
                    "accuracy"
                ],
                "random_median": report["random_codebook_control"][
                    "median_accuracy"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
