from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from .intervention_genesis_v22 import (
    PositionRule,
    information_lower_bound,
    intervention_response,
    observational_equivalence_certificate,
    optimal_worst_case_queries,
    position_rules,
)
from .outcome_alphabet_v23 import ACCEPT, ACTION_NAMES, LEFT, RIGHT
from .terminal_reward_v26 import World, noiseless_response


@dataclass(frozen=True)
class AdaptivePolicy:
    rule: PositionRule
    right_threshold: float
    difference_threshold: float
    trigger_margin: float
    base_replicates: int
    extra_replicates: int
    mapping: tuple[int, ...] = (RIGHT, ACCEPT, RIGHT, LEFT)

    def text(self) -> str:
        return (
            f"rule={self.rule.name};right>{self.right_threshold:.8f};"
            f"right-left>{self.difference_threshold:.8f};"
            f"trigger={self.trigger_margin:.8f};base={self.base_replicates};"
            f"extra={self.extra_replicates};mapping="
            + ",".join(ACTION_NAMES[action] for action in self.mapping)
        )


@dataclass(frozen=True)
class AdaptiveEvaluation:
    accuracy: float
    invalid_transition_rate: float
    mean_queries: float
    mean_replicates_per_query: float
    resample_rate: float


@dataclass(frozen=True)
class SupportStratum:
    name: str
    dimensions: tuple[int, ...]
    rho_low: float
    rho_high: float
    trials: int


@dataclass(frozen=True)
class SupportEvaluation:
    minimum_accuracy: float
    mean_accuracy: float
    maximum_invalid_transition_rate: float
    maximum_mean_replicates_per_query: float
    mean_replicates_per_query: float
    rows: tuple[dict[str, object], ...]


def training_worlds() -> tuple[World, ...]:
    return tuple(
        World(dimension, root, rho)
        for dimension in range(2, 10)
        for rho in (0.24, 0.40, 0.65, 0.90)
        for root in range(dimension)
    )


def support_observations(
    seed: int,
    samples: int = 2_000,
) -> list[tuple[float | None, float | None]]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(samples):
        dimension = int(rng.integers(3, 10))
        root = int(rng.integers(0, dimension))
        query = int(rng.integers(0, dimension))
        rho = float(rng.uniform(0.24, 0.92))
        rows.append(
            intervention_response(
                seed + index * 65_537,
                dimension,
                root,
                rho,
                query,
                256,
            )
        )
    return rows


def candidate_policies(
    observations: list[tuple[float | None, float | None]],
) -> tuple[AdaptivePolicy, ...]:
    values = [
        abs(value)
        for left, right in observations
        for value in (left, right)
        if value is not None
    ]
    scale = float(np.quantile(values, 0.90))
    thresholds = tuple(
        sorted({fraction * scale for fraction in (0.14, 0.18, 0.22, 0.26)})
    )
    margins = tuple(
        sorted({fraction * scale for fraction in (0.05, 0.08, 0.11, 0.14)})
    )
    rule = next(rule for rule in position_rules() if rule.name == "lower_midpoint")
    return tuple(
        AdaptivePolicy(rule, right, difference, margin, 256, extra)
        for right in thresholds
        for difference in thresholds
        for margin in margins
        for extra in (256, 512)
    )


def encode(
    policy: AdaptivePolicy,
    left: float | None,
    right: float | None,
) -> int:
    right_bit = right is not None and right > policy.right_threshold
    difference_bit = (
        left is not None
        and right is not None
        and right - left > policy.difference_threshold
    )
    return int(right_bit) | (int(difference_bit) << 1)


def decode(
    policy: AdaptivePolicy,
    left: float | None,
    right: float | None,
) -> int:
    return policy.mapping[encode(policy, left, right)]


def _combine(
    first: float | None,
    second: float | None,
    first_count: int,
    second_count: int,
) -> float | None:
    if first is None or second is None:
        return None
    return (first_count * first + second_count * second) / (
        first_count + second_count
    )


def adaptive_response(
    seed: int,
    dimension: int,
    root: int,
    rho: float,
    query: int,
    policy: AdaptivePolicy,
) -> tuple[float | None, float | None, int, bool]:
    left, right = intervention_response(
        seed,
        dimension,
        root,
        rho,
        query,
        policy.base_replicates,
    )
    margins = []
    if right is not None:
        margins.append(abs(right - policy.right_threshold))
    if left is not None and right is not None:
        margins.append(abs((right - left) - policy.difference_threshold))
    should_resample = bool(margins) and min(margins) < policy.trigger_margin
    if not should_resample:
        return left, right, policy.base_replicates, False
    extra_left, extra_right = intervention_response(
        seed + 2_147_483_647,
        dimension,
        root,
        rho,
        query,
        policy.extra_replicates,
    )
    combined_left = _combine(
        left,
        extra_left,
        policy.base_replicates,
        policy.extra_replicates,
    )
    combined_right = _combine(
        right,
        extra_right,
        policy.base_replicates,
        policy.extra_replicates,
    )
    return (
        combined_left,
        combined_right,
        policy.base_replicates + policy.extra_replicates,
        True,
    )


def fixed_response(
    seed: int,
    dimension: int,
    root: int,
    rho: float,
    query: int,
    replicates: int,
) -> tuple[float | None, float | None, int, bool]:
    left, right = intervention_response(
        seed, dimension, root, rho, query, replicates
    )
    return left, right, replicates, False


def run_trial(
    seed: int,
    policy: AdaptivePolicy,
    dimension: int,
    root: int,
    rho: float,
    fixed_replicates: int | None = None,
) -> tuple[bool, bool, int, int, int]:
    low = 0
    high = dimension - 1
    queries = 0
    total_replicates = 0
    resamples = 0
    budget = information_lower_bound(dimension)
    while low < high and queries < budget:
        query = low + policy.rule.offset(high - low + 1)
        if fixed_replicates is None:
            left, right, used, resampled = adaptive_response(
                seed + queries * 7_919,
                dimension,
                root,
                rho,
                query,
                policy,
            )
        else:
            left, right, used, resampled = fixed_response(
                seed + queries * 7_919,
                dimension,
                root,
                rho,
                query,
                fixed_replicates,
            )
        total_replicates += used
        resamples += int(resampled)
        action = decode(policy, left, right)
        queries += 1
        if action == ACCEPT:
            return query == root, False, queries, total_replicates, resamples
        if action == LEFT:
            high = query - 1
        elif action == RIGHT:
            low = query + 1
        else:
            raise ValueError(action)
        if low > high:
            return False, True, queries, total_replicates, resamples
    remaining = max(1, high - low + 1)
    prediction = low + (remaining - 1) // 2
    return prediction == root, False, queries, total_replicates, resamples


def exact_training_success(policy: AdaptivePolicy) -> bool:
    for world in training_worlds():
        low = 0
        high = world.dimension - 1
        queries = 0
        while low < high and queries < information_lower_bound(world.dimension):
            query = low + policy.rule.offset(high - low + 1)
            left, right = noiseless_response(
                world.dimension, world.root, world.rho, query
            )
            action = decode(policy, left, right)
            queries += 1
            if action == ACCEPT:
                if query != world.root:
                    return False
                break
            if action == LEFT:
                high = query - 1
            elif action == RIGHT:
                low = query + 1
            if low > high:
                return False
        else:
            remaining = max(1, high - low + 1)
            prediction = low + (remaining - 1) // 2
            if prediction != world.root:
                return False
    return True


def evaluate(
    seed: int,
    policy: AdaptivePolicy,
    dimensions: tuple[int, ...],
    rho_low: float,
    rho_high: float,
    trials: int,
    fixed_replicates: int | None = None,
) -> AdaptiveEvaluation:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(trials):
        dimension = int(rng.choice(dimensions))
        root = int(rng.integers(0, dimension))
        rho = float(rng.uniform(rho_low, rho_high))
        rows.append(
            run_trial(
                seed + index * 104_729,
                policy,
                dimension,
                root,
                rho,
                fixed_replicates,
            )
        )
    total_queries = sum(row[2] for row in rows)
    return AdaptiveEvaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        invalid_transition_rate=float(np.mean([row[1] for row in rows])),
        mean_queries=float(np.mean([row[2] for row in rows])),
        mean_replicates_per_query=(
            sum(row[3] for row in rows) / max(total_queries, 1)
        ),
        resample_rate=(sum(row[4] for row in rows) / max(total_queries, 1)),
    )


def development_strata() -> tuple[SupportStratum, ...]:
    dimensions = (3, 4, 5, 6, 7, 8, 9, 11)
    return (
        SupportStratum("low", dimensions, 0.24, 0.34, 1_000),
        SupportStratum("transition", dimensions, 0.26, 0.42, 1_000),
        SupportStratum("medium", dimensions, 0.34, 0.60, 1_000),
        SupportStratum("high", dimensions, 0.60, 0.96, 1_000),
    )


def hidden_strata() -> tuple[SupportStratum, ...]:
    dimensions = (17, 31, 63, 127, 255)
    return (
        SupportStratum("hidden-low", dimensions, 0.24, 0.34, 2_600),
        SupportStratum("hidden-transition", dimensions, 0.26, 0.42, 2_600),
        SupportStratum("hidden-medium", dimensions, 0.34, 0.60, 2_600),
        SupportStratum("hidden-high", dimensions, 0.60, 0.96, 2_600),
    )


def evaluate_support(
    seed: int,
    policy: AdaptivePolicy,
    strata: tuple[SupportStratum, ...],
    fixed_replicates: int | None = None,
) -> SupportEvaluation:
    rows = []
    for index, stratum in enumerate(strata):
        evaluation = evaluate(
            seed + index * 1_000_003,
            policy,
            stratum.dimensions,
            stratum.rho_low,
            stratum.rho_high,
            stratum.trials,
            fixed_replicates,
        )
        rows.append(
            {
                "name": stratum.name,
                "accuracy": evaluation.accuracy,
                "invalid_transition_rate": evaluation.invalid_transition_rate,
                "mean_queries": evaluation.mean_queries,
                "mean_replicates_per_query": evaluation.mean_replicates_per_query,
                "resample_rate": evaluation.resample_rate,
            }
        )
    return SupportEvaluation(
        minimum_accuracy=min(float(row["accuracy"]) for row in rows),
        mean_accuracy=float(np.mean([float(row["accuracy"]) for row in rows])),
        maximum_invalid_transition_rate=max(
            float(row["invalid_transition_rate"]) for row in rows
        ),
        maximum_mean_replicates_per_query=max(
            float(row["mean_replicates_per_query"]) for row in rows
        ),
        mean_replicates_per_query=float(
            np.mean([float(row["mean_replicates_per_query"]) for row in rows])
        ),
        rows=tuple(rows),
    )


def search_policy(
    seed: int,
) -> tuple[AdaptivePolicy, SupportEvaluation, dict[str, object]]:
    observations = support_observations(seed)
    candidates = [
        policy
        for policy in candidate_policies(observations)
        if exact_training_success(policy)
    ]
    strata = development_strata()
    evaluated = [
        (policy, evaluate_support(seed + 2_000_003, policy, strata))
        for policy in candidates
    ]
    best_minimum = max(score.minimum_accuracy for _, score in evaluated)
    near = [
        (policy, score)
        for policy, score in evaluated
        if score.minimum_accuracy >= best_minimum - 0.001
    ]
    best_mean = max(score.mean_accuracy for _, score in near)
    near_mean = [
        (policy, score)
        for policy, score in near
        if score.mean_accuracy >= best_mean - 0.001
    ]
    selected, score = min(
        near_mean,
        key=lambda item: (
            item[1].mean_replicates_per_query,
            item[0].extra_replicates,
            item[0].right_threshold,
            item[0].difference_threshold,
            item[0].trigger_margin,
        ),
    )
    evidence = {
        "supervision": "terminal_root_success_only",
        "step_action_labels_used": False,
        "selection_objective": "worst_support_accuracy_then_mean_then_samples",
        "candidate_count": len(candidates),
        "selected_policy": selected.text(),
        "development_worst_accuracy": score.minimum_accuracy,
        "development_mean_accuracy": score.mean_accuracy,
        "development_mean_replicates_per_query": score.mean_replicates_per_query,
        "development_rows": list(score.rows),
    }
    return selected, score, evidence


def random_mapping_control(
    seed: int,
    policy: AdaptivePolicy,
    strata: tuple[SupportStratum, ...],
) -> dict[str, float]:
    rng = random.Random(seed)
    scores = []
    for index in range(24):
        mapping = tuple(
            rng.choice((LEFT, ACCEPT, RIGHT)) for _ in range(4)
        )
        candidate = AdaptivePolicy(
            policy.rule,
            policy.right_threshold,
            policy.difference_threshold,
            policy.trigger_margin,
            policy.base_replicates,
            policy.extra_replicates,
            mapping,
        )
        reduced_strata = tuple(
            SupportStratum(
                stratum.name,
                stratum.dimensions,
                stratum.rho_low,
                stratum.rho_high,
                520,
            )
            for stratum in strata
        )
        score = evaluate_support(
            seed + index * 1_000_003,
            candidate,
            reduced_strata,
        )
        scores.append(score.minimum_accuracy)
    return {
        "trials": len(scores),
        "median_worst_accuracy": float(np.median(scores)),
        "maximum_worst_accuracy": max(scores),
    }


def fixed_specialist(policy: AdaptivePolicy) -> AdaptivePolicy:
    return AdaptivePolicy(
        policy.rule,
        0.15,
        0.15,
        0.0,
        768,
        0,
        policy.mapping,
    )


def wrong_query_policy(policy: AdaptivePolicy) -> AdaptivePolicy:
    rule = next(rule for rule in position_rules() if rule.name == "first")
    return AdaptivePolicy(
        rule,
        policy.right_threshold,
        policy.difference_threshold,
        policy.trigger_margin,
        policy.base_replicates,
        policy.extra_replicates,
        policy.mapping,
    )


def digest(policy: AdaptivePolicy) -> str:
    return hashlib.sha256(policy.text().encode("utf-8")).hexdigest()


def serialize(policy: AdaptivePolicy) -> dict[str, object]:
    return {
        "rule": policy.rule.name,
        "right_threshold": policy.right_threshold,
        "difference_threshold": policy.difference_threshold,
        "trigger_margin": policy.trigger_margin,
        "base_replicates": policy.base_replicates,
        "extra_replicates": policy.extra_replicates,
        "mapping": [ACTION_NAMES[action] for action in policy.mapping],
    }


def run(seed: int = 1301) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    policy, development, search = search_policy(seed * 10_000 + 97)
    frozen_digest = digest(policy)

    # Hidden support strata are created only after the adaptive policy, terminal
    # objective and sample accounting are frozen.
    strata = hidden_strata()
    candidate = evaluate_support(
        seed * 10_000 + 13_000_001,
        policy,
        strata,
    )
    fixed_256 = evaluate_support(
        seed * 10_000 + 13_000_003,
        policy,
        strata,
        fixed_replicates=256,
    )
    fixed_512 = evaluate_support(
        seed * 10_000 + 13_000_005,
        policy,
        strata,
        fixed_replicates=512,
    )
    specialist_768 = evaluate_support(
        seed * 10_000 + 13_000_007,
        fixed_specialist(policy),
        strata,
        fixed_replicates=768,
    )
    wrong_query = evaluate_support(
        seed * 10_000 + 13_000_009,
        wrong_query_policy(policy),
        strata,
    )
    random_control = random_mapping_control(
        seed * 10_000 + 13_000_011,
        policy,
        strata,
    )

    fixed_256_gap = candidate.minimum_accuracy - fixed_256.minimum_accuracy
    fixed_512_gap = candidate.minimum_accuracy - fixed_512.minimum_accuracy
    specialist_gap = candidate.minimum_accuracy - specialist_768.minimum_accuracy
    query_gap = candidate.minimum_accuracy - wrong_query.minimum_accuracy
    random_gap = (
        candidate.minimum_accuracy - random_control["median_worst_accuracy"]
    )
    sample_saving = 1.0 - candidate.mean_replicates_per_query / 768.0
    dimensions = strata[0].dimensions
    query_optimality = all(
        optimal_worst_case_queries(dimension)
        == information_lower_bound(dimension)
        for dimension in dimensions
    )
    gate = (
        bool(equivalence["exact_within_tolerance"])
        and search["step_action_labels_used"] is False
        and exact_training_success(policy)
        and policy.rule.name == "lower_midpoint"
        and development.minimum_accuracy >= 0.985
        and candidate.minimum_accuracy >= 0.985
        and candidate.maximum_invalid_transition_rate <= 0.01
        and candidate.maximum_mean_replicates_per_query <= 600.0
        and fixed_256_gap >= 0.03
        and fixed_512_gap >= 0.005
        and specialist_gap >= -0.01
        and query_gap >= 0.20
        and random_gap >= 0.45
        and sample_saving >= 0.20
        and query_optimality
    )
    return {
        "status": (
            "adaptive_measurement_efficiency_candidate" if gate else "not_yet"
        ),
        "claim_scope": (
            "a terminal-reward search synthesizes confidence-triggered extra "
            "measurement that preserves deep-chain accuracy while spending fewer "
            "samples than a fixed high-budget controller; the causal family and "
            "measurement grammar remain human supplied, so this is not a world "
            "breakthrough"
        ),
        "seed": seed,
        "candidate_gate": gate,
        "observational_equivalence": equivalence,
        "search": search,
        "selected_policy": serialize(policy),
        "frozen_policy_digest": frozen_digest,
        "hidden_dimensions": list(dimensions),
        "hidden_query_optimality": query_optimality,
        "candidate": candidate.__dict__,
        "fixed_256_control": fixed_256.__dict__,
        "fixed_512_control": fixed_512.__dict__,
        "specialist_768_control": specialist_768.__dict__,
        "wrong_query_control": wrong_query.__dict__,
        "random_mapping_control": random_control,
        "fixed_256_gap": fixed_256_gap,
        "fixed_512_gap": fixed_512_gap,
        "specialist_gap": specialist_gap,
        "wrong_query_gap": query_gap,
        "random_gap": random_gap,
        "sample_saving_vs_768": sample_saving,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "policy": report["selected_policy"],
                "hidden_worst": report["candidate"]["minimum_accuracy"],
                "mean_replicates": report["candidate"][
                    "mean_replicates_per_query"
                ],
                "sample_saving": report["sample_saving_vs_768"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
