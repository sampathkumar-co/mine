from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from .intervention_genesis_v22 import (
    PositionRule,
    information_lower_bound,
    observational_equivalence_certificate,
    optimal_worst_case_queries,
    position_rules,
)
from .outcome_alphabet_v23 import ACCEPT, ACTION_NAMES, LEFT, RIGHT
from .terminal_reward_v26 import World, noiseless_response


AGGREGATORS = ("mean", "trimmed_mean_10", "median_of_means_8")


@dataclass(frozen=True)
class RobustPolicy:
    rule: PositionRule
    aggregator: str
    right_threshold: float
    difference_threshold: float
    trigger_margin: float
    base_replicates: int = 256
    extra_replicates: int = 512
    mapping: tuple[int, ...] = (RIGHT, ACCEPT, RIGHT, LEFT)

    def text(self) -> str:
        return (
            f"rule={self.rule.name};aggregator={self.aggregator};"
            f"right>{self.right_threshold:.8f};"
            f"right-left>{self.difference_threshold:.8f};"
            f"trigger={self.trigger_margin:.8f};base={self.base_replicates};"
            f"extra={self.extra_replicates};mapping="
            + ",".join(ACTION_NAMES[action] for action in self.mapping)
        )


@dataclass(frozen=True)
class NoiseStratum:
    name: str
    family: str
    dimensions: tuple[int, ...]
    rho_low: float
    rho_high: float
    trials: int


@dataclass(frozen=True)
class TrialEvaluation:
    accuracy: float
    invalid_transition_rate: float
    mean_queries: float
    mean_replicates_per_query: float
    resample_rate: float


@dataclass(frozen=True)
class StrataEvaluation:
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


def aggregate(values: np.ndarray, method: str) -> float:
    if method == "mean":
        return float(np.mean(values))
    if method == "trimmed_mean_10":
        ordered = np.sort(values)
        trim = max(1, int(0.10 * len(ordered)))
        return float(np.mean(ordered[trim:-trim]))
    if method == "median_of_means_8":
        groups = np.array_split(values, 8)
        return float(np.median([np.mean(group) for group in groups]))
    raise ValueError(method)


def _standard_noise(
    rng: np.random.Generator,
    family: str,
    size: int,
) -> np.ndarray:
    if family == "gaussian":
        return rng.normal(size=size)
    if family == "laplace":
        return rng.laplace(scale=1.0 / math.sqrt(2.0), size=size)
    if family == "student5":
        return rng.standard_t(5, size=size) * math.sqrt(3.0 / 5.0)
    if family == "student3":
        return rng.standard_t(3, size=size) / math.sqrt(3.0)
    if family == "contaminated2":
        mask = rng.random(size) < 0.02
        values = rng.normal(size=size)
        values[mask] = rng.normal(scale=7.0, size=int(np.sum(mask)))
        return values / math.sqrt(0.98 + 0.02 * 49.0)
    if family == "contaminated5":
        mask = rng.random(size) < 0.05
        values = rng.normal(size=size)
        values[mask] = rng.normal(scale=8.0, size=int(np.sum(mask)))
        return values / math.sqrt(0.95 + 0.05 * 64.0)
    if family == "skewed":
        return rng.exponential(size=size) - 1.0
    if family == "variance_mixture":
        mask = rng.random(size) < 0.20
        scales = np.where(mask, 2.0, 0.5)
        normalization = math.sqrt(0.20 * 4.0 + 0.80 * 0.25)
        return rng.normal(size=size) * scales / normalization
    raise ValueError(family)


def raw_neighbor_samples(
    rng: np.random.Generator,
    rho: float,
    active: bool,
    replicates: int,
    family: str,
) -> np.ndarray:
    mean = rho if active else 0.0
    variance = 1.0 - rho * rho if active else 1.0
    return mean + math.sqrt(variance) * _standard_noise(rng, family, replicates)


def response_samples(
    seed: int,
    dimension: int,
    root: int,
    rho: float,
    query: int,
    replicates: int,
    family: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    left = None
    right = None
    if query > 0:
        left = raw_neighbor_samples(
            rng,
            rho,
            active=root >= query,
            replicates=replicates,
            family=family,
        )
    if query < dimension - 1:
        right = raw_neighbor_samples(
            rng,
            rho,
            active=root <= query,
            replicates=replicates,
            family=family,
        )
    return left, right


def summarize(
    samples: np.ndarray | None,
    aggregator: str,
) -> float | None:
    return None if samples is None else aggregate(samples, aggregator)


def unsupervised_scale(
    seed: int,
    samples: int = 900,
) -> float:
    rng = np.random.default_rng(seed)
    values = []
    for index in range(samples):
        dimension = int(rng.integers(3, 10))
        root = int(rng.integers(0, dimension))
        query = int(rng.integers(0, dimension))
        rho = float(rng.uniform(0.24, 0.92))
        left, right = response_samples(
            seed + index * 65_537,
            dimension,
            root,
            rho,
            query,
            256,
            "gaussian",
        )
        for raw in (left, right):
            if raw is not None:
                values.append(abs(float(np.mean(raw))))
    return float(np.quantile(values, 0.90))


def candidate_policies(seed: int) -> tuple[RobustPolicy, ...]:
    scale = unsupervised_scale(seed)
    thresholds = tuple(fraction * scale for fraction in (0.14, 0.18, 0.22))
    margins = tuple(fraction * scale for fraction in (0.08, 0.11))
    rule = next(rule for rule in position_rules() if rule.name == "lower_midpoint")
    return tuple(
        RobustPolicy(rule, aggregator, threshold, threshold, margin)
        for aggregator in AGGREGATORS
        for threshold in thresholds
        for margin in margins
    )


def encode(
    policy: RobustPolicy,
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
    policy: RobustPolicy,
    left: float | None,
    right: float | None,
) -> int:
    return policy.mapping[encode(policy, left, right)]


def exact_training_success(policy: RobustPolicy) -> bool:
    for world in training_worlds():
        low = 0
        high = world.dimension - 1
        queries = 0
        budget = information_lower_bound(world.dimension)
        while low < high and queries < budget:
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


def measured_response(
    seed: int,
    dimension: int,
    root: int,
    rho: float,
    query: int,
    family: str,
    policy: RobustPolicy,
    fixed_replicates: int | None = None,
) -> tuple[float | None, float | None, int, bool]:
    initial = policy.base_replicates if fixed_replicates is None else fixed_replicates
    left_samples, right_samples = response_samples(
        seed,
        dimension,
        root,
        rho,
        query,
        initial,
        family,
    )
    left = summarize(left_samples, policy.aggregator)
    right = summarize(right_samples, policy.aggregator)
    if fixed_replicates is not None:
        return left, right, fixed_replicates, False

    margins = []
    if right is not None:
        margins.append(abs(right - policy.right_threshold))
    if left is not None and right is not None:
        margins.append(abs((right - left) - policy.difference_threshold))
    resample = bool(margins) and min(margins) < policy.trigger_margin
    if not resample:
        return left, right, policy.base_replicates, False

    extra_left, extra_right = response_samples(
        seed + 2_147_483_647,
        dimension,
        root,
        rho,
        query,
        policy.extra_replicates,
        family,
    )
    if left_samples is not None and extra_left is not None:
        left_samples = np.concatenate((left_samples, extra_left))
    if right_samples is not None and extra_right is not None:
        right_samples = np.concatenate((right_samples, extra_right))
    return (
        summarize(left_samples, policy.aggregator),
        summarize(right_samples, policy.aggregator),
        policy.base_replicates + policy.extra_replicates,
        True,
    )


def run_trial(
    seed: int,
    policy: RobustPolicy,
    dimension: int,
    root: int,
    rho: float,
    family: str,
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
        left, right, used, resampled = measured_response(
            seed + queries * 7_919,
            dimension,
            root,
            rho,
            query,
            family,
            policy,
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
        if low > high:
            return False, True, queries, total_replicates, resamples
    remaining = max(1, high - low + 1)
    prediction = low + (remaining - 1) // 2
    return prediction == root, False, queries, total_replicates, resamples


def evaluate_trials(
    seed: int,
    policy: RobustPolicy,
    stratum: NoiseStratum,
    fixed_replicates: int | None = None,
) -> TrialEvaluation:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(stratum.trials):
        dimension = int(rng.choice(stratum.dimensions))
        root = int(rng.integers(0, dimension))
        rho = float(rng.uniform(stratum.rho_low, stratum.rho_high))
        rows.append(
            run_trial(
                seed + index * 104_729,
                policy,
                dimension,
                root,
                rho,
                stratum.family,
                fixed_replicates,
            )
        )
    total_queries = sum(row[2] for row in rows)
    return TrialEvaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        invalid_transition_rate=float(np.mean([row[1] for row in rows])),
        mean_queries=float(np.mean([row[2] for row in rows])),
        mean_replicates_per_query=sum(row[3] for row in rows) / max(total_queries, 1),
        resample_rate=sum(row[4] for row in rows) / max(total_queries, 1),
    )


def evaluate_strata(
    seed: int,
    policy: RobustPolicy,
    strata: tuple[NoiseStratum, ...],
    fixed_replicates: int | None = None,
) -> StrataEvaluation:
    rows = []
    for index, stratum in enumerate(strata):
        evaluation = evaluate_trials(
            seed + index * 1_000_003,
            policy,
            stratum,
            fixed_replicates,
        )
        rows.append(
            {
                "name": stratum.name,
                "family": stratum.family,
                "accuracy": evaluation.accuracy,
                "invalid_transition_rate": evaluation.invalid_transition_rate,
                "mean_queries": evaluation.mean_queries,
                "mean_replicates_per_query": evaluation.mean_replicates_per_query,
                "resample_rate": evaluation.resample_rate,
            }
        )
    return StrataEvaluation(
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


def development_strata() -> tuple[NoiseStratum, ...]:
    dimensions = (3, 4, 5, 6, 7, 8, 9, 11)
    return (
        NoiseStratum("gaussian", "gaussian", dimensions, 0.24, 0.96, 400),
        NoiseStratum("laplace", "laplace", dimensions, 0.24, 0.96, 400),
        NoiseStratum("student5", "student5", dimensions, 0.24, 0.96, 400),
        NoiseStratum("contaminated2", "contaminated2", dimensions, 0.24, 0.96, 400),
    )


def hidden_strata() -> tuple[NoiseStratum, ...]:
    dimensions = (17, 31, 63, 127, 255)
    return (
        NoiseStratum("student3", "student3", dimensions, 0.24, 0.96, 1_000),
        NoiseStratum("contaminated5", "contaminated5", dimensions, 0.24, 0.96, 1_000),
        NoiseStratum("skewed", "skewed", dimensions, 0.24, 0.96, 1_000),
        NoiseStratum("variance-mixture", "variance_mixture", dimensions, 0.24, 0.96, 1_000),
    )


def search_policy(seed: int) -> tuple[RobustPolicy, StrataEvaluation, dict[str, object]]:
    candidates = [policy for policy in candidate_policies(seed) if exact_training_success(policy)]
    strata = development_strata()
    evaluated = [
        (policy, evaluate_strata(seed + 2_000_003, policy, strata))
        for policy in candidates
    ]
    best_minimum = max(score.minimum_accuracy for _, score in evaluated)
    near = [
        (policy, score)
        for policy, score in evaluated
        if score.minimum_accuracy >= best_minimum - 0.002
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
            AGGREGATORS.index(item[0].aggregator),
            item[0].right_threshold,
            item[0].trigger_margin,
        ),
    )
    return selected, score, {
        "supervision": "terminal_root_success_only",
        "step_action_labels_used": False,
        "selection_objective": "worst_seen_noise_then_mean_then_samples",
        "candidate_count": len(candidates),
        "selected_aggregator": selected.aggregator,
        "selected_policy": selected.text(),
        "development_worst_accuracy": score.minimum_accuracy,
        "development_mean_accuracy": score.mean_accuracy,
        "development_mean_replicates_per_query": score.mean_replicates_per_query,
        "development_rows": list(score.rows),
    }


def with_aggregator(policy: RobustPolicy, aggregator: str) -> RobustPolicy:
    return RobustPolicy(
        policy.rule,
        aggregator,
        policy.right_threshold,
        policy.difference_threshold,
        policy.trigger_margin,
        policy.base_replicates,
        policy.extra_replicates,
        policy.mapping,
    )


def wrong_query(policy: RobustPolicy) -> RobustPolicy:
    rule = next(rule for rule in position_rules() if rule.name == "first")
    return RobustPolicy(
        rule,
        policy.aggregator,
        policy.right_threshold,
        policy.difference_threshold,
        policy.trigger_margin,
        policy.base_replicates,
        policy.extra_replicates,
        policy.mapping,
    )


def random_mapping_control(
    seed: int,
    policy: RobustPolicy,
    strata: tuple[NoiseStratum, ...],
) -> dict[str, float]:
    rng = random.Random(seed)
    scores = []
    reduced = tuple(
        NoiseStratum(
            stratum.name,
            stratum.family,
            stratum.dimensions,
            stratum.rho_low,
            stratum.rho_high,
            180,
        )
        for stratum in strata
    )
    for index in range(12):
        mapping = tuple(rng.choice((LEFT, ACCEPT, RIGHT)) for _ in range(4))
        candidate = RobustPolicy(
            policy.rule,
            policy.aggregator,
            policy.right_threshold,
            policy.difference_threshold,
            policy.trigger_margin,
            policy.base_replicates,
            policy.extra_replicates,
            mapping,
        )
        score = evaluate_strata(seed + index * 1_000_003, candidate, reduced)
        scores.append(score.minimum_accuracy)
    return {
        "trials": len(scores),
        "median_worst_accuracy": float(np.median(scores)),
        "maximum_worst_accuracy": max(scores),
    }


def digest(policy: RobustPolicy) -> str:
    return hashlib.sha256(policy.text().encode("utf-8")).hexdigest()


def serialize(policy: RobustPolicy) -> dict[str, object]:
    return {
        "rule": policy.rule.name,
        "aggregator": policy.aggregator,
        "right_threshold": policy.right_threshold,
        "difference_threshold": policy.difference_threshold,
        "trigger_margin": policy.trigger_margin,
        "base_replicates": policy.base_replicates,
        "extra_replicates": policy.extra_replicates,
        "mapping": [ACTION_NAMES[action] for action in policy.mapping],
    }


def run(seed: int = 1401) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    policy, development, search = search_policy(seed * 10_000 + 101)
    frozen_digest = digest(policy)

    # Unseen noise laws and larger dimensions are instantiated only after the
    # estimator, thresholds, trigger rule and controller have been frozen.
    strata = hidden_strata()
    candidate = evaluate_strata(seed * 10_000 + 14_000_001, policy, strata)
    mean_adaptive = evaluate_strata(
        seed * 10_000 + 14_000_003,
        with_aggregator(policy, "mean"),
        strata,
    )
    robust_fixed_1024 = evaluate_strata(
        seed * 10_000 + 14_000_005,
        policy,
        strata,
        fixed_replicates=1024,
    )
    mean_fixed_1024 = evaluate_strata(
        seed * 10_000 + 14_000_007,
        with_aggregator(policy, "mean"),
        strata,
        fixed_replicates=1024,
    )
    query_control = evaluate_strata(
        seed * 10_000 + 14_000_009,
        wrong_query(policy),
        strata,
    )
    random_control = random_mapping_control(
        seed * 10_000 + 14_000_011,
        policy,
        strata,
    )

    mean_gap = candidate.minimum_accuracy - mean_adaptive.minimum_accuracy
    robust_fixed_gap = candidate.minimum_accuracy - robust_fixed_1024.minimum_accuracy
    mean_fixed_gap = candidate.minimum_accuracy - mean_fixed_1024.minimum_accuracy
    query_gap = candidate.minimum_accuracy - query_control.minimum_accuracy
    random_gap = candidate.minimum_accuracy - random_control["median_worst_accuracy"]
    sample_saving = 1.0 - candidate.mean_replicates_per_query / 1024.0
    dimensions = strata[0].dimensions
    query_optimality = all(
        optimal_worst_case_queries(dimension) == information_lower_bound(dimension)
        for dimension in dimensions
    )
    gate = (
        bool(equivalence["exact_within_tolerance"])
        and search["step_action_labels_used"] is False
        and policy.aggregator != "mean"
        and exact_training_success(policy)
        and policy.rule.name == "lower_midpoint"
        and development.minimum_accuracy >= 0.985
        and candidate.minimum_accuracy >= 0.98
        and candidate.maximum_invalid_transition_rate <= 0.01
        and candidate.maximum_mean_replicates_per_query <= 720.0
        and mean_gap >= 0.02
        and robust_fixed_gap >= -0.01
        and mean_fixed_gap >= 0.0
        and query_gap >= 0.20
        and random_gap >= 0.45
        and sample_saving >= 0.25
        and query_optimality
    )
    return {
        "status": "noise_law_transfer_candidate" if gate else "not_yet",
        "claim_scope": (
            "terminal-reward synthesis selects a robust estimator and adaptive "
            "measurement rule on several noise laws, freezes it, and transfers "
            "to unseen heavy-tailed, contaminated, skewed and heteroscedastic "
            "noise while saving samples; the estimator grammar and causal family "
            "remain human supplied, so this is not a world breakthrough"
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
        "mean_adaptive_control": mean_adaptive.__dict__,
        "robust_fixed_1024_control": robust_fixed_1024.__dict__,
        "mean_fixed_1024_control": mean_fixed_1024.__dict__,
        "wrong_query_control": query_control.__dict__,
        "random_mapping_control": random_control,
        "mean_adaptive_gap": mean_gap,
        "robust_fixed_gap": robust_fixed_gap,
        "mean_fixed_gap": mean_fixed_gap,
        "wrong_query_gap": query_gap,
        "random_gap": random_gap,
        "sample_saving_vs_1024": sample_saving,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "aggregator": report["selected_policy"]["aggregator"],
                "development_worst": report["search"]["development_worst_accuracy"],
                "hidden_worst": report["candidate"]["minimum_accuracy"],
                "mean_gap": report["mean_adaptive_gap"],
                "sample_saving": report["sample_saving_vs_1024"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
