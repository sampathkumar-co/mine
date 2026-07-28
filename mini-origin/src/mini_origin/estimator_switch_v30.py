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
from .noise_transfer_v29 import aggregate, unsupervised_scale
from .outcome_alphabet_v23 import ACCEPT, ACTION_NAMES, LEFT, RIGHT
from .terminal_reward_v26 import World, noiseless_response


ROBUST_AGGREGATORS = ("trimmed_mean_10", "median_of_means_8")


@dataclass(frozen=True)
class SwitchPolicy:
    rule: PositionRule
    robust_aggregator: str
    skew_threshold: float
    tail_threshold: float
    right_threshold: float
    difference_threshold: float
    trigger_margin: float
    base_replicates: int = 256
    extra_replicates: int = 512
    mapping: tuple[int, ...] = (RIGHT, ACCEPT, RIGHT, LEFT)

    def text(self) -> str:
        return (
            f"rule={self.rule.name};robust={self.robust_aggregator};"
            f"skew>{self.skew_threshold:.6f};tail>{self.tail_threshold:.6f};"
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
    robust_estimator_rate: float


@dataclass(frozen=True)
class StrataEvaluation:
    minimum_accuracy: float
    mean_accuracy: float
    maximum_invalid_transition_rate: float
    maximum_mean_replicates_per_query: float
    mean_replicates_per_query: float
    minimum_robust_estimator_rate: float
    maximum_robust_estimator_rate: float
    rows: tuple[dict[str, object], ...]


def training_worlds() -> tuple[World, ...]:
    return tuple(
        World(dimension, root, rho)
        for dimension in range(2, 10)
        for rho in (0.24, 0.40, 0.65, 0.90)
        for root in range(dimension)
    )


def _standard_noise(
    rng: np.random.Generator,
    family: str,
    size: int,
) -> np.ndarray:
    if family == "gaussian":
        return rng.normal(size=size)
    if family == "student3":
        return rng.standard_t(3, size=size) / math.sqrt(3.0)
    if family == "contaminated2x20":
        mask = rng.random(size) < 0.02
        values = rng.normal(size=size)
        values[mask] = rng.normal(scale=20.0, size=int(np.sum(mask)))
        return values
    if family == "skewed_exponential":
        return rng.exponential(size=size) - 1.0
    if family == "student2_5":
        return rng.standard_t(2.5, size=size) * math.sqrt(0.5 / 2.5)
    if family == "contaminated1x40":
        mask = rng.random(size) < 0.01
        values = rng.normal(size=size)
        values[mask] = rng.normal(scale=40.0, size=int(np.sum(mask)))
        return values
    if family == "centered_lognormal":
        sigma = 0.8
        values = rng.lognormal(mean=0.0, sigma=sigma, size=size)
        mean = math.exp(0.5 * sigma * sigma)
        variance = (math.exp(sigma * sigma) - 1.0) * math.exp(sigma * sigma)
        return (values - mean) / math.sqrt(variance)
    if family == "asymmetric_mixture":
        mask = rng.random(size) < 0.05
        values = rng.normal(size=size)
        values[mask] += 14.0
        return (values - 0.70) / math.sqrt(0.95 + 0.05 * 197.0 - 0.70 * 0.70)
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
            rng, rho, root >= query, replicates, family
        )
    if query < dimension - 1:
        right = raw_neighbor_samples(
            rng, rho, root <= query, replicates, family
        )
    return left, right


def sample_diagnostics(values: np.ndarray) -> tuple[float, float]:
    q01, q25, q50, q75, q99 = np.quantile(
        values, (0.01, 0.25, 0.50, 0.75, 0.99)
    )
    iqr = max(float(q75 - q25), 1e-12)
    bowley = abs(float(q75 + q25 - 2.0 * q50)) / iqr
    upper = max(float(q99 - q50), 0.0)
    lower = max(float(q50 - q01), 0.0)
    tail_asymmetry = abs(upper - lower) / max(upper + lower, 1e-12)
    skew = max(bowley, tail_asymmetry)
    tail = float(q99 - q01) / iqr
    return skew, tail


def switched_estimate(
    values: np.ndarray,
    policy: SwitchPolicy,
    mode: str = "switch",
) -> tuple[float, bool]:
    if mode == "mean":
        return float(np.mean(values)), False
    if mode == "robust":
        return aggregate(values, policy.robust_aggregator), True
    skew, tail = sample_diagnostics(values)
    use_robust = skew <= policy.skew_threshold and tail > policy.tail_threshold
    if use_robust:
        return aggregate(values, policy.robust_aggregator), True
    return float(np.mean(values)), False


def candidate_policies(seed: int) -> tuple[SwitchPolicy, ...]:
    scale = unsupervised_scale(seed)
    thresholds = tuple(fraction * scale for fraction in (0.14, 0.18))
    margins = tuple(fraction * scale for fraction in (0.08, 0.11))
    rule = next(rule for rule in position_rules() if rule.name == "lower_midpoint")
    return tuple(
        SwitchPolicy(
            rule,
            robust,
            skew_threshold,
            tail_threshold,
            threshold,
            threshold,
            margin,
        )
        for robust in ROBUST_AGGREGATORS
        for skew_threshold in (0.15, 0.25, 0.35)
        for tail_threshold in (4.5, 6.0, 8.0)
        for threshold in thresholds
        for margin in margins
    )


def encode(
    policy: SwitchPolicy,
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
    policy: SwitchPolicy,
    left: float | None,
    right: float | None,
) -> int:
    return policy.mapping[encode(policy, left, right)]


def exact_training_success(policy: SwitchPolicy) -> bool:
    for world in training_worlds():
        low, high, queries = 0, world.dimension - 1, 0
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
            prediction = low + (max(1, high - low + 1) - 1) // 2
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
    policy: SwitchPolicy,
    estimator_mode: str = "switch",
    fixed_replicates: int | None = None,
) -> tuple[float | None, float | None, int, bool, int, int]:
    initial = policy.base_replicates if fixed_replicates is None else fixed_replicates
    left_samples, right_samples = response_samples(
        seed, dimension, root, rho, query, initial, family
    )

    def estimate(raw: np.ndarray | None) -> tuple[float | None, bool]:
        if raw is None:
            return None, False
        return switched_estimate(raw, policy, estimator_mode)

    left, left_robust = estimate(left_samples)
    right, right_robust = estimate(right_samples)
    estimate_count = int(left is not None) + int(right is not None)
    robust_count = int(left_robust) + int(right_robust)
    if fixed_replicates is not None:
        return left, right, fixed_replicates, False, robust_count, estimate_count

    margins = []
    if right is not None:
        margins.append(abs(right - policy.right_threshold))
    if left is not None and right is not None:
        margins.append(abs((right - left) - policy.difference_threshold))
    resample = bool(margins) and min(margins) < policy.trigger_margin
    if not resample:
        return left, right, policy.base_replicates, False, robust_count, estimate_count

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
    left, left_robust = estimate(left_samples)
    right, right_robust = estimate(right_samples)
    estimate_count = int(left is not None) + int(right is not None)
    robust_count = int(left_robust) + int(right_robust)
    return (
        left,
        right,
        policy.base_replicates + policy.extra_replicates,
        True,
        robust_count,
        estimate_count,
    )


def run_trial(
    seed: int,
    policy: SwitchPolicy,
    dimension: int,
    root: int,
    rho: float,
    family: str,
    estimator_mode: str = "switch",
    fixed_replicates: int | None = None,
) -> tuple[bool, bool, int, int, int, int, int]:
    low, high, queries = 0, dimension - 1, 0
    total_replicates = resamples = robust_count = estimate_count = 0
    budget = information_lower_bound(dimension)
    while low < high and queries < budget:
        query = low + policy.rule.offset(high - low + 1)
        left, right, used, resampled, robust, estimates = measured_response(
            seed + queries * 7_919,
            dimension,
            root,
            rho,
            query,
            family,
            policy,
            estimator_mode,
            fixed_replicates,
        )
        total_replicates += used
        resamples += int(resampled)
        robust_count += robust
        estimate_count += estimates
        action = decode(policy, left, right)
        queries += 1
        if action == ACCEPT:
            return query == root, False, queries, total_replicates, resamples, robust_count, estimate_count
        if action == LEFT:
            high = query - 1
        elif action == RIGHT:
            low = query + 1
        if low > high:
            return False, True, queries, total_replicates, resamples, robust_count, estimate_count
    prediction = low + (max(1, high - low + 1) - 1) // 2
    return prediction == root, False, queries, total_replicates, resamples, robust_count, estimate_count


def evaluate_trials(
    seed: int,
    policy: SwitchPolicy,
    stratum: NoiseStratum,
    estimator_mode: str = "switch",
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
                estimator_mode,
                fixed_replicates,
            )
        )
    total_queries = sum(row[2] for row in rows)
    total_estimates = sum(row[6] for row in rows)
    return TrialEvaluation(
        accuracy=float(np.mean([row[0] for row in rows])),
        invalid_transition_rate=float(np.mean([row[1] for row in rows])),
        mean_queries=float(np.mean([row[2] for row in rows])),
        mean_replicates_per_query=sum(row[3] for row in rows) / max(total_queries, 1),
        resample_rate=sum(row[4] for row in rows) / max(total_queries, 1),
        robust_estimator_rate=sum(row[5] for row in rows) / max(total_estimates, 1),
    )


def evaluate_strata(
    seed: int,
    policy: SwitchPolicy,
    strata: tuple[NoiseStratum, ...],
    estimator_mode: str = "switch",
    fixed_replicates: int | None = None,
) -> StrataEvaluation:
    rows = []
    for index, stratum in enumerate(strata):
        evaluation = evaluate_trials(
            seed + index * 1_000_003,
            policy,
            stratum,
            estimator_mode,
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
                "robust_estimator_rate": evaluation.robust_estimator_rate,
            }
        )
    robust_rates = [float(row["robust_estimator_rate"]) for row in rows]
    return StrataEvaluation(
        minimum_accuracy=min(float(row["accuracy"]) for row in rows),
        mean_accuracy=float(np.mean([float(row["accuracy"]) for row in rows])),
        maximum_invalid_transition_rate=max(float(row["invalid_transition_rate"]) for row in rows),
        maximum_mean_replicates_per_query=max(float(row["mean_replicates_per_query"]) for row in rows),
        mean_replicates_per_query=float(np.mean([float(row["mean_replicates_per_query"]) for row in rows])),
        minimum_robust_estimator_rate=min(robust_rates),
        maximum_robust_estimator_rate=max(robust_rates),
        rows=tuple(rows),
    )


def development_strata() -> tuple[NoiseStratum, ...]:
    dimensions = (3, 4, 5, 6, 7, 8, 9, 11)
    return (
        NoiseStratum("gaussian", "gaussian", dimensions, 0.24, 0.96, 220),
        NoiseStratum("student3", "student3", dimensions, 0.24, 0.96, 220),
        NoiseStratum("symmetric-contamination", "contaminated2x20", dimensions, 0.24, 0.96, 220),
        NoiseStratum("skewed", "skewed_exponential", dimensions, 0.24, 0.96, 220),
    )


def hidden_strata() -> tuple[NoiseStratum, ...]:
    dimensions = (17, 31, 63, 127, 255)
    return (
        NoiseStratum("student2.5", "student2_5", dimensions, 0.24, 0.96, 900),
        NoiseStratum("rare-extreme-contamination", "contaminated1x40", dimensions, 0.24, 0.96, 900),
        NoiseStratum("lognormal-skew", "centered_lognormal", dimensions, 0.24, 0.96, 900),
        NoiseStratum("asymmetric-mixture", "asymmetric_mixture", dimensions, 0.24, 0.96, 900),
    )


def search_policy(seed: int) -> tuple[SwitchPolicy, StrataEvaluation, dict[str, object]]:
    candidates = [policy for policy in candidate_policies(seed) if exact_training_success(policy)]
    strata = development_strata()
    evaluated = [(policy, evaluate_strata(seed + 2_000_003, policy, strata)) for policy in candidates]
    best_minimum = max(score.minimum_accuracy for _, score in evaluated)
    near = [(policy, score) for policy, score in evaluated if score.minimum_accuracy >= best_minimum - 0.003]
    best_mean = max(score.mean_accuracy for _, score in near)
    near_mean = [(policy, score) for policy, score in near if score.mean_accuracy >= best_mean - 0.002]
    selected, score = min(
        near_mean,
        key=lambda item: (
            item[1].mean_replicates_per_query,
            ROBUST_AGGREGATORS.index(item[0].robust_aggregator),
            item[0].skew_threshold,
            item[0].tail_threshold,
            item[0].right_threshold,
            item[0].trigger_margin,
        ),
    )
    return selected, score, {
        "supervision": "terminal_root_success_only",
        "step_action_labels_used": False,
        "selection_objective": "worst_seen_law_then_mean_then_samples",
        "candidate_count": len(candidates),
        "selected_policy": selected.text(),
        "development_worst_accuracy": score.minimum_accuracy,
        "development_mean_accuracy": score.mean_accuracy,
        "development_robust_rate_range": [score.minimum_robust_estimator_rate, score.maximum_robust_estimator_rate],
        "development_rows": list(score.rows),
    }


def wrong_query(policy: SwitchPolicy) -> SwitchPolicy:
    rule = next(rule for rule in position_rules() if rule.name == "first")
    return SwitchPolicy(
        rule,
        policy.robust_aggregator,
        policy.skew_threshold,
        policy.tail_threshold,
        policy.right_threshold,
        policy.difference_threshold,
        policy.trigger_margin,
        policy.base_replicates,
        policy.extra_replicates,
        policy.mapping,
    )


def random_mapping_control(seed: int, policy: SwitchPolicy, strata: tuple[NoiseStratum, ...]) -> dict[str, float]:
    rng = random.Random(seed)
    reduced = tuple(NoiseStratum(s.name, s.family, s.dimensions, s.rho_low, s.rho_high, 160) for s in strata)
    scores = []
    for index in range(10):
        mapping = tuple(rng.choice((LEFT, ACCEPT, RIGHT)) for _ in range(4))
        candidate = SwitchPolicy(
            policy.rule,
            policy.robust_aggregator,
            policy.skew_threshold,
            policy.tail_threshold,
            policy.right_threshold,
            policy.difference_threshold,
            policy.trigger_margin,
            policy.base_replicates,
            policy.extra_replicates,
            mapping,
        )
        scores.append(evaluate_strata(seed + index * 1_000_003, candidate, reduced).minimum_accuracy)
    return {"trials": len(scores), "median_worst_accuracy": float(np.median(scores)), "maximum_worst_accuracy": max(scores)}


def digest(policy: SwitchPolicy) -> str:
    return hashlib.sha256(policy.text().encode("utf-8")).hexdigest()


def serialize(policy: SwitchPolicy) -> dict[str, object]:
    return {
        "rule": policy.rule.name,
        "robust_aggregator": policy.robust_aggregator,
        "skew_threshold": policy.skew_threshold,
        "tail_threshold": policy.tail_threshold,
        "right_threshold": policy.right_threshold,
        "difference_threshold": policy.difference_threshold,
        "trigger_margin": policy.trigger_margin,
        "base_replicates": policy.base_replicates,
        "extra_replicates": policy.extra_replicates,
        "mapping": [ACTION_NAMES[action] for action in policy.mapping],
    }


def run(seed: int = 1501) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    policy, development, search = search_policy(seed * 10_000 + 103)
    frozen_digest = digest(policy)

    # Fresh noise laws and larger dimensions are instantiated only after the
    # diagnostic switch, robust estimator and measurement policy are frozen.
    strata = hidden_strata()
    candidate = evaluate_strata(seed * 10_000 + 15_000_001, policy, strata)
    mean_only = evaluate_strata(seed * 10_000 + 15_000_003, policy, strata, estimator_mode="mean")
    robust_only = evaluate_strata(seed * 10_000 + 15_000_005, policy, strata, estimator_mode="robust")
    fixed_switch_1024 = evaluate_strata(seed * 10_000 + 15_000_007, policy, strata, fixed_replicates=1024)
    query_control = evaluate_strata(seed * 10_000 + 15_000_009, wrong_query(policy), strata)
    random_control = random_mapping_control(seed * 10_000 + 15_000_011, policy, strata)

    best_single = max(mean_only.minimum_accuracy, robust_only.minimum_accuracy)
    portfolio_gap = candidate.minimum_accuracy - best_single
    fixed_gap = candidate.minimum_accuracy - fixed_switch_1024.minimum_accuracy
    query_gap = candidate.minimum_accuracy - query_control.minimum_accuracy
    random_gap = candidate.minimum_accuracy - random_control["median_worst_accuracy"]
    sample_saving = 1.0 - candidate.mean_replicates_per_query / 1024.0
    switch_range = candidate.maximum_robust_estimator_rate - candidate.minimum_robust_estimator_rate
    dimensions = strata[0].dimensions
    query_optimality = all(optimal_worst_case_queries(d) == information_lower_bound(d) for d in dimensions)
    gate = (
        bool(equivalence["exact_within_tolerance"])
        and search["step_action_labels_used"] is False
        and exact_training_success(policy)
        and policy.rule.name == "lower_midpoint"
        and development.minimum_accuracy >= 0.98
        and candidate.minimum_accuracy >= 0.97
        and candidate.maximum_invalid_transition_rate <= 0.01
        and candidate.maximum_mean_replicates_per_query <= 720.0
        and portfolio_gap >= 0.02
        and fixed_gap >= -0.01
        and query_gap >= 0.20
        and random_gap >= 0.45
        and sample_saving >= 0.25
        and switch_range >= 0.20
        and query_optimality
    )
    return {
        "status": "online_estimator_switch_candidate" if gate else "not_yet",
        "claim_scope": (
            "terminal-reward synthesis learns a sample-diagnostic switch between "
            "unbiased mean and robust aggregation, freezes it, and transfers to "
            "fresh symmetric-heavy-tail and asymmetric-skew laws while saving "
            "samples; the diagnostic grammar and causal family remain human "
            "supplied, so this is not a world breakthrough"
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
        "mean_only_control": mean_only.__dict__,
        "robust_only_control": robust_only.__dict__,
        "fixed_switch_1024_control": fixed_switch_1024.__dict__,
        "wrong_query_control": query_control.__dict__,
        "random_mapping_control": random_control,
        "portfolio_gap": portfolio_gap,
        "fixed_switch_gap": fixed_gap,
        "wrong_query_gap": query_gap,
        "random_gap": random_gap,
        "sample_saving_vs_1024": sample_saving,
        "hidden_switch_rate_range": switch_range,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "policy": report["selected_policy"],
        "hidden_worst": report["candidate"]["minimum_accuracy"],
        "portfolio_gap": report["portfolio_gap"],
        "switch_range": report["hidden_switch_rate_range"],
        "sample_saving": report["sample_saving_vs_1024"],
    }, indent=2))


if __name__ == "__main__":
    main()
