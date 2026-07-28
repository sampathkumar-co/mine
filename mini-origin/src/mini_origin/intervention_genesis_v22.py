from __future__ import annotations

from dataclasses import dataclass
import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class PositionRule:
    name: str
    complexity: int
    offset: Callable[[int], int]


@dataclass(frozen=True)
class ThresholdFit:
    threshold: float
    balanced_accuracy: float
    true_positive_rate: float
    true_negative_rate: float
    active_count: int
    inactive_count: int


@dataclass(frozen=True)
class PolicyEvaluation:
    accuracy: float
    mean_queries: float
    maximum_queries: int
    mean_remaining_candidates: float


@dataclass(frozen=True)
class QueryTrace:
    predicted_root: int
    correct: bool
    queries: int
    remaining_candidates: int


def ar1_covariance(dimension: int, rho: float) -> np.ndarray:
    indices = np.arange(dimension)
    return rho ** np.abs(indices[:, None] - indices[None, :])


def root_chain_structural_covariance(
    dimension: int,
    root: int,
    rho: float,
) -> np.ndarray:
    """Covariance of a chain whose arrows point away from ``root``."""
    coefficient = np.zeros((dimension, dimension), dtype=np.float64)
    for index in range(root - 1, -1, -1):
        coefficient[index, index + 1] = rho
    for index in range(root + 1, dimension):
        coefficient[index, index - 1] = rho

    noise = np.full(dimension, 1.0 - rho * rho, dtype=np.float64)
    noise[root] = 1.0
    inverse = np.linalg.inv(np.eye(dimension) - coefficient)
    return inverse @ np.diag(noise) @ inverse.T


def observational_equivalence_certificate(
    dimensions: Iterable[int] = (3, 4, 5, 6, 7),
    rhos: Iterable[float] = (0.25, 0.43, 0.67, 0.88),
) -> dict[str, object]:
    maximum_difference = 0.0
    checked = 0
    rows = []
    for dimension in dimensions:
        for rho in rhos:
            target = ar1_covariance(dimension, rho)
            root_errors = []
            for root in range(dimension):
                covariance = root_chain_structural_covariance(
                    dimension, root, rho
                )
                error = float(np.max(np.abs(covariance - target)))
                maximum_difference = max(maximum_difference, error)
                root_errors.append(error)
                checked += 1
            rows.append(
                {
                    "dimension": dimension,
                    "rho": rho,
                    "maximum_root_error": max(root_errors),
                }
            )
    return {
        "checked_models": checked,
        "maximum_covariance_difference": maximum_difference,
        "exact_within_tolerance": maximum_difference <= 1e-10,
        "rows": rows,
    }


@functools.lru_cache(maxsize=None)
def optimal_worst_case_queries(candidate_count: int) -> int:
    if candidate_count <= 1:
        return 0
    return min(
        1
        + max(
            optimal_worst_case_queries(offset),
            optimal_worst_case_queries(candidate_count - offset - 1),
        )
        for offset in range(candidate_count)
    )


def optimal_query_offset(candidate_count: int) -> int:
    if candidate_count <= 1:
        return 0
    candidates = []
    for offset in range(candidate_count):
        left = offset
        right = candidate_count - offset - 1
        depth = 1 + max(
            optimal_worst_case_queries(left),
            optimal_worst_case_queries(right),
        )
        candidates.append((depth, abs(left - right), offset))
    return min(candidates)[2]


def information_lower_bound(candidate_count: int) -> int:
    if candidate_count <= 1:
        return 0
    # A depth-q comparison/intervention tree can identify at most
    # 2^(q+1)-1 ordered root locations: one equality leaf at the query and
    # recursively one interval on either side.
    return max(0, math.ceil(math.log2(candidate_count + 1)) - 1)


def position_rules() -> list[PositionRule]:
    return [
        PositionRule("first", 1, lambda size: 0),
        PositionRule("last", 1, lambda size: max(0, size - 1)),
        PositionRule("lower_midpoint", 2, lambda size: max(0, (size - 1) // 2)),
        PositionRule("upper_midpoint", 2, lambda size: max(0, size // 2)),
        PositionRule("left_quarter", 3, lambda size: max(0, (size - 1) // 4)),
        PositionRule(
            "right_quarter",
            3,
            lambda size: max(0, (3 * (size - 1)) // 4),
        ),
    ]


def synthesize_position_rule(
    training_sizes: Iterable[int] = range(2, 8),
) -> tuple[PositionRule, dict[str, object]]:
    sizes = tuple(training_sizes)
    exhaustive_offsets = {
        size: optimal_query_offset(size) for size in sizes
    }
    candidates = [
        rule
        for rule in position_rules()
        if all(rule.offset(size) == exhaustive_offsets[size] for size in sizes)
    ]
    if not candidates:
        raise RuntimeError("no size-polymorphic rule explains the optimal trees")
    candidates.sort(key=lambda rule: (rule.complexity, rule.name))
    selected = candidates[0]
    optimality = {
        size: {
            "exhaustive_offset": exhaustive_offsets[size],
            "selected_offset": selected.offset(size),
            "dynamic_program_depth": optimal_worst_case_queries(size),
            "information_lower_bound": information_lower_bound(size),
        }
        for size in sizes
    }
    return selected, {
        "training_sizes": list(sizes),
        "selected_rule": selected.name,
        "selected_complexity": selected.complexity,
        "matching_rules": [rule.name for rule in candidates],
        "optimality": optimality,
        "all_training_depths_meet_lower_bound": all(
            row["dynamic_program_depth"] == row["information_lower_bound"]
            for row in optimality.values()
        ),
    }


def _neighbor_sample_mean(
    rng: np.random.Generator,
    rho: float,
    active: bool,
    replicates: int,
) -> float:
    mean = rho if active else 0.0
    variance = 1.0 - rho * rho if active else 1.0
    return float(rng.normal(mean, math.sqrt(variance / replicates)))


def intervention_response(
    seed: int,
    dimension: int,
    root: int,
    rho: float,
    query: int,
    replicates: int,
) -> tuple[float | None, float | None]:
    """Sample immediate-neighbour responses after do(X_query=1)."""
    rng = np.random.default_rng(seed)
    left = None
    right = None
    if query > 0:
        # The left neighbour is a child when root is at or to the right.
        left = _neighbor_sample_mean(
            rng, rho, active=root >= query, replicates=replicates
        )
    if query < dimension - 1:
        # The right neighbour is a child when root is at or to the left.
        right = _neighbor_sample_mean(
            rng, rho, active=root <= query, replicates=replicates
        )
    return left, right


def fit_activity_threshold(
    seed: int,
    samples: int = 1_400,
    replicates: int = 192,
) -> ThresholdFit:
    rng = np.random.default_rng(seed)
    active: list[float] = []
    inactive: list[float] = []
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
        if left is not None:
            (active if root >= query else inactive).append(left)
        if right is not None:
            (active if root <= query else inactive).append(right)

    values = sorted(set(active + inactive))
    thresholds = [values[0] - 1e-9]
    thresholds.extend(
        0.5 * (left + right) for left, right in zip(values, values[1:])
    )
    thresholds.append(values[-1] + 1e-9)

    best: tuple[float, float, float, float] | None = None
    for threshold in thresholds:
        true_positive = sum(value > threshold for value in active) / len(active)
        true_negative = sum(value <= threshold for value in inactive) / len(inactive)
        balanced = 0.5 * (true_positive + true_negative)
        candidate = (balanced, -abs(threshold - 0.20), threshold, true_positive)
        if best is None or candidate > best:
            best = candidate
            best_true_negative = true_negative
    assert best is not None
    balanced, _, threshold, true_positive = best
    return ThresholdFit(
        threshold=float(threshold),
        balanced_accuracy=float(balanced),
        true_positive_rate=float(true_positive),
        true_negative_rate=float(best_true_negative),
        active_count=len(active),
        inactive_count=len(inactive),
    )


def decode_relation(
    left: float | None,
    right: float | None,
    query: int,
    dimension: int,
    threshold: float,
) -> str:
    left_active = left is not None and left > threshold
    right_active = right is not None and right > threshold

    if query == 0:
        return "equal" if right_active else "right"
    if query == dimension - 1:
        return "equal" if left_active else "left"
    if left_active and right_active:
        return "equal"
    if left_active:
        return "right"
    if right_active:
        return "left"

    # A noisy double miss is not allowed to create a fourth outcome. Use the
    # larger response as the least-committal ordered comparison.
    assert left is not None and right is not None
    return "right" if left > right else "left"


def run_policy(
    seed: int,
    dimension: int,
    root: int,
    rho: float,
    threshold: float,
    replicates: int,
    query_budget: int,
    mode: str,
    rule: PositionRule,
) -> QueryTrace:
    rng = np.random.default_rng(seed + 991)
    low = 0
    high = dimension - 1
    queries = 0
    while low < high and queries < query_budget:
        size = high - low + 1
        if mode == "synthesized":
            query = low + rule.offset(size)
        elif mode == "random":
            query = int(rng.integers(low, high + 1))
        elif mode == "left_scan":
            query = low
        elif mode == "right_scan":
            query = high
        else:
            raise ValueError(mode)

        left, right = intervention_response(
            seed + queries * 7_919,
            dimension,
            root,
            rho,
            query,
            replicates,
        )
        relation = decode_relation(
            left, right, query, dimension, threshold
        )
        queries += 1
        if relation == "equal":
            return QueryTrace(query, query == root, queries, 1)
        if relation == "left":
            high = query - 1
        else:
            low = query + 1
        if low > high:
            # Contradictory noisy evidence: return the closest legal location.
            prediction = min(max(query, 0), dimension - 1)
            return QueryTrace(prediction, prediction == root, queries, 0)

    remaining = max(1, high - low + 1)
    prediction = low + (remaining - 1) // 2
    return QueryTrace(
        prediction,
        prediction == root,
        queries,
        remaining,
    )


def evaluate_policy(
    seed: int,
    rule: PositionRule,
    threshold: float,
    mode: str,
    trials: int = 2_400,
    replicates: int = 192,
    dimensions: tuple[int, ...] = (8, 16, 32, 64),
) -> PolicyEvaluation:
    rng = np.random.default_rng(seed)
    traces: list[QueryTrace] = []
    for index in range(trials):
        dimension = int(rng.choice(dimensions))
        root = int(rng.integers(0, dimension))
        rho = float(rng.uniform(0.30, 0.90))
        budget = information_lower_bound(dimension)
        traces.append(
            run_policy(
                seed + index * 104_729,
                dimension,
                root,
                rho,
                threshold,
                replicates,
                budget,
                mode,
                rule,
            )
        )
    return PolicyEvaluation(
        accuracy=float(np.mean([trace.correct for trace in traces])),
        mean_queries=float(np.mean([trace.queries for trace in traces])),
        maximum_queries=max(trace.queries for trace in traces),
        mean_remaining_candidates=float(
            np.mean([trace.remaining_candidates for trace in traces])
        ),
    )


def passive_guess_control(
    seed: int,
    trials: int = 2_400,
    dimensions: tuple[int, ...] = (8, 16, 32, 64),
) -> float:
    rng = np.random.default_rng(seed)
    correct = []
    for _ in range(trials):
        dimension = int(rng.choice(dimensions))
        root = int(rng.integers(0, dimension))
        # Since every root induces exactly the same observational Gaussian,
        # no passive rule has information. This is a deterministic midpoint
        # representative of the chance-level equivalence class.
        prediction = (dimension - 1) // 2
        correct.append(prediction == root)
    return float(np.mean(correct))


def policy_digest(rule: PositionRule, threshold: float) -> str:
    return hashlib.sha256(
        f"{rule.name}:{threshold:.17g}:do-one:read-neighbours".encode("utf-8")
    ).hexdigest()


def run(seed: int = 701) -> dict[str, object]:
    equivalence = observational_equivalence_certificate()
    rule, synthesis = synthesize_position_rule()
    threshold = fit_activity_threshold(seed * 10_000 + 37)
    frozen_digest = policy_digest(rule, threshold.threshold)

    # Hidden dimensions, roots, structural strengths and response noise are
    # generated only after the position rule and activity threshold are frozen.
    candidate = evaluate_policy(
        seed * 10_000 + 7_000_001,
        rule,
        threshold.threshold,
        "synthesized",
    )
    random_control = evaluate_policy(
        seed * 10_000 + 7_000_003,
        rule,
        threshold.threshold,
        "random",
    )
    left_control = evaluate_policy(
        seed * 10_000 + 7_000_005,
        rule,
        threshold.threshold,
        "left_scan",
    )
    right_control = evaluate_policy(
        seed * 10_000 + 7_000_007,
        rule,
        threshold.threshold,
        "right_scan",
    )
    passive = passive_guess_control(seed * 10_000 + 7_000_009)

    hidden_dimensions = (8, 16, 32, 64)
    optimality = {
        dimension: {
            "candidate_query_budget": information_lower_bound(dimension),
            "dynamic_program_optimum": optimal_worst_case_queries(dimension),
            "information_lower_bound": information_lower_bound(dimension),
        }
        for dimension in hidden_dimensions
    }
    exact_query_optimality = all(
        row["candidate_query_budget"]
        == row["dynamic_program_optimum"]
        == row["information_lower_bound"]
        for row in optimality.values()
    )

    strongest_scan = max(left_control.accuracy, right_control.accuracy)
    candidate_gate = (
        bool(equivalence["exact_within_tolerance"])
        and synthesis["all_training_depths_meet_lower_bound"]
        and rule.name == "lower_midpoint"
        and threshold.balanced_accuracy >= 0.99
        and exact_query_optimality
        and candidate.accuracy >= 0.975
        and candidate.maximum_queries
        <= max(information_lower_bound(value) for value in hidden_dimensions)
        and candidate.accuracy >= random_control.accuracy + 0.18
        and candidate.accuracy >= strongest_scan + 0.45
        and candidate.accuracy >= passive + 0.85
    )

    return {
        "status": (
            "proof_carrying_adaptive_intervention_candidate"
            if candidate_gate
            else "not_yet"
        ),
        "claim_scope": (
            "the system certifies observational equivalence of a Gaussian causal-chain class, "
            "exhaustively synthesizes optimal small decision trees, induces and freezes a "
            "size-polymorphic midpoint intervention program, and identifies unseen roots at the "
            "information-theoretic query bound; this rediscovers binary search within active causal "
            "identification and is not a world breakthrough"
        ),
        "seed": seed,
        "candidate_gate": candidate_gate,
        "observational_equivalence": equivalence,
        "synthesis": synthesis,
        "activity_threshold": threshold.__dict__,
        "frozen_policy_digest": frozen_digest,
        "hidden_dimensions": list(hidden_dimensions),
        "hidden_optimality": optimality,
        "exact_hidden_query_optimality": exact_query_optimality,
        "candidate": candidate.__dict__,
        "random_equal_budget_control": random_control.__dict__,
        "left_scan_equal_budget_control": left_control.__dict__,
        "right_scan_equal_budget_control": right_control.__dict__,
        "passive_observation_control_accuracy": passive,
        "candidate_random_gap": candidate.accuracy - random_control.accuracy,
        "candidate_scan_gap": candidate.accuracy - strongest_scan,
        "candidate_passive_gap": candidate.accuracy - passive,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "rule": report["synthesis"]["selected_rule"],
                "threshold_accuracy": report["activity_threshold"][
                    "balanced_accuracy"
                ],
                "candidate_accuracy": report["candidate"]["accuracy"],
                "random_accuracy": report["random_equal_budget_control"][
                    "accuracy"
                ],
                "scan_gap": report["candidate_scan_gap"],
                "passive_accuracy": report[
                    "passive_observation_control_accuracy"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
