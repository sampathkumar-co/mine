from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from . import exact_tail_v36 as v36
from . import robust_regret_v35 as v35
from . import state_policy_v34 as v34


THRESHOLDS = (4, 6, 8, 10, 12, 16)
FEATURE_LIMITS = (4, 6, 8, 10, 12)


@dataclass(frozen=True)
class SafePolicy:
    candidate_threshold: int
    feature_limit: int
    fallback: str

    def text(self) -> str:
        return (
            f"safe_exact_if(candidates<={self.candidate_threshold},"
            f"features<={self.feature_limit}):else->{self.fallback}"
        )


@dataclass(frozen=True)
class SafeEvaluation:
    diagnosed_fraction: float
    mean_queries: float
    worst_queries: int
    unresolved: int
    candidates: int
    exact_query_uses: int


class FallbackPlanner:
    def __init__(self, task: object, objective: str) -> None:
        self.task = task
        self.objective = objective
        self.cache: dict[tuple[int, int], v36.Plan] = {}

    def solve(self, allowed: int, remaining: int) -> v36.Plan:
        key = (allowed, remaining)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        size = allowed.bit_count()
        if v34.base.pure_label(self.task, allowed) is not None:
            plan = v36.Plan(size, 0, 0, None)
            self.cache[key] = plan
            return plan
        try:
            query = v34.base.select_query(
                self.task,
                allowed,
                remaining,
                v34.OBJECTIVES[self.objective],
            )
        except RuntimeError:
            plan = v36.Plan(0, 0, 0, None)
            self.cache[key] = plan
            return plan
        next_remaining = remaining & ~(1 << query)
        children = [
            allowed & mask
            for mask in self.task.masks_for(query).values()
            if allowed & mask
        ]
        child_plans = [
            self.solve(child, next_remaining)
            for child in children
        ]
        plan = v36.Plan(
            diagnosed=sum(row.diagnosed for row in child_plans),
            worst_queries=1 + max(
                row.worst_queries for row in child_plans
            ),
            total_queries=size + sum(
                row.total_queries for row in child_plans
            ),
            query=query,
        )
        self.cache[key] = plan
        return plan


def strictly_dominates(
    candidate: v36.Plan,
    baseline: v36.Plan,
) -> bool:
    return candidate.score() > baseline.score()


def select_query(
    task: object,
    allowed: int,
    remaining: int,
    policy: SafePolicy,
    exact: v36.ExactPlanner,
    fallback: FallbackPlanner,
) -> tuple[int, bool]:
    eligible = (
        allowed.bit_count() <= policy.candidate_threshold
        and remaining.bit_count() <= policy.feature_limit
    )
    if eligible:
        exact_plan = exact.solve(allowed, remaining)
        fallback_plan = fallback.solve(allowed, remaining)
        if (
            exact_plan.query is not None
            and strictly_dominates(exact_plan, fallback_plan)
        ):
            return exact_plan.query, True
    fallback_plan = fallback.solve(allowed, remaining)
    if fallback_plan.query is None:
        raise RuntimeError("fallback cannot separate state")
    return fallback_plan.query, False


def analyse_task(
    task: object,
    policy: SafePolicy,
    exact: v36.ExactPlanner,
    fallback: FallbackPlanner,
) -> tuple[tuple[bool, int, int], ...]:
    results: list[tuple[bool, int, int] | None] = [
        None
    ] * task.candidate_count

    def assign(
        allowed: int,
        prediction: str | None,
        queries: int,
        exact_uses: int,
    ) -> None:
        mask = allowed
        while mask:
            bit = mask & -mask
            candidate = bit.bit_length() - 1
            results[candidate] = (
                prediction == task.labels[candidate],
                queries,
                exact_uses,
            )
            mask ^= bit

    def visit(
        allowed: int,
        remaining: int,
        queries: int,
        exact_uses: int,
    ) -> None:
        prediction = v34.base.pure_label(task, allowed)
        if prediction is not None:
            assign(allowed, prediction, queries, exact_uses)
            return
        try:
            query, use_exact = select_query(
                task,
                allowed,
                remaining,
                policy,
                exact,
                fallback,
            )
        except RuntimeError:
            assign(allowed, None, queries, exact_uses)
            return
        next_remaining = remaining & ~(1 << query)
        covered = 0
        for mask in task.masks_for(query).values():
            child = allowed & mask
            if not child:
                continue
            covered |= child
            visit(
                child,
                next_remaining,
                queries + 1,
                exact_uses + int(use_exact),
            )
        if covered != allowed:
            raise AssertionError("query outcomes did not partition state")

    visit(
        task.full_mask,
        (1 << task.query_count) - 1,
        0,
        0,
    )
    if any(row is None for row in results):
        raise AssertionError("missing candidate result")
    return tuple(row for row in results if row is not None)


def evaluate(
    task: object,
    policy: SafePolicy,
    exact: v36.ExactPlanner,
    fallback: FallbackPlanner,
) -> SafeEvaluation:
    rows = analyse_task(task, policy, exact, fallback)
    diagnosed = sum(int(row[0]) for row in rows)
    queries = [row[1] for row in rows]
    return SafeEvaluation(
        diagnosed_fraction=diagnosed / task.candidate_count,
        mean_queries=float(np.mean(queries)),
        worst_queries=max(queries),
        unresolved=task.candidate_count - diagnosed,
        candidates=task.candidate_count,
        exact_query_uses=sum(row[2] for row in rows),
    )


def regret_row(
    candidate: SafeEvaluation,
    baseline: dict[str, float | int],
) -> dict[str, float | int]:
    diagnosed_regret = (
        float(baseline["best_diagnosed"])
        - candidate.diagnosed_fraction
    )
    worst_regret = (
        candidate.worst_queries
        - int(baseline["best_worst"])
    )
    mean_regret = (
        candidate.mean_queries
        - float(baseline["best_mean"])
    )
    return {
        "diagnosed_regret": diagnosed_regret,
        "worst_query_regret": worst_regret,
        "mean_query_regret": mean_regret,
        "strict_win": int(
            diagnosed_regret < -1e-12
            or (
                diagnosed_regret <= 1e-12
                and (worst_regret < 0 or mean_regret < -1e-12)
            )
        ),
        "no_harm": int(
            diagnosed_regret <= 0.005
            and worst_regret <= 0
            and mean_regret <= 1e-12
        ),
    }


def portfolio_score(
    regret: dict[str, float | int],
    candidate: SafeEvaluation,
    fallback: str,
) -> tuple[float | int | str, ...]:
    return (
        -float(regret["diagnosed_regret"]),
        -max(0, int(regret["worst_query_regret"])),
        -max(0.0, float(regret["mean_query_regret"])),
        -int(regret["worst_query_regret"]),
        -float(regret["mean_query_regret"]),
        int(candidate.exact_query_uses > 0),
        fallback,
    )


def evaluate_portfolio(
    task: object,
    threshold: int,
    feature_limit: int,
    exact: v36.ExactPlanner,
    fallback_planners: dict[str, FallbackPlanner],
    baseline: dict[str, float | int],
) -> dict[str, object]:
    rows = []
    for fallback in v34.OBJECTIVE_NAMES:
        policy = SafePolicy(threshold, feature_limit, fallback)
        candidate = evaluate(
            task,
            policy,
            exact,
            fallback_planners[fallback],
        )
        regret = regret_row(candidate, baseline)
        rows.append(
            (
                portfolio_score(regret, candidate, fallback),
                fallback,
                candidate,
                regret,
            )
        )
    _, fallback, candidate, regret = max(
        rows,
        key=lambda row: row[0],
    )
    return {
        "selected_fallback": fallback,
        "candidate": candidate.__dict__,
        **regret,
    }


def profile_configuration(
    tasks: list[object],
    threshold: int,
    feature_limit: int,
    exact_planners: dict[str, v36.ExactPlanner],
    fallback_planners: dict[str, dict[str, FallbackPlanner]],
    baselines: dict[str, dict[str, float | int]],
) -> tuple[dict[str, object], dict[str, object]]:
    details = {
        task.name: evaluate_portfolio(
            task,
            threshold,
            feature_limit,
            exact_planners[task.name],
            fallback_planners[task.name],
            baselines[task.name],
        )
        for task in tasks
    }
    diagnosed = [
        float(row["diagnosed_regret"])
        for row in details.values()
    ]
    worst = [
        int(row["worst_query_regret"])
        for row in details.values()
    ]
    mean = [
        float(row["mean_query_regret"])
        for row in details.values()
    ]
    profile = {
        "diagnosed_regret_max": max(diagnosed),
        "worst_query_regret_max": max(worst),
        "mean_query_regret_max": max(mean),
        "mean_query_regret_median": float(np.median(mean)),
        "strict_wins": sum(
            int(row["strict_win"])
            for row in details.values()
        ),
        "no_harm_tasks": sum(
            int(row["no_harm"])
            for row in details.values()
        ),
        "exact_query_uses": sum(
            int(row["candidate"]["exact_query_uses"])
            for row in details.values()
        ),
    }
    return profile, details


def configuration_score(
    profile: dict[str, object],
    threshold: int,
    feature_limit: int,
) -> tuple[float | int, ...]:
    return (
        -float(profile["diagnosed_regret_max"]),
        -int(profile["worst_query_regret_max"]),
        -float(profile["mean_query_regret_max"]),
        -float(profile["mean_query_regret_median"]),
        int(profile["no_harm_tasks"]),
        int(profile["strict_wins"]),
        int(profile["exact_query_uses"] > 0),
        -threshold,
        -feature_limit,
    )


def run() -> dict[str, object]:
    tasks = v35.opened_domain_pool()
    baselines = v35.control_baselines(tasks)
    exact_planners = {
        task.name: v36.ExactPlanner(task)
        for task in tasks
    }
    fallback_planners = {
        task.name: {
            objective: FallbackPlanner(task, objective)
            for objective in v34.OBJECTIVE_NAMES
        }
        for task in tasks
    }
    rows = []
    for threshold, feature_limit in itertools.product(
        THRESHOLDS,
        FEATURE_LIMITS,
    ):
        profile, details = profile_configuration(
            tasks,
            threshold,
            feature_limit,
            exact_planners,
            fallback_planners,
            baselines,
        )
        rows.append(
            (
                configuration_score(
                    profile,
                    threshold,
                    feature_limit,
                ),
                threshold,
                feature_limit,
                profile,
                details,
            )
        )
    _, threshold, feature_limit, profile, details = max(
        rows,
        key=lambda row: row[0],
    )
    gate = (
        float(profile["diagnosed_regret_max"]) <= 0.005
        and int(profile["worst_query_regret_max"]) <= 0
        and float(profile["mean_query_regret_max"]) <= 1e-12
        and int(profile["no_harm_tasks"]) == len(tasks)
        and int(profile["strict_wins"]) >= 3
        and int(profile["exact_query_uses"]) > 0
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "threshold": threshold,
                "feature_limit": feature_limit,
                "portfolio": "evaluate_all_safe_repairs_v1",
                "domains": [task.name for task in tasks],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": "safe_portfolio_ready" if gate else "not_yet",
        "claim_scope": (
            "all constant specialists are configured per task, and exact "
            "subtrees replace a specialist subtree only under a machine-checked "
            "lexicographic dominance certificate; this is development evidence"
        ),
        "development_gate": gate,
        "domain_count": len(tasks),
        "configuration_count": len(THRESHOLDS) * len(FEATURE_LIMITS),
        "selected_configuration": {
            "candidate_threshold": threshold,
            "feature_limit": feature_limit,
        },
        "profile": profile,
        "task_regrets": details,
        "frozen_portfolio_digest": digest,
        "planner_cache_states": {
            task.name: {
                "exact": len(exact_planners[task.name].cache),
                "fallback": {
                    objective: len(planner.cache)
                    for objective, planner in fallback_planners[
                        task.name
                    ].items()
                },
            }
            for task in tasks
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "selected_configuration": report[
            "selected_configuration"
        ],
        "profile": report["profile"],
    }, indent=2))


if __name__ == "__main__":
    main()
