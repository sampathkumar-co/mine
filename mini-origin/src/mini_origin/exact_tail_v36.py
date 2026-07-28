from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from . import robust_regret_v35 as v35
from . import state_policy_v34 as v34


THRESHOLDS = (4, 6, 8, 10, 12, 16, 24, 32)
FEATURE_LIMITS = (4, 6, 8, 12, 16, 32)


@dataclass(frozen=True)
class TailPolicy:
    candidate_threshold: int
    feature_limit: int
    fallback: str

    def text(self) -> str:
        return (
            f"exact_if(candidates<={self.candidate_threshold},"
            f"features<={self.feature_limit}):else->{self.fallback}"
        )


@dataclass(frozen=True)
class Plan:
    diagnosed: int
    worst_queries: int
    total_queries: int
    query: int | None

    def score(self) -> tuple[int, int, int, int]:
        return (
            self.diagnosed,
            -self.worst_queries,
            -self.total_queries,
            -(self.query if self.query is not None else 10**9),
        )


@dataclass(frozen=True)
class TailEvaluation:
    diagnosed_fraction: float
    mean_queries: float
    worst_queries: int
    unresolved: int
    candidates: int
    exact_query_uses: int


class ExactPlanner:
    def __init__(self, task: object) -> None:
        self.task = task
        self.cache: dict[tuple[int, int], Plan] = {}

    def solve(self, allowed: int, remaining: int) -> Plan:
        key = (allowed, remaining)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        size = allowed.bit_count()
        if v34.base.pure_label(self.task, allowed) is not None:
            plan = Plan(size, 0, 0, None)
            self.cache[key] = plan
            return plan
        candidates: list[Plan] = []
        query_bits = remaining
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            children = []
            for mask in self.task.masks_for(query).values():
                child = allowed & mask
                if child:
                    children.append(child)
            if len(children) <= 1:
                continue
            next_remaining = remaining & ~(1 << query)
            child_plans = [
                self.solve(child, next_remaining)
                for child in children
            ]
            candidates.append(
                Plan(
                    diagnosed=sum(row.diagnosed for row in child_plans),
                    worst_queries=1 + max(
                        row.worst_queries for row in child_plans
                    ),
                    total_queries=size + sum(
                        row.total_queries for row in child_plans
                    ),
                    query=query,
                )
            )
        if candidates:
            plan = max(candidates, key=lambda row: row.score())
        else:
            plan = Plan(0, 0, 0, None)
        self.cache[key] = plan
        return plan


def grammar() -> tuple[TailPolicy, ...]:
    return tuple(
        TailPolicy(threshold, feature_limit, fallback)
        for threshold, feature_limit, fallback in itertools.product(
            THRESHOLDS,
            FEATURE_LIMITS,
            v34.OBJECTIVE_NAMES,
        )
    )


def select_query(
    task: object,
    allowed: int,
    remaining: int,
    policy: TailPolicy,
    planner: ExactPlanner,
) -> tuple[int, bool]:
    exact = (
        allowed.bit_count() <= policy.candidate_threshold
        and remaining.bit_count() <= policy.feature_limit
    )
    if exact:
        plan = planner.solve(allowed, remaining)
        if plan.query is not None:
            return plan.query, True
    query = v34.base.select_query(
        task,
        allowed,
        remaining,
        v34.OBJECTIVES[policy.fallback],
    )
    return query, False


def analyse_task(
    task: object,
    policy: TailPolicy,
    planner: ExactPlanner,
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
            query, exact = select_query(
                task,
                allowed,
                remaining,
                policy,
                planner,
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
                exact_uses + int(exact),
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
    policy: TailPolicy,
    planner: ExactPlanner,
) -> TailEvaluation:
    rows = analyse_task(task, policy, planner)
    diagnosed = sum(int(row[0]) for row in rows)
    queries = [row[1] for row in rows]
    return TailEvaluation(
        diagnosed_fraction=diagnosed / task.candidate_count,
        mean_queries=float(np.mean(queries)),
        worst_queries=max(queries),
        unresolved=task.candidate_count - diagnosed,
        candidates=task.candidate_count,
        exact_query_uses=sum(row[2] for row in rows),
    )


def profile_policy(
    tasks: list[object],
    policy: TailPolicy,
    planners: dict[str, ExactPlanner],
    baselines: dict[str, dict[str, float | int]],
) -> tuple[dict[str, object], dict[str, object]]:
    details = {}
    diagnosed_regrets = []
    worst_regrets = []
    mean_regrets = []
    strict_wins = 0
    no_harm = 0
    exact_uses = 0
    for task in tasks:
        candidate = evaluate(task, policy, planners[task.name])
        baseline = baselines[task.name]
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
        win = int(
            diagnosed_regret < -1e-12
            or (
                diagnosed_regret <= 1e-12
                and (worst_regret < 0 or mean_regret < -1e-12)
            )
        )
        safe = int(
            diagnosed_regret <= 0.005
            and worst_regret <= 1
            and mean_regret <= 0.10
        )
        diagnosed_regrets.append(diagnosed_regret)
        worst_regrets.append(worst_regret)
        mean_regrets.append(mean_regret)
        strict_wins += win
        no_harm += safe
        exact_uses += candidate.exact_query_uses
        details[task.name] = {
            "candidate": candidate.__dict__,
            "diagnosed_regret": diagnosed_regret,
            "worst_query_regret": worst_regret,
            "mean_query_regret": mean_regret,
            "strict_win": win,
            "no_harm": safe,
        }
    profile = {
        "diagnosed_regret_max": float(max(diagnosed_regrets)),
        "worst_query_regret_max": int(max(worst_regrets)),
        "mean_query_regret_max": float(max(mean_regrets)),
        "mean_query_regret_median": float(np.median(mean_regrets)),
        "strict_wins": strict_wins,
        "no_harm_tasks": no_harm,
        "exact_query_uses": exact_uses,
    }
    return profile, details


def score(
    profile: dict[str, object],
    policy: TailPolicy,
) -> tuple[float | int | str, ...]:
    return (
        -float(profile["diagnosed_regret_max"]),
        -int(profile["worst_query_regret_max"]),
        -float(profile["mean_query_regret_max"]),
        -float(profile["mean_query_regret_median"]),
        int(profile["no_harm_tasks"]),
        int(profile["strict_wins"]),
        int(profile["exact_query_uses"] > 0),
        -policy.candidate_threshold,
        -policy.feature_limit,
        policy.text(),
    )


def run() -> dict[str, object]:
    tasks = v35.opened_domain_pool()
    planners = {
        task.name: ExactPlanner(task)
        for task in tasks
    }
    baselines = v35.control_baselines(tasks)
    rows = []
    for policy in grammar():
        profile, details = profile_policy(
            tasks,
            policy,
            planners,
            baselines,
        )
        rows.append((score(profile, policy), policy, profile, details))
    _, selected, profile, details = max(rows, key=lambda row: row[0])
    gate = (
        float(profile["diagnosed_regret_max"]) <= 0.005
        and int(profile["worst_query_regret_max"]) <= 0
        and float(profile["mean_query_regret_max"]) <= 0.10
        and float(profile["mean_query_regret_median"]) <= -0.05
        and int(profile["strict_wins"]) >= 3
        and int(profile["no_harm_tasks"]) == len(tasks)
        and int(profile["exact_query_uses"]) > 0
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "policy": selected.text(),
                "domains": [task.name for task in tasks],
                "planner": "exact_diagnosis_worst_mean_v1",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": "exact_tail_selector_ready" if gate else "not_yet",
        "claim_scope": (
            "an exact dynamic-programming tail solver is composed with a "
            "frozen heuristic fallback and selected entirely on previously "
            "opened domains; no fresh external dataset is evaluated here"
        ),
        "development_gate": gate,
        "domain_count": len(tasks),
        "policy_count": len(grammar()),
        "selected_policy": selected.text(),
        "profile": profile,
        "task_regrets": details,
        "frozen_selector_digest": digest,
        "planner_cache_states": {
            name: len(planner.cache)
            for name, planner in planners.items()
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
        "selected_policy": report["selected_policy"],
        "profile": report["profile"],
    }, indent=2))


if __name__ == "__main__":
    main()
