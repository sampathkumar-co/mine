from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from . import average_odt_frontier_v44 as frontier
from . import exact_quotient_certificate_v42 as corpus
from . import response_cost_pareto_v56 as parent
from . import state_policy_v34 as state

BUDGET = 500_000
BUDGET_LADDER = (10_000, 50_000, 250_000, 500_000)
PROFILE_SEEDS = (5801, 5802, 5803)
RANDOM_THEOREM_TASKS = 48


@dataclass(frozen=True)
class AvailabilityProfile:
    response_cost: parent.ResponseCostProfile
    successor_mask_by_query_hypothesis: tuple[tuple[int, ...], ...]
    seed: int


def stable_value(seed: int, token: str, low: int, high: int) -> int:
    raw = hashlib.sha256(f"{seed}:{token}".encode()).digest()
    return low + int.from_bytes(raw[:8], "big") % (high - low + 1)


def full_partition_families(task: object) -> dict[tuple[int, ...], tuple[int, ...]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for query in range(task.query_count):
        groups.setdefault(parent.partition(task, task.full_mask, query), []).append(query)
    return {signature: tuple(queries) for signature, queries in groups.items()}


def availability_profile_for_task(task: object, seed: int) -> AvailabilityProfile:
    response_cost = parent.profile_for_task(task, seed)
    all_queries = (1 << task.query_count) - 1
    families = full_partition_families(task)
    family_for_query = {
        query: signature for signature, queries in families.items() for query in queries
    }
    rows: list[tuple[int, ...]] = []
    for query in range(task.query_count):
        family = families[family_for_query[query]]
        by_child: dict[int, int] = {}
        for child in parent.partition(task, task.full_mask, query):
            token = f"{task.name}:{family_for_query[query]}:{child}"
            mask = all_queries
            # Globally equivalent tests are jointly consumed. Additional tests
            # are deterministically disabled by the observed response.
            for member in family:
                mask &= ~(1 << member)
            for other in range(task.query_count):
                if other in family:
                    continue
                if stable_value(seed, f"{token}:{other}", 0, 99) < 12:
                    mask &= ~(1 << other)
            by_child[child] = mask
        row = []
        for hypothesis in range(task.candidate_count):
            bit = 1 << hypothesis
            child = next(
                child for child in by_child
                if child & bit
            )
            row.append(by_child[child])
        rows.append(tuple(row))
    return AvailabilityProfile(response_cost, tuple(rows), seed)


def successor_mask(
    profile: AvailabilityProfile,
    remaining: int,
    query: int,
    child: int,
) -> int:
    index = (child & -child).bit_length() - 1
    return (
        remaining
        & ~(1 << query)
        & profile.successor_mask_by_query_hypothesis[query][index]
    )


def successor_vector(
    task: object,
    profile: AvailabilityProfile,
    allowed: int,
    remaining: int,
    query: int,
) -> tuple[int, ...]:
    return tuple(
        successor_mask(profile, remaining, query, child)
        for child in parent.partition(task, allowed, query)
    )


def compatible_dominates(
    task: object,
    profile: AvailabilityProfile,
    allowed: int,
    remaining: int,
    left: int,
    right: int,
) -> bool:
    left_signature = parent.partition(task, allowed, left)
    right_signature = parent.partition(task, allowed, right)
    if left_signature != right_signature:
        return False
    if successor_vector(task, profile, allowed, remaining, left) != successor_vector(
        task, profile, allowed, remaining, right
    ):
        return False
    left_cost = tuple(parent.cell_cost(profile.response_cost, left, child) for child in left_signature)
    right_cost = tuple(parent.cell_cost(profile.response_cost, right, child) for child in right_signature)
    return parent.representative_dominates(left, left_cost, right, right_cost)


def canonical_mask(
    task: object,
    profile: AvailabilityProfile,
    allowed: int,
    remaining: int,
) -> int:
    informative = [
        query for query in range(task.query_count)
        if remaining & (1 << query)
        and len(parent.partition(task, allowed, query)) > 1
    ]
    keep = 0
    for query in informative:
        if any(
            other != query and compatible_dominates(
                task, profile, allowed, remaining, other, query
            )
            for other in informative
        ):
            continue
        keep |= 1 << query
    return keep


class Planner:
    def __init__(self, task: object, profile: AvailabilityProfile, budget: int, quotient: bool):
        self.task = task
        self.profile = profile
        self.budget = budget
        self.quotient = quotient
        self.memo: dict[tuple[int, int], parent.Plan] = {}
        self.calls = self.expansions = self.memo_hits = 0
        self.raw = self.kept = self.removed = 0

    def solve(self, allowed: int, remaining: int) -> parent.Plan:
        self.calls += 1
        raw = remaining.bit_count()
        active = canonical_mask(self.task, self.profile, allowed, remaining) if self.quotient else remaining
        self.raw += raw
        self.kept += active.bit_count()
        self.removed += max(0, raw - active.bit_count()) if self.quotient else 0
        key = (allowed, active if self.quotient else remaining)
        if key in self.memo:
            self.memo_hits += 1
            return self.memo[key]
        mass = parent.subset_mass(self.profile.response_cost, allowed)
        if state.base.pure_label(self.task, allowed) is not None:
            answer = parent.Plan(mass, 0, 0, None)
            self.memo[key] = answer
            return answer
        candidates = []
        pending = active
        while pending:
            bit = pending & -pending
            query = bit.bit_length() - 1
            pending ^= bit
            children = parent.partition(self.task, allowed, query)
            if len(children) <= 1:
                continue
            self.expansions += 1
            if self.expansions > self.budget:
                raise parent.BudgetExceeded("history availability budget exceeded")
            child_plans = [
                self.solve(child, successor_mask(self.profile, remaining, query, child))
                for child in children
            ]
            candidates.append(parent.Plan(
                sum(plan.diagnosed_mass for plan in child_plans),
                parent.immediate_expected_cost(self.profile.response_cost, allowed, query)
                + sum(plan.expected_cost_numerator for plan in child_plans),
                max(
                    parent.cell_cost(self.profile.response_cost, query, child) + plan.worst_cost
                    for child, plan in zip(children, child_plans)
                ),
                query,
            ))
        answer = max(candidates, key=parent.plan_score) if candidates else parent.Plan(0, 0, 0, None)
        self.memo[key] = answer
        return answer

    def result(self, allowed: int, remaining: int) -> parent.SolveResult:
        plan = self.solve(allowed, remaining)
        return parent.SolveResult(plan, parent.SolverStats(
            self.calls, len(self.memo), self.expansions, self.memo_hits,
            self.raw, self.kept, self.removed,
        ))


def availability_counterexample() -> dict[str, object]:
    task = state.base.make_task(
        "availability-counterexample",
        ("same-a", "same-b", "finish"),
        (("0", "x", "0"), ("1", "y", "0"), ("1", "y", "1")),
        ("a", "b", "c"),
    )
    costs = parent.ResponseCostProfile((1, 1, 1), ((1, 1, 1), (2, 2, 2), (1, 1, 1)), 1)
    all_mask = 0b111
    transitions = (
        (0b100, 0b100, 0b100),
        (0, 0, 0),
        (0, 0, 0),
    )
    profile = AvailabilityProfile(costs, transitions, 1)
    plain = Planner(task, profile, 100, False).result(task.full_mask, all_mask).plan
    unsafe_cost_only = parent.vector_dominates(
        parent.cost_vector(task, costs, task.full_mask, 0),
        parent.cost_vector(task, costs, task.full_mask, 1),
    )
    compatible = compatible_dominates(task, profile, task.full_mask, all_mask, 0, 1)
    return {
        "passed": unsafe_cost_only and not compatible and plain.query == 0,
        "plain_optimal_query": plain.query,
        "cost_only_would_remove_query_1": unsafe_cost_only,
        "history_aware_removes_query_1": compatible,
    }


def random_theorem_certificate() -> dict[str, object]:
    rows = []
    mismatches = []
    reductions = 0
    for seed in range(58001, 58001 + RANDOM_THEOREM_TASKS):
        task, _ = parent.random_task_and_profile(seed)
        profile = availability_profile_for_task(task, seed)
        remaining = (1 << task.query_count) - 1
        plain = Planner(task, profile, BUDGET, False).result(task.full_mask, remaining)
        quotient = Planner(task, profile, BUDGET, True).result(task.full_mask, remaining)
        matched = parent.plan_metrics(plain.plan) == parent.plan_metrics(quotient.plan)
        if quotient.stats.dominated_queries_removed:
            reductions += 1
        if not matched:
            mismatches.append({"seed": seed, "plain": parent.plan_metrics(plain.plan), "quotient": parent.plan_metrics(quotient.plan)})
        rows.append({"seed": seed, "matched": matched, "removed": quotient.stats.dominated_queries_removed})
    return {
        "passed": not mismatches and reductions >= 12,
        "task_count": len(rows),
        "tasks_with_reduction": reductions,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "rows": rows,
    }


def evaluate(task: object, profile: AvailabilityProfile, allowed: int, remaining: int) -> dict[str, object]:
    quotient_result = plain_result = None
    try:
        quotient_result = Planner(task, profile, BUDGET, True).result(allowed, remaining)
    except parent.BudgetExceeded:
        pass
    try:
        plain_result = Planner(task, profile, BUDGET, False).result(allowed, remaining)
    except parent.BudgetExceeded:
        pass
    quotient_solved = quotient_result is not None
    plain_solved = plain_result is not None
    matched = (
        quotient_solved and plain_solved
        and parent.plan_metrics(quotient_result.plan) == parent.plan_metrics(plain_result.plan)
    )
    quotient_exp = quotient_result.stats.query_expansions if quotient_result else BUDGET + 1
    plain_lower = plain_result.stats.query_expansions if plain_result else BUDGET + 1
    return {
        "quotient_solved": quotient_solved,
        "plain_solved": plain_solved,
        "matched_if_both": matched,
        "quotient_plan": parent.plan_metrics(quotient_result.plan) if quotient_result else None,
        "plain_plan": parent.plan_metrics(plain_result.plan) if plain_result else None,
        "quotient_stats": quotient_result.stats.__dict__ if quotient_result else None,
        "plain_stats": plain_result.stats.__dict__ if plain_result else None,
        "expansion_ratio_lower_bound": plain_lower / max(1, quotient_exp),
        "budget_ladder": {
            str(b): {
                "quotient_solved": quotient_solved and quotient_exp <= b,
                "plain_solved": plain_solved and plain_result.stats.query_expansions <= b,
            } for b in BUDGET_LADDER
        },
    }


def run() -> dict[str, object]:
    counterexample = availability_counterexample()
    theorem = random_theorem_certificate()
    tasks, verification = corpus.load_all_opened_tasks()
    rows = []
    base_digests = set()
    for task in tasks:
        selected, _ = frontier.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            base_digest = hashlib.sha256(f"{task.name}:{allowed}:{remaining}".encode()).hexdigest()
            base_digests.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile = availability_profile_for_task(task, seed)
                row = evaluate(task, profile, allowed, remaining)
                row.update(task=task.name, profile_seed=seed, base_state_digest=base_digest)
                rows.append(row)
    solved = [r for r in rows if r["quotient_solved"]]
    both = [r for r in solved if r["plain_solved"]]
    only = [r for r in solved if not r["plain_solved"]]
    ratios = [float(r["expansion_ratio_lower_bound"]) for r in solved]
    removed = sum(int(r["quotient_stats"]["dominated_queries_removed"]) for r in solved)
    ladder = {
        str(b): {k: sum(int(r["budget_ladder"][str(b)][k]) for r in rows) for k in ("quotient_solved", "plain_solved")}
        for b in BUDGET_LADDER
    }
    gate = (
        counterexample["passed"] and theorem["passed"]
        and verification["v39"]["all_hashes_match"] and verification["v41"]["all_hashes_match"]
        and len(base_digests) == 65 and len(rows) == 195
        and len(solved) >= 170 and len(both) >= 100 and len(only) >= 15
        and all(r["matched_if_both"] for r in both)
        and removed >= 300
        and float(np.median(ratios)) >= 2.0
        and float(np.quantile(ratios, 0.9)) >= 8.0
        and ladder["50000"]["quotient_solved"] >= ladder["50000"]["plain_solved"] + 12
    )
    frozen_digest = hashlib.sha256(json.dumps({
        "budget": BUDGET,
        "budget_ladder": BUDGET_LADDER,
        "profile_seeds": PROFILE_SEEDS,
        "random_theorem_tasks": RANDOM_THEOREM_TASKS,
        "rule": "same_partition_same_successor_availability_retain_cost_pareto_frontier",
        "scope": "positive_static_masses_positive_response_costs_response_triggered_test_disabling",
        "base_state_digests": sorted(base_digests),
    }, sort_keys=True).encode()).hexdigest()
    return {
        "status": "history_aware_availability_candidate" if gate else "not_yet",
        "development_gate": gate,
        "claim_scope": (
            "For response-triggered future test disabling, an equivalent test is removable only when a surviving test has a componentwise no-worse response-cost vector and identical successor availability for every corresponding response. Arbitrary action-dependent state transitions remain outside scope."
        ),
        "protocol": {"budget": BUDGET, "budget_ladder": list(BUDGET_LADDER), "profile_seeds": list(PROFILE_SEEDS), "random_theorem_tasks": RANDOM_THEOREM_TASKS},
        "availability_counterexample": counterexample,
        "theorem_certificate": theorem,
        "base_state_count": len(base_digests),
        "state_count": len(rows),
        "quotient_solved_count": len(solved),
        "plain_solved_count": len(both),
        "quotient_only_solved_count": len(only),
        "both_plan_match_count": sum(int(r["matched_if_both"]) for r in both),
        "dominated_queries_removed": removed,
        "expansion_ratio_lower_bound_median": float(np.median(ratios)) if ratios else None,
        "expansion_ratio_lower_bound_p90": float(np.quantile(ratios, 0.9)) if ratios else None,
        "budget_ladder_results": ladder,
        "archive_verification": verification,
        "frozen_history_availability_digest": frozen_digest,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "counterexample": report["availability_counterexample"]["passed"],
        "theorem": report["theorem_certificate"]["passed"],
        "states": report["state_count"],
        "quotient_solved": report["quotient_solved_count"],
        "plain_solved": report["plain_solved_count"],
        "quotient_only": report["quotient_only_solved_count"],
        "removed": report["dominated_queries_removed"],
        "median_ratio": report["expansion_ratio_lower_bound_median"],
        "p90_ratio": report["expansion_ratio_lower_bound_p90"],
    }, indent=2))


if __name__ == "__main__":
    main()
