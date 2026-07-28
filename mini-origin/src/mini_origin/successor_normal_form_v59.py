from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import average_odt_frontier_v44 as frontier
from . import exact_quotient_certificate_v42 as corpus
from . import history_availability_v58 as previous
from . import response_cost_pareto_v56 as parent
from . import state_policy_v34 as state

BUDGET = previous.BUDGET
BUDGET_LADDER = previous.BUDGET_LADDER
PROFILE_SEEDS = previous.PROFILE_SEEDS
RANDOM_THEOREM_TASKS = previous.RANDOM_THEOREM_TASKS


class NormalFormReducer:
    def __init__(self, task: object, profile: previous.AvailabilityProfile):
        self.task = task
        self.profile = profile
        self.cache: dict[tuple[int, int], int] = {}
        self.in_progress: set[tuple[int, int]] = set()

    def normal_form(self, allowed: int, remaining: int) -> int:
        key = (allowed, remaining)
        if key in self.cache:
            return self.cache[key]
        if state.base.pure_label(self.task, allowed) is not None:
            self.cache[key] = 0
            return 0
        if key in self.in_progress:
            # Informative queries strictly shrink `allowed`; this guard is only
            # a defensive fallback for malformed non-shrinking transitions.
            return remaining
        self.in_progress.add(key)
        informative = [
            query for query in range(self.task.query_count)
            if remaining & (1 << query)
            and len(parent.partition(self.task, allowed, query)) > 1
        ]
        keep = 0
        for query in informative:
            if any(
                other != query and self.normal_form_dominates(
                    allowed, remaining, other, query
                )
                for other in informative
            ):
                continue
            keep |= 1 << query
        self.in_progress.remove(key)
        self.cache[key] = keep
        return keep

    def successor_normal_vector(
        self, allowed: int, remaining: int, query: int
    ) -> tuple[int, ...]:
        return tuple(
            self.normal_form(
                child,
                previous.successor_mask(self.profile, remaining, query, child),
            )
            for child in parent.partition(self.task, allowed, query)
        )

    def normal_form_dominates(
        self, allowed: int, remaining: int, left: int, right: int
    ) -> bool:
        left_partition = parent.partition(self.task, allowed, left)
        right_partition = parent.partition(self.task, allowed, right)
        if left_partition != right_partition:
            return False
        if self.successor_normal_vector(allowed, remaining, left) != self.successor_normal_vector(
            allowed, remaining, right
        ):
            return False
        left_cost = tuple(
            parent.cell_cost(self.profile.response_cost, left, child)
            for child in left_partition
        )
        right_cost = tuple(
            parent.cell_cost(self.profile.response_cost, right, child)
            for child in right_partition
        )
        return parent.representative_dominates(left, left_cost, right, right_cost)


class Planner:
    def __init__(
        self,
        task: object,
        profile: previous.AvailabilityProfile,
        budget: int,
        quotient: bool,
    ):
        self.task = task
        self.profile = profile
        self.budget = budget
        self.quotient = quotient
        self.reducer = NormalFormReducer(task, profile)
        self.memo: dict[tuple[int, int], parent.Plan] = {}
        self.calls = self.expansions = self.memo_hits = 0
        self.raw = self.kept = self.removed = 0

    def solve(self, allowed: int, remaining: int) -> parent.Plan:
        self.calls += 1
        raw = remaining.bit_count()
        active = self.reducer.normal_form(allowed, remaining) if self.quotient else remaining
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
        candidates: list[parent.Plan] = []
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
                raise parent.BudgetExceeded("successor normal-form budget exceeded")
            child_plans = [
                self.solve(
                    child,
                    previous.successor_mask(self.profile, remaining, query, child),
                )
                for child in children
            ]
            candidates.append(parent.Plan(
                sum(plan.diagnosed_mass for plan in child_plans),
                parent.immediate_expected_cost(self.profile.response_cost, allowed, query)
                + sum(plan.expected_cost_numerator for plan in child_plans),
                max(
                    parent.cell_cost(self.profile.response_cost, query, child)
                    + plan.worst_cost
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
            self.calls,
            len(self.memo),
            self.expansions,
            self.memo_hits,
            self.raw,
            self.kept,
            self.removed,
        ))


def normal_form_counterexample() -> dict[str, object]:
    task = state.base.make_task(
        "successor-normal-form-counterexample",
        ("same-cheap", "same-costly", "finish-cheap", "finish-dominated"),
        (
            ("0", "0", "0", "0"),
            ("0", "0", "1", "1"),
            ("1", "1", "0", "0"),
            ("1", "1", "1", "1"),
        ),
        ("a", "b", "c", "d"),
    )
    costs = parent.ResponseCostProfile(
        (1, 1, 1, 1),
        (
            (1, 1, 1, 1),
            (2, 2, 2, 2),
            (1, 1, 1, 1),
            (2, 2, 2, 2),
        ),
        1,
    )
    # q0 leaves q2 and its dominated duplicate q3; q1 leaves only q2.
    transitions = (
        (0b1100, 0b1100, 0b1100, 0b1100),
        (0b0100, 0b0100, 0b0100, 0b0100),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    profile = previous.AvailabilityProfile(costs, transitions, 1)
    remaining = 0b1111
    strict_raw = previous.compatible_dominates(
        task, profile, task.full_mask, remaining, 0, 1
    )
    reducer = NormalFormReducer(task, profile)
    recursive = reducer.normal_form_dominates(task.full_mask, remaining, 0, 1)
    plain = Planner(task, profile, 1000, False).result(task.full_mask, remaining)
    quotient = Planner(task, profile, 1000, True).result(task.full_mask, remaining)
    return {
        "passed": (
            not strict_raw
            and recursive
            and parent.plan_metrics(plain.plan) == parent.plan_metrics(quotient.plan)
            and quotient.stats.dominated_queries_removed > 0
        ),
        "raw_successor_rule_can_remove": strict_raw,
        "normal_form_rule_can_remove": recursive,
        "plain_plan": parent.plan_metrics(plain.plan),
        "quotient_plan": parent.plan_metrics(quotient.plan),
        "removed": quotient.stats.dominated_queries_removed,
    }


def random_theorem_certificate() -> dict[str, object]:
    rows = []
    mismatches = []
    reductions = 0
    normal_form_extra = 0
    for seed in range(59001, 59001 + RANDOM_THEOREM_TASKS):
        task, _ = parent.random_task_and_profile(seed)
        profile = previous.availability_profile_for_task(task, seed)
        remaining = (1 << task.query_count) - 1
        plain = Planner(task, profile, BUDGET, False).result(task.full_mask, remaining)
        quotient = Planner(task, profile, BUDGET, True).result(task.full_mask, remaining)
        strict = previous.Planner(task, profile, BUDGET, True).result(task.full_mask, remaining)
        matched = parent.plan_metrics(plain.plan) == parent.plan_metrics(quotient.plan)
        removed = quotient.stats.dominated_queries_removed
        if removed:
            reductions += 1
        if removed > strict.stats.dominated_queries_removed:
            normal_form_extra += 1
        if not matched:
            mismatches.append({
                "seed": seed,
                "plain": parent.plan_metrics(plain.plan),
                "quotient": parent.plan_metrics(quotient.plan),
            })
        rows.append({
            "seed": seed,
            "matched": matched,
            "removed": removed,
            "strict_removed": strict.stats.dominated_queries_removed,
        })
    return {
        "passed": not mismatches and reductions >= 12 and normal_form_extra >= 8,
        "task_count": len(rows),
        "tasks_with_reduction": reductions,
        "tasks_with_extra_normal_form_reduction": normal_form_extra,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "rows": rows,
    }


def evaluate(
    task: object,
    profile: previous.AvailabilityProfile,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    quotient_result = plain_result = strict_result = None
    try:
        quotient_result = Planner(task, profile, BUDGET, True).result(allowed, remaining)
    except parent.BudgetExceeded:
        pass
    try:
        plain_result = Planner(task, profile, BUDGET, False).result(allowed, remaining)
    except parent.BudgetExceeded:
        pass
    try:
        strict_result = previous.Planner(task, profile, BUDGET, True).result(allowed, remaining)
    except parent.BudgetExceeded:
        pass
    quotient_solved = quotient_result is not None
    plain_solved = plain_result is not None
    strict_solved = strict_result is not None
    matched = (
        quotient_solved
        and plain_solved
        and parent.plan_metrics(quotient_result.plan) == parent.plan_metrics(plain_result.plan)
    )
    quotient_exp = quotient_result.stats.query_expansions if quotient_result else BUDGET + 1
    plain_lower = plain_result.stats.query_expansions if plain_result else BUDGET + 1
    return {
        "quotient_solved": quotient_solved,
        "plain_solved": plain_solved,
        "strict_v58_solved": strict_solved,
        "matched_if_both": matched,
        "quotient_plan": parent.plan_metrics(quotient_result.plan) if quotient_result else None,
        "plain_plan": parent.plan_metrics(plain_result.plan) if plain_result else None,
        "quotient_stats": quotient_result.stats.__dict__ if quotient_result else None,
        "plain_stats": plain_result.stats.__dict__ if plain_result else None,
        "strict_v58_stats": strict_result.stats.__dict__ if strict_result else None,
        "expansion_ratio_lower_bound": plain_lower / max(1, quotient_exp),
        "budget_ladder": {
            str(b): {
                "quotient_solved": quotient_solved and quotient_exp <= b,
                "plain_solved": plain_solved and plain_result.stats.query_expansions <= b,
                "strict_v58_solved": strict_solved and strict_result.stats.query_expansions <= b,
            }
            for b in BUDGET_LADDER
        },
    }


def run() -> dict[str, object]:
    counterexample = normal_form_counterexample()
    theorem = random_theorem_certificate()
    tasks, verification = corpus.load_all_opened_tasks()
    rows = []
    base_digests = set()
    for task in tasks:
        selected, _ = frontier.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            base_digest = hashlib.sha256(
                f"{task.name}:{allowed}:{remaining}".encode()
            ).hexdigest()
            base_digests.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile = previous.availability_profile_for_task(task, seed)
                row = evaluate(task, profile, allowed, remaining)
                row.update(
                    task=task.name,
                    profile_seed=seed,
                    base_state_digest=base_digest,
                )
                rows.append(row)
    solved = [row for row in rows if row["quotient_solved"]]
    both = [row for row in solved if row["plain_solved"]]
    only = [row for row in solved if not row["plain_solved"]]
    strict_only_recovered = [
        row for row in solved if not row["strict_v58_solved"]
    ]
    ratios = [float(row["expansion_ratio_lower_bound"]) for row in solved]
    removed = sum(
        int(row["quotient_stats"]["dominated_queries_removed"])
        for row in solved
    )
    ladder = {
        str(b): {
            key: sum(int(row["budget_ladder"][str(b)][key]) for row in rows)
            for key in ("quotient_solved", "plain_solved", "strict_v58_solved")
        }
        for b in BUDGET_LADDER
    }
    gate = (
        counterexample["passed"]
        and theorem["passed"]
        and verification["v39"]["all_hashes_match"]
        and verification["v41"]["all_hashes_match"]
        and len(base_digests) == 65
        and len(rows) == 195
        and len(solved) >= 170
        and len(both) >= 100
        and len(only) >= 15
        and all(row["matched_if_both"] for row in both)
        and removed >= 300
        and float(np.median(ratios)) >= 2.0
        and float(np.quantile(ratios, 0.9)) >= 8.0
        and ladder["50000"]["quotient_solved"]
        >= ladder["50000"]["plain_solved"] + 12
        and ladder["50000"]["quotient_solved"]
        >= ladder["50000"]["strict_v58_solved"] + 3
    )
    frozen_digest = hashlib.sha256(json.dumps({
        "budget": BUDGET,
        "budget_ladder": BUDGET_LADDER,
        "profile_seeds": PROFILE_SEEDS,
        "random_theorem_tasks": RANDOM_THEOREM_TASKS,
        "rule": "same_partition_cost_dominance_equal_recursive_successor_normal_forms",
        "scope": "positive_static_masses_positive_response_costs_response_triggered_test_disabling",
        "base_state_digests": sorted(base_digests),
    }, sort_keys=True).encode()).hexdigest()
    return {
        "status": "successor_normal_form_candidate" if gate else "not_yet",
        "development_gate": gate,
        "claim_scope": (
            "For response-triggered future test disabling, recursively normalised successor availability states may replace exact raw successor-mask equality. The claim is limited to inherited disabling transitions, positive static masses and positive response-dependent costs."
        ),
        "protocol": {
            "budget": BUDGET,
            "budget_ladder": list(BUDGET_LADDER),
            "profile_seeds": list(PROFILE_SEEDS),
            "random_theorem_tasks": RANDOM_THEOREM_TASKS,
        },
        "normal_form_counterexample": counterexample,
        "theorem_certificate": theorem,
        "base_state_count": len(base_digests),
        "state_count": len(rows),
        "quotient_solved_count": len(solved),
        "plain_solved_count": len(both),
        "quotient_only_solved_count": len(only),
        "strict_v58_unsolved_recovered_count": len(strict_only_recovered),
        "both_plan_match_count": sum(int(row["matched_if_both"]) for row in both),
        "dominated_queries_removed": removed,
        "expansion_ratio_lower_bound_median": float(np.median(ratios)) if ratios else None,
        "expansion_ratio_lower_bound_p90": float(np.quantile(ratios, 0.9)) if ratios else None,
        "budget_ladder_results": ladder,
        "archive_verification": verification,
        "frozen_successor_normal_form_digest": frozen_digest,
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
        "counterexample": report["normal_form_counterexample"]["passed"],
        "theorem": report["theorem_certificate"]["passed"],
        "states": report["state_count"],
        "quotient_solved": report["quotient_solved_count"],
        "plain_solved": report["plain_solved_count"],
        "quotient_only": report["quotient_only_solved_count"],
        "v58_recovered": report["strict_v58_unsolved_recovered_count"],
        "median_ratio": report["expansion_ratio_lower_bound_median"],
        "p90_ratio": report["expansion_ratio_lower_bound_p90"],
    }, indent=2))


if __name__ == "__main__":
    main()
