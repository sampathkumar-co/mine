from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import exact_quotient_certificate_v42 as v42
from . import exact_tail_v36 as v36
from . import state_policy_v34 as v34


FRONTIER_BUDGET = 250_000
MAX_STATES_PER_TASK = 15
MIN_CANDIDATES = 8
MAX_CANDIDATES = 24
MIN_RAW_QUERIES = 18
MAX_QUOTIENT_REPRESENTATIVES = 18
MIN_REDUNDANCY = 6
BUDGET_LADDER = (10_000, 50_000, 250_000)
PARENT_CERTIFICATE_DIGEST = (
    "338cfc90f5afd477b8124e42681ccdc707394f3c577c060d16c00124476deee6"
)


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class SolverStats:
    calls: int
    cache_states: int
    query_expansions: int
    cache_hits: int
    raw_queries_considered: int
    representative_queries_considered: int


@dataclass(frozen=True)
class SolveResult:
    plan: v36.Plan
    stats: SolverStats


def average_plan_score(plan: v36.Plan) -> tuple[int, int, int, int]:
    return (
        plan.diagnosed,
        -plan.total_queries,
        -plan.worst_queries,
        -(plan.query if plan.query is not None else 10**9),
    )


def plan_metrics(plan: v36.Plan) -> tuple[int, int, int]:
    return plan.diagnosed, plan.total_queries, plan.worst_queries


def partition_signature(
    task: object,
    allowed: int,
    query: int,
) -> tuple[int, ...]:
    children = []
    for mask in task.masks_for(query).values():
        child = allowed & mask
        if child:
            children.append(child)
    return tuple(sorted(children))


class AveragePlainPlanner:
    def __init__(self, task: object, budget: int) -> None:
        self.task = task
        self.budget = budget
        self.cache: dict[tuple[int, int], v36.Plan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.cache_hits = 0
        self.raw_queries_considered = 0

    def solve(self, allowed: int, remaining: int) -> v36.Plan:
        self.calls += 1
        key = (allowed, remaining)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        size = allowed.bit_count()
        if v34.base.pure_label(self.task, allowed) is not None:
            plan = v36.Plan(size, 0, 0, None)
            self.cache[key] = plan
            return plan
        candidates = []
        query_bits = remaining
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            self.raw_queries_considered += 1
            children = partition_signature(self.task, allowed, query)
            if len(children) <= 1:
                continue
            self.query_expansions += 1
            if self.query_expansions > self.budget:
                raise BudgetExceeded("plain average-cost budget exceeded")
            next_remaining = remaining & ~(1 << query)
            child_plans = [
                self.solve(child, next_remaining)
                for child in children
            ]
            candidates.append(v36.Plan(
                diagnosed=sum(row.diagnosed for row in child_plans),
                worst_queries=1 + max(
                    row.worst_queries for row in child_plans
                ),
                total_queries=size + sum(
                    row.total_queries for row in child_plans
                ),
                query=query,
            ))
        plan = (
            max(candidates, key=average_plan_score)
            if candidates else v36.Plan(0, 0, 0, None)
        )
        self.cache[key] = plan
        return plan

    def result(self, allowed: int, remaining: int) -> SolveResult:
        plan = self.solve(allowed, remaining)
        return SolveResult(
            plan=plan,
            stats=SolverStats(
                calls=self.calls,
                cache_states=len(self.cache),
                query_expansions=self.query_expansions,
                cache_hits=self.cache_hits,
                raw_queries_considered=self.raw_queries_considered,
                representative_queries_considered=(
                    self.raw_queries_considered
                ),
            ),
        )


class AverageQuotientPlanner:
    def __init__(self, task: object, budget: int) -> None:
        self.task = task
        self.budget = budget
        self.cache: dict[tuple[int, int], v36.Plan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.cache_hits = 0
        self.raw_queries_considered = 0
        self.representative_queries_considered = 0

    def canonical_remaining(self, allowed: int, remaining: int) -> int:
        representatives: dict[tuple[int, ...], int] = {}
        query_bits = remaining
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            self.raw_queries_considered += 1
            signature = partition_signature(self.task, allowed, query)
            if len(signature) <= 1:
                continue
            previous = representatives.get(signature)
            if previous is None or query < previous:
                representatives[signature] = query
        self.representative_queries_considered += len(representatives)
        canonical = 0
        for query in representatives.values():
            canonical |= 1 << query
        return canonical

    def solve(self, allowed: int, remaining: int) -> v36.Plan:
        self.calls += 1
        canonical = self.canonical_remaining(allowed, remaining)
        key = (allowed, canonical)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        size = allowed.bit_count()
        if v34.base.pure_label(self.task, allowed) is not None:
            plan = v36.Plan(size, 0, 0, None)
            self.cache[key] = plan
            return plan
        candidates = []
        query_bits = canonical
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            children = partition_signature(self.task, allowed, query)
            self.query_expansions += 1
            if self.query_expansions > self.budget:
                raise BudgetExceeded("quotient average-cost budget exceeded")
            next_remaining = canonical & ~(1 << query)
            child_plans = [
                self.solve(child, next_remaining)
                for child in children
            ]
            candidates.append(v36.Plan(
                diagnosed=sum(row.diagnosed for row in child_plans),
                worst_queries=1 + max(
                    row.worst_queries for row in child_plans
                ),
                total_queries=size + sum(
                    row.total_queries for row in child_plans
                ),
                query=query,
            ))
        plan = (
            max(candidates, key=average_plan_score)
            if candidates else v36.Plan(0, 0, 0, None)
        )
        self.cache[key] = plan
        return plan

    def result(self, allowed: int, remaining: int) -> SolveResult:
        plan = self.solve(allowed, remaining)
        return SolveResult(
            plan=plan,
            stats=SolverStats(
                calls=self.calls,
                cache_states=len(self.cache),
                query_expansions=self.query_expansions,
                cache_hits=self.cache_hits,
                raw_queries_considered=self.raw_queries_considered,
                representative_queries_considered=(
                    self.representative_queries_considered
                ),
            ),
        )


def expected_elimination_value(
    task: object,
    allowed: int,
    query: int,
) -> int:
    size = allowed.bit_count()
    total = 0
    for child in partition_signature(task, allowed, query):
        bucket = child.bit_count()
        total += bucket * (size - bucket)
    return total


def select_expected_elimination_query(
    task: object,
    allowed: int,
    remaining: int,
) -> int:
    candidates = []
    query_bits = remaining
    while query_bits:
        bit = query_bits & -query_bits
        query = bit.bit_length() - 1
        query_bits ^= bit
        if len(partition_signature(task, allowed, query)) <= 1:
            continue
        candidates.append(query)
    if not candidates:
        raise RuntimeError("no separating query")
    return max(
        candidates,
        key=lambda query: (
            expected_elimination_value(task, allowed, query),
            -query,
        ),
    )


class ExpectedEliminationGreedy:
    def __init__(self, task: object) -> None:
        self.task = task
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
            query = select_expected_elimination_query(
                self.task, allowed, remaining
            )
        except RuntimeError:
            plan = v36.Plan(0, 0, 0, None)
            self.cache[key] = plan
            return plan
        next_remaining = remaining & ~(1 << query)
        child_plans = [
            self.solve(child, next_remaining)
            for child in partition_signature(self.task, allowed, query)
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


def quotient_representative_count(
    task: object,
    allowed: int,
    remaining: int,
) -> int:
    signatures = set()
    query_bits = remaining
    while query_bits:
        bit = query_bits & -query_bits
        query = bit.bit_length() - 1
        query_bits ^= bit
        signature = partition_signature(task, allowed, query)
        if len(signature) > 1:
            signatures.add(signature)
    return len(signatures)


def structural_rank(
    task_name: str,
    allowed: int,
    remaining: int,
    representatives: int,
) -> tuple[int, int, int, str]:
    return (
        -allowed.bit_count(),
        -representatives,
        -remaining.bit_count(),
        hashlib.sha256(
            f"{task_name}:{allowed}:{remaining}".encode("utf-8")
        ).hexdigest(),
    )


def select_frontier_states(task: object) -> tuple[
    list[tuple[int, int, int]],
    dict[str, int],
]:
    states = v42.collect_policy_states(task)
    candidates = []
    for allowed, remaining in states:
        size = allowed.bit_count()
        raw = remaining.bit_count()
        representatives = quotient_representative_count(
            task, allowed, remaining
        )
        if (
            MIN_CANDIDATES <= size <= MAX_CANDIDATES
            and raw >= MIN_RAW_QUERIES
            and representatives <= MAX_QUOTIENT_REPRESENTATIVES
            and raw - representatives >= MIN_REDUNDANCY
        ):
            candidates.append((allowed, remaining, representatives))
    candidates.sort(
        key=lambda row: structural_rank(
            task.name, row[0], row[1], row[2]
        )
    )
    return candidates[:MAX_STATES_PER_TASK], {
        "harvested_states": len(states),
        "frontier_candidates": len(candidates),
        "selected_states": min(len(candidates), MAX_STATES_PER_TASK),
    }


def solve_state(
    task: object,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    quotient_solved = False
    plain_solved = False
    quotient_result = None
    plain_result = None
    try:
        quotient_result = AverageQuotientPlanner(
            task, FRONTIER_BUDGET
        ).result(allowed, remaining)
        quotient_solved = True
    except BudgetExceeded:
        pass
    if quotient_solved:
        try:
            plain_result = AveragePlainPlanner(
                task, FRONTIER_BUDGET
            ).result(allowed, remaining)
            plain_solved = True
        except BudgetExceeded:
            pass
    greedy_plan = ExpectedEliminationGreedy(task).solve(
        allowed, remaining
    )
    matched = (
        quotient_solved
        and plain_solved
        and plan_metrics(quotient_result.plan)
        == plan_metrics(plain_result.plan)
    )
    exact_dominates_greedy = (
        quotient_solved
        and average_plan_score(quotient_result.plan)
        >= average_plan_score(greedy_plan)
    )
    strict_exact_gain = (
        quotient_solved
        and average_plan_score(quotient_result.plan)
        > average_plan_score(greedy_plan)
    )
    equal_diagnosis = (
        quotient_solved
        and quotient_result.plan.diagnosed == greedy_plan.diagnosed
    )
    total_query_saving = (
        greedy_plan.total_queries - quotient_result.plan.total_queries
        if equal_diagnosis else None
    )
    if quotient_solved:
        plain_lower = (
            plain_result.stats.query_expansions
            if plain_solved else FRONTIER_BUDGET + 1
        )
        ratio = plain_lower / max(
            1, quotient_result.stats.query_expansions
        )
    else:
        ratio = None
    ladder = {}
    for budget in BUDGET_LADDER:
        ladder[str(budget)] = {
            "quotient_solved": (
                quotient_solved
                and quotient_result.stats.query_expansions <= budget
            ),
            "plain_solved": (
                plain_solved
                and plain_result.stats.query_expansions <= budget
            ),
        }
    root_selected = None
    root_maximum = None
    try:
        root_query = select_expected_elimination_query(
            task, allowed, remaining
        )
        root_selected = expected_elimination_value(
            task, allowed, root_query
        )
        values = []
        query_bits = remaining
        while query_bits:
            bit = query_bits & -query_bits
            query = bit.bit_length() - 1
            query_bits ^= bit
            if len(partition_signature(task, allowed, query)) > 1:
                values.append(expected_elimination_value(
                    task, allowed, query
                ))
        root_maximum = max(values)
    except RuntimeError:
        pass
    return {
        "candidate_count": allowed.bit_count(),
        "raw_remaining_queries": remaining.bit_count(),
        "quotient_representatives": quotient_representative_count(
            task, allowed, remaining
        ),
        "quotient_solved": quotient_solved,
        "plain_solved": plain_solved,
        "matched_if_both": matched,
        "quotient_plan": (
            plan_metrics(quotient_result.plan)
            if quotient_result is not None else None
        ),
        "plain_plan": (
            plan_metrics(plain_result.plan)
            if plain_result is not None else None
        ),
        "greedy_plan": plan_metrics(greedy_plan),
        "exact_dominates_greedy": exact_dominates_greedy,
        "strict_exact_gain": strict_exact_gain,
        "equal_diagnosis": equal_diagnosis,
        "total_query_saving": total_query_saving,
        "quotient_stats": (
            quotient_result.stats.__dict__
            if quotient_result is not None else None
        ),
        "plain_stats": (
            plain_result.stats.__dict__
            if plain_result is not None else None
        ),
        "expansion_ratio_lower_bound": ratio,
        "expected_elimination_root_certificate": {
            "selected_value": root_selected,
            "maximum_value": root_maximum,
            "passed": root_selected == root_maximum,
        },
        "budget_ladder": ladder,
    }


def run() -> dict[str, object]:
    tasks, verification = v42.load_all_opened_tasks()
    rows = []
    task_summaries = []
    for task in tasks:
        selected, summary = select_frontier_states(task)
        summary["task"] = task.name
        task_summaries.append(summary)
        for allowed, remaining, representatives in selected:
            row = solve_state(task, allowed, remaining)
            assert row["quotient_representatives"] == representatives
            row["task"] = task.name
            row["state_digest"] = hashlib.sha256(
                f"{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            rows.append(row)
    solved = [row for row in rows if row["quotient_solved"]]
    both = [row for row in solved if row["plain_solved"]]
    quotient_only = [
        row for row in solved if not row["plain_solved"]
    ]
    ratios = [
        float(row["expansion_ratio_lower_bound"])
        for row in solved
    ]
    equal_diagnosis_rows = [
        row for row in solved if row["equal_diagnosis"]
    ]
    savings = [
        int(row["total_query_saving"])
        for row in equal_diagnosis_rows
    ]
    ladder = {
        str(budget): {
            "quotient_solved": sum(
                int(row["budget_ladder"][str(budget)][
                    "quotient_solved"
                ])
                for row in rows
            ),
            "plain_solved": sum(
                int(row["budget_ladder"][str(budget)][
                    "plain_solved"
                ])
                for row in rows
            ),
        }
        for budget in BUDGET_LADDER
    }
    gate = (
        verification["v39"]["all_hashes_match"]
        and verification["v41"]["all_hashes_match"]
        and len(rows) >= 50
        and len(solved) >= 45
        and len(quotient_only) >= 10
        and len(both) >= 20
        and all(row["matched_if_both"] for row in both)
        and all(row["exact_dominates_greedy"] for row in solved)
        and sum(int(row["strict_exact_gain"]) for row in solved) >= 15
        and len(equal_diagnosis_rows) == len(solved)
        and sum(savings) >= 100
        and all(
            row["expected_elimination_root_certificate"]["passed"]
            for row in rows
        )
        and float(np.median(ratios)) >= 5.0
        and float(np.quantile(ratios, 0.9)) >= 20.0
        and ladder["50000"]["quotient_solved"] >= (
            ladder["50000"]["plain_solved"] + 10
        )
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "parent_certificate_digest": PARENT_CERTIFICATE_DIGEST,
                "budget": FRONTIER_BUDGET,
                "budget_ladder": BUDGET_LADDER,
                "selection": {
                    "max_states_per_task": MAX_STATES_PER_TASK,
                    "candidate_range": [
                        MIN_CANDIDATES, MAX_CANDIDATES
                    ],
                    "minimum_raw_queries": MIN_RAW_QUERIES,
                    "maximum_quotient_representatives": (
                        MAX_QUOTIENT_REPRESENTATIVES
                    ),
                    "minimum_redundancy": MIN_REDUNDANCY,
                },
                "objective": (
                    "max_diagnosed_min_total_queries_min_worst_depth"
                ),
                "greedy": (
                    "uniform_expected_eliminations_per_unit_cost"
                ),
                "task_names": [task.name for task in tasks],
                "verification": verification,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": (
            "average_cost_exact_frontier_candidate"
            if gate else "not_yet"
        ),
        "claim_scope": (
            "a state-local response-partition quotient is evaluated against a "
            "matched unquotiented global exact solver for uniform-prior total "
            "identification cost, while the current expected-elimination greedy "
            "approximation is implemented directly; a pass is a strong exact "
            "frontier and baseline result, but independent implementation and "
            "novelty review remain necessary before a world-class claim"
        ),
        "development_gate": gate,
        "protocol": {
            "parent_certificate_digest": PARENT_CERTIFICATE_DIGEST,
            "frontier_budget": FRONTIER_BUDGET,
            "budget_ladder": list(BUDGET_LADDER),
            "max_states_per_task": MAX_STATES_PER_TASK,
            "candidate_range": [MIN_CANDIDATES, MAX_CANDIDATES],
            "minimum_raw_queries": MIN_RAW_QUERIES,
            "maximum_quotient_representatives": (
                MAX_QUOTIENT_REPRESENTATIVES
            ),
            "minimum_redundancy": MIN_REDUNDANCY,
        },
        "task_count": len(tasks),
        "task_summaries": task_summaries,
        "frontier_state_count": len(rows),
        "quotient_solved_count": len(solved),
        "plain_solved_count": len(both),
        "quotient_only_solved_count": len(quotient_only),
        "both_plan_match_count": sum(
            int(row["matched_if_both"]) for row in both
        ),
        "exact_dominates_greedy_count": sum(
            int(row["exact_dominates_greedy"]) for row in solved
        ),
        "strict_exact_gain_count": sum(
            int(row["strict_exact_gain"]) for row in solved
        ),
        "aggregate_total_query_saving_vs_greedy": sum(savings),
        "expansion_ratio_lower_bound_median": (
            float(np.median(ratios)) if ratios else None
        ),
        "expansion_ratio_lower_bound_p90": (
            float(np.quantile(ratios, 0.9)) if ratios else None
        ),
        "budget_ladder_results": ladder,
        "rows": rows,
        "archive_verification": verification,
        "frozen_frontier_digest": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "frontier_states": report["frontier_state_count"],
        "quotient_solved": report["quotient_solved_count"],
        "plain_solved": report["plain_solved_count"],
        "quotient_only": report["quotient_only_solved_count"],
        "strict_exact_gains": report["strict_exact_gain_count"],
        "query_saving": report[
            "aggregate_total_query_saving_vs_greedy"
        ],
        "median_ratio": report[
            "expansion_ratio_lower_bound_median"
        ],
        "p90_ratio": report[
            "expansion_ratio_lower_bound_p90"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
