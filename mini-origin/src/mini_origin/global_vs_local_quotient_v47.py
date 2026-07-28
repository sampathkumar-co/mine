from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import average_odt_frontier_v44 as v44
from . import exact_quotient_certificate_v42 as corpus
from . import exact_tail_v36 as plans
from . import state_policy_v34 as state


BUDGET = 250_000
BUDGET_LADDER = (10_000, 50_000, 250_000)


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Stats:
    calls: int
    memo_entries: int
    query_expansions: int
    memo_hits: int
    root_raw_queries: int
    root_representatives: int


@dataclass(frozen=True)
class Result:
    plan: plans.Plan
    stats: Stats


def partition(task: object, allowed: int, query: int) -> tuple[int, ...]:
    children = []
    for mask in task.masks_for(query).values():
        child = allowed & mask
        if child:
            children.append(child)
    return tuple(sorted(children))


def one_time_global_mask(
    task: object,
    root_allowed: int,
    remaining: int,
) -> int:
    """Collapse tests equal on the benchmark state's initial candidate set."""
    representatives: dict[tuple[int, ...], int] = {}
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        signature = partition(task, root_allowed, query)
        if len(signature) <= 1:
            continue
        previous = representatives.get(signature)
        if previous is None or query < previous:
            representatives[signature] = query
    mask = 0
    for query in representatives.values():
        mask |= 1 << query
    return mask


class GlobalOnlyQuotientPlanner:
    """Exact DP after one root-level duplicate-feature preprocessing pass."""

    def __init__(
        self,
        task: object,
        root_allowed: int,
        root_remaining: int,
        budget: int,
    ) -> None:
        self.task = task
        self.budget = budget
        self.root_raw_queries = root_remaining.bit_count()
        self.root_mask = one_time_global_mask(
            task, root_allowed, root_remaining
        )
        self.memo: dict[tuple[int, int], plans.Plan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.memo_hits = 0

    def solve(self, allowed: int, remaining: int) -> plans.Plan:
        self.calls += 1
        # No descendant canonicalisation: remaining is only intersected with the
        # one-time root representative set.
        available = remaining & self.root_mask
        key = allowed, available
        cached = self.memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            return cached
        population = allowed.bit_count()
        if state.base.pure_label(self.task, allowed) is not None:
            answer = plans.Plan(population, 0, 0, None)
            self.memo[key] = answer
            return answer

        candidates = []
        pending = available
        while pending:
            bit = pending & -pending
            query = bit.bit_length() - 1
            pending ^= bit
            children = partition(self.task, allowed, query)
            if len(children) <= 1:
                continue
            self.query_expansions += 1
            if self.query_expansions > self.budget:
                raise BudgetExceeded(
                    "global-only quotient expansion budget exceeded"
                )
            next_remaining = available & ~(1 << query)
            child_plans = [
                self.solve(child, next_remaining) for child in children
            ]
            candidates.append(plans.Plan(
                diagnosed=sum(row.diagnosed for row in child_plans),
                total_queries=population + sum(
                    row.total_queries for row in child_plans
                ),
                worst_queries=1 + max(
                    row.worst_queries for row in child_plans
                ),
                query=query,
            ))
        answer = (
            max(candidates, key=v44.average_plan_score)
            if candidates else plans.Plan(0, 0, 0, None)
        )
        self.memo[key] = answer
        return answer

    def result(self, allowed: int, remaining: int) -> Result:
        answer = self.solve(allowed, remaining)
        return Result(
            answer,
            Stats(
                calls=self.calls,
                memo_entries=len(self.memo),
                query_expansions=self.query_expansions,
                memo_hits=self.memo_hits,
                root_raw_queries=self.root_raw_queries,
                root_representatives=self.root_mask.bit_count(),
            ),
        )


def evaluate_state(
    task: object,
    allowed: int,
    remaining: int,
) -> dict[str, object]:
    local_result = None
    global_result = None
    local_solved = False
    global_solved = False
    try:
        local_result = v44.AverageQuotientPlanner(
            task, BUDGET
        ).result(allowed, remaining)
        local_solved = True
    except v44.BudgetExceeded:
        pass
    try:
        global_result = GlobalOnlyQuotientPlanner(
            task, allowed, remaining, BUDGET
        ).result(allowed, remaining)
        global_solved = True
    except BudgetExceeded:
        pass

    matched = (
        local_solved
        and global_solved
        and v44.plan_metrics(local_result.plan)
        == v44.plan_metrics(global_result.plan)
    )
    if local_solved:
        global_lower = (
            global_result.stats.query_expansions
            if global_solved else BUDGET + 1
        )
        ratio = global_lower / max(
            1, local_result.stats.query_expansions
        )
    else:
        ratio = None
    ladder = {}
    for budget in BUDGET_LADDER:
        ladder[str(budget)] = {
            "local_solved": (
                local_solved
                and local_result.stats.query_expansions <= budget
            ),
            "global_solved": (
                global_solved
                and global_result.stats.query_expansions <= budget
            ),
        }
    return {
        "candidate_count": allowed.bit_count(),
        "raw_remaining_queries": remaining.bit_count(),
        "root_global_representatives": one_time_global_mask(
            task, allowed, remaining
        ).bit_count(),
        "local_root_representatives": v44.quotient_representative_count(
            task, allowed, remaining
        ),
        "local_solved": local_solved,
        "global_solved": global_solved,
        "matched_if_both": matched,
        "local_plan": (
            v44.plan_metrics(local_result.plan)
            if local_result is not None else None
        ),
        "global_plan": (
            v44.plan_metrics(global_result.plan)
            if global_result is not None else None
        ),
        "local_stats": (
            local_result.stats.__dict__
            if local_result is not None else None
        ),
        "global_stats": (
            global_result.stats.__dict__
            if global_result is not None else None
        ),
        "expansion_ratio_lower_bound": ratio,
        "budget_ladder": ladder,
    }


def run() -> dict[str, object]:
    tasks, verification = corpus.load_all_opened_tasks()
    rows = []
    for task in tasks:
        selected, _ = v44.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            row = evaluate_state(task, allowed, remaining)
            row["task"] = task.name
            row["state_digest"] = hashlib.sha256(
                f"{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            rows.append(row)

    local = [row for row in rows if row["local_solved"]]
    both = [row for row in local if row["global_solved"]]
    local_only = [row for row in local if not row["global_solved"]]
    ratios = [
        float(row["expansion_ratio_lower_bound"]) for row in local
    ]
    ladder = {
        str(budget): {
            "local_solved": sum(
                int(row["budget_ladder"][str(budget)]["local_solved"])
                for row in rows
            ),
            "global_solved": sum(
                int(row["budget_ladder"][str(budget)]["global_solved"])
                for row in rows
            ),
        }
        for budget in BUDGET_LADDER
    }
    gate = (
        verification["v39"]["all_hashes_match"]
        and verification["v41"]["all_hashes_match"]
        and len(rows) == 65
        and len(local) == 65
        and len(local_only) >= 5
        and len(both) >= 20
        and all(row["matched_if_both"] for row in both)
        and float(np.median(ratios)) >= 2.0
        and float(np.quantile(ratios, 0.9)) >= 5.0
        and ladder["50000"]["local_solved"] >= (
            ladder["50000"]["global_solved"] + 10
        )
    )
    return {
        "status": (
            "state_local_advantage_certified" if gate else "not_yet"
        ),
        "development_gate": gate,
        "comparison_scope": (
            "The global-only solver performs the strongest one-time analogue "
            "of global duplicate-feature preprocessing on each benchmark state. "
            "The local solver recomputes response-partition equivalence at every "
            "descendant. Both optimise the same exact average-cost objective."
        ),
        "state_count": len(rows),
        "local_solved_count": len(local),
        "global_solved_count": len(both),
        "local_only_solved_count": len(local_only),
        "both_plan_match_count": sum(
            int(row["matched_if_both"]) for row in both
        ),
        "expansion_ratio_lower_bound_median": float(np.median(ratios)),
        "expansion_ratio_lower_bound_p90": float(np.quantile(ratios, 0.9)),
        "budget_ladder_results": ladder,
        "archive_verification": verification,
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
        "states": report["state_count"],
        "local_solved": report["local_solved_count"],
        "global_solved": report["global_solved_count"],
        "local_only": report["local_only_solved_count"],
        "median_ratio": report["expansion_ratio_lower_bound_median"],
        "p90_ratio": report["expansion_ratio_lower_bound_p90"],
        "ladder": report["budget_ladder_results"],
    }, indent=2))


if __name__ == "__main__":
    main()
