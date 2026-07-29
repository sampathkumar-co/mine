from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from . import clean_external_conditioned_v64 as clean_v64
from . import conditioned_cell_frontier_v60 as conditioned
from . import external_response_cost_v58 as external
from . import response_cost_pareto_v56 as response
from . import state_policy_v34 as state


RANDOM_CERTIFICATE_TASKS = 96
RANDOM_SEED_START = 65_001
BUDGET = response.BUDGET
BUDGET_LADDER = response.BUDGET_LADDER
PREREGISTRATION = Path(__file__).resolve().parents[2] / "campaigns" / "v65-pareto-lower-bound-preregistration.json"
V64_EVIDENCE = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v64-clean-external-rejected.json"


@dataclass(frozen=True)
class BoundStats:
    calls: int
    memo_entries: int
    query_expansions: int
    memo_hits: int
    raw_queries_considered: int
    representative_queries_considered: int
    dominated_queries_removed: int
    bound_evaluations: int
    bound_pruned_queries: int
    impossible_full_diagnosis_prunes: int


@dataclass(frozen=True)
class BoundResult:
    plan: response.Plan
    stats: BoundStats


@dataclass(frozen=True)
class CandidateBound:
    query: int
    children: tuple[int, ...]
    full_diagnosis_possible: bool
    expected_cost_lower_bound: int
    worst_cost_lower_bound: int


def impure_child_first_step_bound(
    task: object,
    profile: response.ResponseCostProfile,
    allowed: int,
    remaining: int,
) -> tuple[bool, int, int]:
    """Return an optimistic lexicographic first-step bound for full diagnosis.

    For an impure child, every full-diagnosis continuation must first choose one
    separating remaining test. Its expected cost is therefore at least the
    minimum immediate expected cost of such a test. Conditional on attaining
    that minimum, its worst cost is at least the minimum immediate worst cost
    among tied tests. If no separating test remains, full diagnosis is
    impossible from the child.
    """
    if state.base.pure_label(task, allowed) is not None:
        return True, 0, 0

    rows: list[tuple[int, int, int]] = []
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        children = response.partition(task, allowed, query)
        if len(children) <= 1:
            continue
        immediate_expected = response.immediate_expected_cost(
            profile, allowed, query
        )
        immediate_worst = max(
            response.cell_cost(profile, query, child)
            for child in children
        )
        rows.append((immediate_expected, immediate_worst, query))

    if not rows:
        return False, 0, 0
    minimum_expected = min(row[0] for row in rows)
    minimum_worst_at_minimum_expected = min(
        row[1] for row in rows if row[0] == minimum_expected
    )
    return True, minimum_expected, minimum_worst_at_minimum_expected


def candidate_lower_bound(
    task: object,
    profile: response.ResponseCostProfile,
    allowed: int,
    canonical: int,
    query: int,
) -> CandidateBound:
    children = response.partition(task, allowed, query)
    next_remaining = canonical & ~(1 << query)
    expected = response.immediate_expected_cost(profile, allowed, query)
    branch_worst: list[int] = []
    possible = True
    for child in children:
        child_possible, child_expected, child_worst = (
            impure_child_first_step_bound(
                task, profile, child, next_remaining
            )
        )
        possible = possible and child_possible
        expected += child_expected
        branch_worst.append(
            response.cell_cost(profile, query, child) + child_worst
        )
    return CandidateBound(
        query=query,
        children=children,
        full_diagnosis_possible=possible,
        expected_cost_lower_bound=expected,
        worst_cost_lower_bound=max(branch_worst),
    )


def bound_order(row: CandidateBound) -> tuple[int, int, int, int]:
    return (
        int(not row.full_diagnosis_possible),
        row.expected_cost_lower_bound,
        row.worst_cost_lower_bound,
        row.query,
    )


def incumbent_dominates_bound(
    incumbent: response.Plan,
    state_mass: int,
    row: CandidateBound,
) -> tuple[bool, bool]:
    """Return (prune, impossible-full-diagnosis-prune).

    Cost bounds are used only after an incumbent diagnoses the full state mass.
    A candidate can then beat it only by also diagnosing the full mass and
    improving the lexicographic cost/query objective.
    """
    if incumbent.diagnosed_mass != state_mass:
        return False, False
    if not row.full_diagnosis_possible:
        return True, True
    if row.expected_cost_lower_bound > incumbent.expected_cost_numerator:
        return True, False
    if row.expected_cost_lower_bound < incumbent.expected_cost_numerator:
        return False, False
    if row.worst_cost_lower_bound > incumbent.worst_cost:
        return True, False
    if row.worst_cost_lower_bound < incumbent.worst_cost:
        return False, False
    incumbent_query = (
        incumbent.query if incumbent.query is not None else 10**9
    )
    return row.query > incumbent_query, False


class LowerBoundParetoPlanner:
    def __init__(
        self,
        task: object,
        profile: response.ResponseCostProfile,
        budget: int,
    ) -> None:
        self.task = task
        self.profile = profile
        self.budget = budget
        self.memo: dict[tuple[int, int], response.Plan] = {}
        self.calls = 0
        self.query_expansions = 0
        self.memo_hits = 0
        self.raw_queries_considered = 0
        self.representative_queries_considered = 0
        self.dominated_queries_removed = 0
        self.bound_evaluations = 0
        self.bound_pruned_queries = 0
        self.impossible_full_diagnosis_prunes = 0

    def canonical(self, allowed: int, remaining: int) -> int:
        raw = remaining.bit_count()
        canonical = response.pareto_mask(
            self.task, self.profile, allowed, remaining
        )
        kept = canonical.bit_count()
        self.raw_queries_considered += raw
        self.representative_queries_considered += kept
        self.dominated_queries_removed += max(0, raw - kept)
        return canonical

    def solve(self, allowed: int, remaining: int) -> response.Plan:
        self.calls += 1
        canonical = self.canonical(allowed, remaining)
        key = allowed, canonical
        cached = self.memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            return cached

        mass = response.subset_mass(self.profile, allowed)
        if state.base.pure_label(self.task, allowed) is not None:
            answer = response.Plan(mass, 0, 0, None)
            self.memo[key] = answer
            return answer

        bounds: list[CandidateBound] = []
        pending = canonical
        while pending:
            bit = pending & -pending
            query = bit.bit_length() - 1
            pending ^= bit
            bounds.append(candidate_lower_bound(
                self.task, self.profile, allowed, canonical, query
            ))
            self.bound_evaluations += 1
        bounds.sort(key=bound_order)

        incumbent: response.Plan | None = None
        for row in bounds:
            if incumbent is not None:
                prune, impossible = incumbent_dominates_bound(
                    incumbent, mass, row
                )
                if prune:
                    self.bound_pruned_queries += 1
                    self.impossible_full_diagnosis_prunes += int(impossible)
                    continue

            self.query_expansions += 1
            if self.query_expansions > self.budget:
                raise response.BudgetExceeded(
                    "lower-bound Pareto response-cost budget exceeded"
                )
            next_remaining = canonical & ~(1 << row.query)
            child_plans = [
                self.solve(child, next_remaining)
                for child in row.children
            ]
            candidate = response.Plan(
                diagnosed_mass=sum(
                    child.diagnosed_mass for child in child_plans
                ),
                expected_cost_numerator=(
                    response.immediate_expected_cost(
                        self.profile, allowed, row.query
                    )
                    + sum(
                        child.expected_cost_numerator
                        for child in child_plans
                    )
                ),
                worst_cost=max(
                    response.cell_cost(
                        self.profile, row.query, child_mask
                    )
                    + child_plan.worst_cost
                    for child_mask, child_plan
                    in zip(row.children, child_plans)
                ),
                query=row.query,
            )
            if (
                incumbent is None
                or response.plan_score(candidate)
                > response.plan_score(incumbent)
            ):
                incumbent = candidate

        answer = (
            incumbent
            if incumbent is not None
            else response.Plan(0, 0, 0, None)
        )
        self.memo[key] = answer
        return answer

    def result(self, allowed: int, remaining: int) -> BoundResult:
        answer = self.solve(allowed, remaining)
        return BoundResult(answer, BoundStats(
            calls=self.calls,
            memo_entries=len(self.memo),
            query_expansions=self.query_expansions,
            memo_hits=self.memo_hits,
            raw_queries_considered=self.raw_queries_considered,
            representative_queries_considered=(
                self.representative_queries_considered
            ),
            dominated_queries_removed=self.dominated_queries_removed,
            bound_evaluations=self.bound_evaluations,
            bound_pruned_queries=self.bound_pruned_queries,
            impossible_full_diagnosis_prunes=(
                self.impossible_full_diagnosis_prunes
            ),
        ))


def exact_plan_tuple(plan: response.Plan) -> tuple[int, int, int, int | None]:
    return (
        plan.diagnosed_mass,
        plan.expected_cost_numerator,
        plan.worst_cost,
        plan.query,
    )


def random_certificate() -> dict[str, object]:
    rows = []
    mismatches = []
    for seed in range(
        RANDOM_SEED_START,
        RANDOM_SEED_START + RANDOM_CERTIFICATE_TASKS,
    ):
        task, profile = response.random_task_and_profile(seed)
        remaining = (1 << task.query_count) - 1
        current = response.ParetoPlanner(
            task, profile, BUDGET
        ).result(task.full_mask, remaining)
        bounded = LowerBoundParetoPlanner(
            task, profile, BUDGET
        ).result(task.full_mask, remaining)
        matched = exact_plan_tuple(current.plan) == exact_plan_tuple(
            bounded.plan
        )
        if not matched:
            mismatches.append({
                "seed": seed,
                "current": exact_plan_tuple(current.plan),
                "bounded": exact_plan_tuple(bounded.plan),
            })
        rows.append({
            "seed": seed,
            "matched": matched,
            "current_expansions": current.stats.query_expansions,
            "bounded_expansions": bounded.stats.query_expansions,
            "bound_pruned_queries": bounded.stats.bound_pruned_queries,
        })
    return {
        "task_count": len(rows),
        "exact_match_count": sum(int(row["matched"]) for row in rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "tasks_with_bound_pruning": sum(
            int(row["bound_pruned_queries"] > 0) for row in rows
        ),
        "passed": (
            len(rows) == RANDOM_CERTIFICATE_TASKS
            and not mismatches
        ),
        "rows": rows,
    }


def load_v64_tasks() -> tuple[
    list[tuple[object, list[tuple[int, int, int]]]],
    list[dict[str, object]],
]:
    manifest = json.loads(clean_v64.MANIFEST.read_text(encoding="utf-8"))
    if manifest["lock_digest"] != clean_v64.LOCK_DIGEST:
        raise RuntimeError("v0.64 lock digest changed")
    tasks = []
    summaries = []
    for dataset in manifest["datasets"]:
        payload = clean_v64.download(str(dataset["url"]))
        if (
            hashlib.sha256(payload).hexdigest() != dataset["sha256"]
            or len(payload) != int(dataset["bytes"])
        ):
            raise RuntimeError(f"archive mismatch for {dataset['name']}")
        records = clean_v64.parse_records(str(dataset["name"]), payload)
        task, summary = external.task_from_records(
            str(dataset["name"]), records
        )
        selected, selection = conditioned.select_states(task)
        summary.update(selection)
        summary["task"] = task.name
        summary["uci_id"] = dataset["uci_id"]
        summaries.append(summary)
        tasks.append((task, selected))
    return tasks, summaries


def evaluate_opened_v64() -> dict[str, object]:
    tasks, summaries = load_v64_tasks()
    rows = []
    base_states: set[str] = set()
    for task, selected in tasks:
        for allowed, remaining, representatives in selected:
            base_digest = hashlib.sha256(
                f"v64:{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            base_states.add(base_digest)
            for seed in conditioned.PROFILE_SEEDS:
                profile = response.profile_for_task(task, seed)
                current = None
                bounded = None
                plain = None
                current_solved = False
                bounded_solved = False
                plain_solved = False
                try:
                    current = response.ParetoPlanner(
                        task, profile, BUDGET
                    ).result(allowed, remaining)
                    current_solved = True
                except response.BudgetExceeded:
                    pass
                try:
                    bounded = LowerBoundParetoPlanner(
                        task, profile, BUDGET
                    ).result(allowed, remaining)
                    bounded_solved = True
                except response.BudgetExceeded:
                    pass
                if bounded_solved:
                    try:
                        plain = response.PlainPlanner(
                            task, profile, BUDGET
                        ).result(allowed, remaining)
                        plain_solved = True
                    except response.BudgetExceeded:
                        pass

                matched = (
                    current_solved
                    and bounded_solved
                    and exact_plan_tuple(current.plan)
                    == exact_plan_tuple(bounded.plan)
                )
                bounded_expansions = (
                    bounded.stats.query_expansions
                    if bounded is not None else BUDGET + 1
                )
                current_expansions = (
                    current.stats.query_expansions
                    if current is not None else BUDGET + 1
                )
                plain_lower = (
                    plain.stats.query_expansions
                    if plain is not None else BUDGET + 1
                )
                rows.append({
                    "task": task.name,
                    "profile_seed": seed,
                    "base_state_digest": base_digest,
                    "structural_partition_representatives": representatives,
                    "candidate_count": allowed.bit_count(),
                    "raw_remaining_queries": remaining.bit_count(),
                    "current_solved": current_solved,
                    "bounded_solved": bounded_solved,
                    "plain_solved": plain_solved,
                    "matched": matched,
                    "current_plan": (
                        exact_plan_tuple(current.plan)
                        if current is not None else None
                    ),
                    "bounded_plan": (
                        exact_plan_tuple(bounded.plan)
                        if bounded is not None else None
                    ),
                    "current_expansions": current_expansions,
                    "bounded_expansions": bounded_expansions,
                    "plain_expansions_lower_bound": plain_lower,
                    "current_over_bounded_ratio": (
                        current_expansions / max(1, bounded_expansions)
                    ),
                    "plain_over_bounded_ratio": (
                        plain_lower / max(1, bounded_expansions)
                    ),
                    "bound_pruned_queries": (
                        bounded.stats.bound_pruned_queries
                        if bounded is not None else 0
                    ),
                    "impossible_full_diagnosis_prunes": (
                        bounded.stats.impossible_full_diagnosis_prunes
                        if bounded is not None else 0
                    ),
                })

    rows.sort(key=lambda row: (
        str(row["task"]),
        str(row["base_state_digest"]),
        int(row["profile_seed"]),
    ))
    current_solved = [row for row in rows if row["current_solved"]]
    bounded_solved = [row for row in rows if row["bounded_solved"]]
    current_ratios = [
        float(row["current_over_bounded_ratio"])
        for row in bounded_solved
    ]
    plain_ratios = [
        float(row["plain_over_bounded_ratio"])
        for row in bounded_solved
    ]
    sum_current = sum(
        int(row["current_expansions"]) for row in current_solved
    )
    sum_bounded = sum(
        int(row["bounded_expansions"]) for row in bounded_solved
    )
    ladder = {
        str(budget): {
            "bounded_solved": sum(
                int(
                    row["bounded_solved"]
                    and int(row["bounded_expansions"]) <= budget
                )
                for row in rows
            ),
            "plain_solved": sum(
                int(
                    row["plain_solved"]
                    and int(row["plain_expansions_lower_bound"]) <= budget
                )
                for row in rows
            ),
        }
        for budget in BUDGET_LADDER
    }
    return {
        "dataset_summaries": summaries,
        "base_state_count": len(base_states),
        "profiled_state_count": len(rows),
        "current_solved_count": len(current_solved),
        "bounded_solved_count": len(bounded_solved),
        "exact_plan_match_count": sum(int(row["matched"]) for row in rows),
        "mismatch_count": sum(int(not row["matched"]) for row in rows),
        "states_with_bound_pruning": sum(
            int(int(row["bound_pruned_queries"]) > 0)
            for row in rows
        ),
        "states_with_expansion_regression": sum(
            int(
                int(row["bounded_expansions"])
                > int(row["current_expansions"])
            )
            for row in rows
        ),
        "current_query_expansions": sum_current,
        "bounded_query_expansions": sum_bounded,
        "aggregate_expansion_reduction_fraction": (
            1.0 - (sum_bounded / sum_current)
            if sum_current else 0.0
        ),
        "median_current_over_bounded_ratio": (
            float(np.median(current_ratios))
            if current_ratios else None
        ),
        "median_plain_over_bounded_ratio": (
            float(np.median(plain_ratios))
            if plain_ratios else None
        ),
        "p90_plain_over_bounded_ratio": (
            float(np.quantile(plain_ratios, 0.9))
            if plain_ratios else None
        ),
        "budget_ladder_summary": ladder,
        "rows": rows,
    }


def run() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    parent = json.loads(V64_EVIDENCE.read_text(encoding="utf-8"))
    if preregistration["parent_v64_evidence_digest"] != parent["frozen_external_digest"]:
        raise RuntimeError("v0.64 parent evidence changed")
    if parent["development_gate"]:
        raise RuntimeError("v0.64 parent must remain rejected")

    certificate = random_certificate()
    opened = evaluate_opened_v64()
    ladder = opened["budget_ladder_summary"]
    gate = (
        certificate["passed"]
        and int(opened["base_state_count"]) == 60
        and int(opened["profiled_state_count"]) == 180
        and int(opened["current_solved_count"]) == 180
        and int(opened["bounded_solved_count"]) == 180
        and int(opened["exact_plan_match_count"]) == 180
        and int(opened["mismatch_count"]) == 0
        and int(opened["states_with_bound_pruning"]) >= 30
        and int(opened["states_with_expansion_regression"]) == 0
        and float(opened["aggregate_expansion_reduction_fraction"]) >= 0.10
        and float(opened["median_current_over_bounded_ratio"]) >= 1.10
        and float(opened["median_plain_over_bounded_ratio"]) >= 10.0
        and float(opened["p90_plain_over_bounded_ratio"]) >= 30.0
        and int(ladder["50000"]["bounded_solved"])
        >= int(ladder["50000"]["plain_solved"]) + 20
    )
    result = {
        "status": (
            "pareto_lower_bound_development_pass"
            if gate else "pareto_lower_bound_development_rejected"
        ),
        "development_gate": gate,
        "claim_scope": (
            "Exact one-step full-diagnosis lower-bound pruning over the unchanged "
            "response-cost Pareto quotient, evaluated only on opened v0.64 states. "
            "A pass freezes a mechanism for independent compiled reproduction and "
            "a later fresh campaign; it is not external validation."
        ),
        "preregistration": preregistration,
        "parent_v64_status": parent["status"],
        "parent_v64_frozen_external_digest": parent["frozen_external_digest"],
        "random_certificate": certificate,
        "opened_v64_evaluation": opened,
    }
    result["evidence_digest"] = hashlib.sha256(json.dumps({
        "preregistration": preregistration,
        "certificate_summary": {
            key: certificate[key]
            for key in (
                "task_count", "exact_match_count", "mismatch_count",
                "tasks_with_bound_pruning", "passed",
            )
        },
        "opened_summary": {
            key: opened[key]
            for key in (
                "base_state_count", "profiled_state_count",
                "current_solved_count", "bounded_solved_count",
                "exact_plan_match_count", "mismatch_count",
                "states_with_bound_pruning",
                "states_with_expansion_regression",
                "aggregate_expansion_reduction_fraction",
                "median_current_over_bounded_ratio",
                "median_plain_over_bounded_ratio",
                "p90_plain_over_bounded_ratio",
                "budget_ladder_summary",
            )
        },
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    opened = result["opened_v64_evaluation"]
    print(json.dumps({
        "status": result["status"],
        "certificate_matches": result["random_certificate"]["exact_match_count"],
        "v64_matches": opened["exact_plan_match_count"],
        "states_with_pruning": opened["states_with_bound_pruning"],
        "aggregate_reduction": opened["aggregate_expansion_reduction_fraction"],
        "current_over_bounded_median": opened["median_current_over_bounded_ratio"],
        "plain_over_bounded_median": opened["median_plain_over_bounded_ratio"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
