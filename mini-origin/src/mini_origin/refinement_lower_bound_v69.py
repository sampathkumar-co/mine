from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from . import conditioned_cell_frontier_v60 as conditioned
from . import response_cost_lower_bound_v65 as lower
from . import response_cost_pareto_v56 as response


RANDOM_CERTIFICATE_TASKS = 96
RANDOM_SEED_START = 69_001
BUDGET = lower.BUDGET
PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v69-refinement-lower-bound-preregistration.json"
)
LOWER_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v65-pareto-lower-bound-pass.json"
)
REFINEMENT_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v65-refinement-dominance-pass.json"
)


@dataclass(frozen=True)
class RefinementStats:
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
    refinement_queries_removed: int


@dataclass(frozen=True)
class RefinementResult:
    plan: response.Plan
    stats: RefinementStats


def strictly_refines(
    fine: tuple[int, ...],
    coarse: tuple[int, ...],
) -> bool:
    """Return whether the fine partition is a proper refinement of coarse."""
    if fine == coarse or len(fine) <= 1 or len(coarse) <= 1:
        return False
    return all(
        any((fine_cell & coarse_cell) == fine_cell for coarse_cell in coarse)
        for fine_cell in fine
    )


def pointwise_strict_cost_dominates(
    profile: response.ResponseCostProfile,
    allowed: int,
    fine_query: int,
    coarse_query: int,
) -> bool:
    """Require no-greater cost everywhere and strict gain on positive mass."""
    strict = False
    pending = allowed
    while pending:
        bit = pending & -pending
        index = bit.bit_length() - 1
        pending ^= bit
        fine_cost = profile.hypothesis_cost_by_query[fine_query][index]
        coarse_cost = profile.hypothesis_cost_by_query[coarse_query][index]
        if fine_cost > coarse_cost:
            return False
        if (
            fine_cost < coarse_cost
            and profile.hypothesis_mass[index] > 0
        ):
            strict = True
    return strict


def refinement_mask(
    task: object,
    profile: response.ResponseCostProfile,
    allowed: int,
    remaining: int,
) -> tuple[int, int, int]:
    """Apply equivalence Pareto pruning, then strict refinement dominance.

    Returns `(mask, equivalent_removed, refinement_removed)`.  The first pass is
    exactly the frozen v0.56 response-partition Pareto quotient.  The second
    removes a retained coarse test only if another retained test induces a
    strict local partition refinement and is pointwise no more expensive with a
    strict saving on at least one positive-mass active hypothesis.
    """
    base = response.pareto_mask(task, profile, allowed, remaining)
    base_queries: list[int] = []
    pending = base
    while pending:
        bit = pending & -pending
        base_queries.append(bit.bit_length() - 1)
        pending ^= bit

    partitions = {
        query: response.partition(task, allowed, query)
        for query in base_queries
    }
    removed: set[int] = set()
    for coarse_query in base_queries:
        coarse_partition = partitions[coarse_query]
        for fine_query in base_queries:
            if fine_query == coarse_query:
                continue
            if not strictly_refines(
                partitions[fine_query], coarse_partition
            ):
                continue
            if pointwise_strict_cost_dominates(
                profile, allowed, fine_query, coarse_query
            ):
                removed.add(coarse_query)
                break

    refined = base
    for query in removed:
        refined &= ~(1 << query)
    return (
        refined,
        max(0, remaining.bit_count() - base.bit_count()),
        len(removed),
    )


class RefinementLowerBoundPlanner(lower.LowerBoundParetoPlanner):
    def __init__(
        self,
        task: object,
        profile: response.ResponseCostProfile,
        budget: int,
    ) -> None:
        super().__init__(task, profile, budget)
        self.refinement_queries_removed = 0

    def canonical(self, allowed: int, remaining: int) -> int:
        raw = remaining.bit_count()
        canonical, _, refinement_removed = refinement_mask(
            self.task, self.profile, allowed, remaining
        )
        kept = canonical.bit_count()
        self.raw_queries_considered += raw
        self.representative_queries_considered += kept
        self.dominated_queries_removed += max(0, raw - kept)
        self.refinement_queries_removed += refinement_removed
        return canonical

    def result(self, allowed: int, remaining: int) -> RefinementResult:
        answer = self.solve(allowed, remaining)
        return RefinementResult(answer, RefinementStats(
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
            refinement_queries_removed=(
                self.refinement_queries_removed
            ),
        ))


def random_certificate() -> dict[str, object]:
    rows = []
    mismatches = []
    regressions = []
    for seed in range(
        RANDOM_SEED_START,
        RANDOM_SEED_START + RANDOM_CERTIFICATE_TASKS,
    ):
        task, profile = response.random_task_and_profile(seed)
        remaining = (1 << task.query_count) - 1
        current = lower.LowerBoundParetoPlanner(
            task, profile, BUDGET
        ).result(task.full_mask, remaining)
        refined = RefinementLowerBoundPlanner(
            task, profile, BUDGET
        ).result(task.full_mask, remaining)
        matched = lower.exact_plan_tuple(
            current.plan
        ) == lower.exact_plan_tuple(refined.plan)
        regressed = (
            refined.stats.query_expansions
            > current.stats.query_expansions
        )
        if not matched:
            mismatches.append({
                "seed": seed,
                "current": lower.exact_plan_tuple(current.plan),
                "refined": lower.exact_plan_tuple(refined.plan),
            })
        if regressed:
            regressions.append({
                "seed": seed,
                "current_expansions": current.stats.query_expansions,
                "refined_expansions": refined.stats.query_expansions,
            })
        rows.append({
            "seed": seed,
            "matched": matched,
            "regressed": regressed,
            "current_expansions": current.stats.query_expansions,
            "refined_expansions": refined.stats.query_expansions,
            "refinement_queries_removed": (
                refined.stats.refinement_queries_removed
            ),
        })
    return {
        "task_count": len(rows),
        "exact_match_count": sum(int(row["matched"]) for row in rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "expansion_regression_count": len(regressions),
        "expansion_regressions": regressions,
        "tasks_with_refinement_removal": sum(
            int(int(row["refinement_queries_removed"]) > 0)
            for row in rows
        ),
        "refinement_queries_removed": sum(
            int(row["refinement_queries_removed"])
            for row in rows
        ),
        "passed": (
            len(rows) == RANDOM_CERTIFICATE_TASKS
            and not mismatches
            and not regressions
        ),
        "rows": rows,
    }


def evaluate_opened_v64() -> dict[str, object]:
    tasks, summaries = lower.load_v64_tasks()
    rows = []
    base_states: set[str] = set()
    for task, selected in tasks:
        for allowed, remaining, representatives in selected:
            base_digest = hashlib.sha256(
                f"v69:{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            base_states.add(base_digest)
            for seed in conditioned.PROFILE_SEEDS:
                profile = response.profile_for_task(task, seed)
                current = None
                refined = None
                current_solved = False
                refined_solved = False
                try:
                    current = lower.LowerBoundParetoPlanner(
                        task, profile, BUDGET
                    ).result(allowed, remaining)
                    current_solved = True
                except response.BudgetExceeded:
                    pass
                try:
                    refined = RefinementLowerBoundPlanner(
                        task, profile, BUDGET
                    ).result(allowed, remaining)
                    refined_solved = True
                except response.BudgetExceeded:
                    pass

                matched = (
                    current_solved
                    and refined_solved
                    and lower.exact_plan_tuple(current.plan)
                    == lower.exact_plan_tuple(refined.plan)
                )
                current_expansions = (
                    current.stats.query_expansions
                    if current is not None else BUDGET + 1
                )
                refined_expansions = (
                    refined.stats.query_expansions
                    if refined is not None else BUDGET + 1
                )
                ratio = (
                    current_expansions / max(1, refined_expansions)
                    if current_expansions or refined_expansions
                    else 1.0
                )
                rows.append({
                    "task": task.name,
                    "profile_seed": seed,
                    "base_state_digest": base_digest,
                    "structural_partition_representatives": representatives,
                    "candidate_count": allowed.bit_count(),
                    "raw_remaining_queries": remaining.bit_count(),
                    "current_solved": current_solved,
                    "refined_solved": refined_solved,
                    "matched": matched,
                    "current_plan": (
                        lower.exact_plan_tuple(current.plan)
                        if current is not None else None
                    ),
                    "refined_plan": (
                        lower.exact_plan_tuple(refined.plan)
                        if refined is not None else None
                    ),
                    "current_expansions": current_expansions,
                    "refined_expansions": refined_expansions,
                    "current_over_refined_ratio": ratio,
                    "refinement_queries_removed": (
                        refined.stats.refinement_queries_removed
                        if refined is not None else 0
                    ),
                    "total_dominated_queries_removed": (
                        refined.stats.dominated_queries_removed
                        if refined is not None else 0
                    ),
                    "bound_pruned_queries": (
                        refined.stats.bound_pruned_queries
                        if refined is not None else 0
                    ),
                })

    rows.sort(key=lambda row: (
        str(row["task"]),
        str(row["base_state_digest"]),
        int(row["profile_seed"]),
    ))
    current_solved = [row for row in rows if row["current_solved"]]
    refined_solved = [row for row in rows if row["refined_solved"]]
    jointly_solved = [
        row for row in rows
        if row["current_solved"] and row["refined_solved"]
    ]
    ratios = [
        float(row["current_over_refined_ratio"])
        for row in jointly_solved
    ]
    sum_current = sum(
        int(row["current_expansions"])
        for row in jointly_solved
    )
    sum_refined = sum(
        int(row["refined_expansions"])
        for row in jointly_solved
    )
    return {
        "dataset_summaries": summaries,
        "base_state_count": len(base_states),
        "profiled_state_count": len(rows),
        "current_solved_count": len(current_solved),
        "refined_solved_count": len(refined_solved),
        "jointly_solved_count": len(jointly_solved),
        "exact_plan_match_count": sum(
            int(row["matched"]) for row in rows
        ),
        "mismatch_count": sum(
            int(not row["matched"]) for row in rows
        ),
        "expansion_regression_count": sum(
            int(
                int(row["refined_expansions"])
                > int(row["current_expansions"])
            )
            for row in jointly_solved
        ),
        "states_with_refinement_removal": sum(
            int(int(row["refinement_queries_removed"]) > 0)
            for row in rows
        ),
        "refinement_queries_removed": sum(
            int(row["refinement_queries_removed"])
            for row in rows
        ),
        "states_with_1_05x_reduction": sum(
            int(
                int(row["refined_expansions"]) > 0
                and float(row["current_over_refined_ratio"]) >= 1.05
            )
            for row in jointly_solved
        ),
        "current_query_expansions": sum_current,
        "refined_query_expansions": sum_refined,
        "aggregate_expansion_reduction_fraction": (
            1.0 - (sum_refined / sum_current)
            if sum_current else 0.0
        ),
        "median_current_over_refined_ratio": (
            float(np.median(ratios)) if ratios else None
        ),
        "p90_current_over_refined_ratio": (
            float(np.quantile(ratios, 0.9)) if ratios else None
        ),
        "rows": rows,
    }


def run() -> dict[str, object]:
    preregistration = json.loads(
        PREREGISTRATION.read_text(encoding="utf-8")
    )
    lower_evidence = json.loads(
        LOWER_EVIDENCE.read_text(encoding="utf-8")
    )
    refinement_evidence = json.loads(
        REFINEMENT_EVIDENCE.read_text(encoding="utf-8")
    )
    if (
        preregistration["parent_v65_lower_bound_evidence_digest"]
        != lower_evidence["evidence_digest"]
        or not lower_evidence["development_gate"]
    ):
        raise RuntimeError("v0.65 lower-bound evidence changed")
    if (
        preregistration["parent_refinement_certificate_protocol_digest"]
        != refinement_evidence["protocol_digest"]
        or not refinement_evidence["development_gate"]
        or int(refinement_evidence["mismatch_count"]) != 0
    ):
        raise RuntimeError("refinement certificate changed")

    certificate = random_certificate()
    opened = evaluate_opened_v64()
    gate_values = preregistration["locked_gate"]
    gate = (
        certificate["passed"]
        and int(certificate["exact_match_count"])
        == int(gate_values["random_exact_plan_matches"])
        and int(certificate["expansion_regression_count"])
        == int(gate_values["random_expansion_regressions"])
        and int(opened["base_state_count"])
        == int(gate_values["base_states"])
        and int(opened["profiled_state_count"])
        == int(gate_values["profiled_states"])
        and int(opened["current_solved_count"])
        == int(gate_values["current_lower_bound_solves"])
        and int(opened["refined_solved_count"])
        == int(gate_values["refinement_lower_bound_solves"])
        and int(opened["exact_plan_match_count"])
        == int(gate_values["exact_plan_matches"])
        and int(opened["mismatch_count"])
        == int(gate_values["plan_mismatches"])
        and int(opened["expansion_regression_count"])
        == int(gate_values["expansion_regressions"])
        and int(opened["states_with_refinement_removal"])
        >= int(gate_values["minimum_states_with_refinement_removal"])
        and int(opened["refinement_queries_removed"])
        >= int(gate_values["minimum_refinement_queries_removed"])
        and float(opened["aggregate_expansion_reduction_fraction"])
        >= float(gate_values["minimum_aggregate_expansion_reduction_fraction"])
        and int(opened["states_with_1_05x_reduction"])
        >= int(gate_values["minimum_states_with_1_05x_expansion_reduction"])
        and opened["median_current_over_refined_ratio"] is not None
        and float(opened["median_current_over_refined_ratio"])
        >= float(gate_values["minimum_median_current_over_refinement_ratio"])
    )
    result = {
        "status": (
            "refinement_lower_bound_development_pass"
            if gate else "refinement_lower_bound_development_rejected"
        ),
        "development_gate": gate,
        "claim_scope": (
            "Strict descendant-local partition-refinement dominance combined with "
            "the independently reproduced lower-bound Pareto planner, evaluated only "
            "on opened v0.64 states. A pass is development evidence, not a fresh "
            "external result, outside-human reproduction, publication novelty, peer "
            "review, or a world-level claim."
        ),
        "preregistration": preregistration,
        "parent_v65_lower_bound_evidence_digest": (
            lower_evidence["evidence_digest"]
        ),
        "parent_refinement_certificate_protocol_digest": (
            refinement_evidence["protocol_digest"]
        ),
        "random_certificate": certificate,
        "opened_v64_evaluation": opened,
    }
    result["evidence_digest"] = hashlib.sha256(
        json.dumps({
            "preregistration": preregistration,
            "parent_lower": result[
                "parent_v65_lower_bound_evidence_digest"
            ],
            "parent_refinement": result[
                "parent_refinement_certificate_protocol_digest"
            ],
            "certificate_summary": {
                key: certificate[key]
                for key in (
                    "task_count",
                    "exact_match_count",
                    "mismatch_count",
                    "expansion_regression_count",
                    "tasks_with_refinement_removal",
                    "refinement_queries_removed",
                    "passed",
                )
            },
            "opened_summary": {
                key: opened[key]
                for key in (
                    "base_state_count",
                    "profiled_state_count",
                    "current_solved_count",
                    "refined_solved_count",
                    "jointly_solved_count",
                    "exact_plan_match_count",
                    "mismatch_count",
                    "expansion_regression_count",
                    "states_with_refinement_removal",
                    "refinement_queries_removed",
                    "states_with_1_05x_reduction",
                    "current_query_expansions",
                    "refined_query_expansions",
                    "aggregate_expansion_reduction_fraction",
                    "median_current_over_refined_ratio",
                    "p90_current_over_refined_ratio",
                )
            },
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()
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
        "random_matches": result["random_certificate"]["exact_match_count"],
        "v64_matches": opened["exact_plan_match_count"],
        "states_with_refinement": opened["states_with_refinement_removal"],
        "refinement_queries_removed": opened["refinement_queries_removed"],
        "aggregate_reduction": opened[
            "aggregate_expansion_reduction_fraction"
        ],
        "median_ratio": opened["median_current_over_refined_ratio"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
