from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import response_cost_lower_bound_v65 as lower_bound
from . import response_cost_pareto_v56 as response
from . import state_policy_v34 as state


TASK_COUNT = 64
SEED_START = 75_001


def exact_plan(plan: response.Plan) -> tuple[int, int, int, int | None]:
    return (
        plan.diagnosed_mass,
        plan.expected_cost_numerator,
        plan.worst_cost,
        plan.query,
    )


def profile_conforms(task: object, profile: response.ResponseCostProfile) -> bool:
    if not profile.hypothesis_mass or any(mass <= 0 for mass in profile.hypothesis_mass):
        return False
    if len(profile.hypothesis_mass) != task.candidate_count:
        return False
    if len(profile.hypothesis_cost_by_query) != task.query_count:
        return False
    for query, costs in enumerate(profile.hypothesis_cost_by_query):
        if len(costs) != task.candidate_count or any(cost < 0 for cost in costs):
            return False
        by_response: dict[str, int] = {}
        for index in range(task.candidate_count):
            observed = task.rows[index][query]
            cost = int(costs[index])
            previous = by_response.setdefault(observed, cost)
            if previous != cost:
                return False
    return True


def partition_conforms(task: object, allowed: int, query: int) -> bool:
    cells = response.partition(task, allowed, query)
    if not cells:
        return allowed == 0
    union = 0
    for index, cell in enumerate(cells):
        if not cell or cell & ~allowed:
            return False
        if union & cell:
            return False
        union |= cell
        if index and cells[index - 1] >= cell:
            return False
    return union == allowed


def tie_scope_witness() -> dict[str, object]:
    # Query 1 is strictly cheaper on the first of three common cells, so it
    # dominates query 0 at the full state despite having a larger identifier.
    # After the first cell disappears, their vectors become equal and query 0
    # would win a freshly computed local identifier tie. Carrying the ancestor
    # canonical set retains query 1 instead. Metrics are identical; only a
    # recursive node-identifier convention not present in Plan would differ.
    task = state.base.make_task(
        "v75-tie-scope-witness",
        ("lower-id", "ancestor-cheaper"),
        (("x", "a"), ("y", "b"), ("z", "c")),
        ("L0", "L1", "L2"),
    )
    profile = response.ResponseCostProfile(
        hypothesis_mass=(1, 1, 1),
        hypothesis_cost_by_query=(
            (2, 2, 2),
            (1, 2, 2),
        ),
        seed=75_000,
    )
    full_remaining = 0b11
    removed_at_full = response.dominance_map(
        task, profile, task.full_mask, full_remaining
    )
    descendant = 0b110
    fresh_descendant_mask = response.pareto_mask(
        task, profile, descendant, full_remaining
    )
    carried_descendant_mask = response.pareto_mask(
        task,
        profile,
        descendant,
        full_remaining & ~(1 << 0),
    )
    full_vectors = {
        query: response.cost_vector(task, profile, task.full_mask, query)
        for query in range(2)
    }
    descendant_vectors = {
        query: response.cost_vector(task, profile, descendant, query)
        for query in range(2)
    }
    passed = (
        removed_at_full.get(0) == 1
        and fresh_descendant_mask == 0b01
        and carried_descendant_mask == 0b10
        and descendant_vectors[0] == descendant_vectors[1]
    )
    return {
        "passed": passed,
        "removed_at_full": removed_at_full,
        "full_vectors": full_vectors,
        "descendant_vectors": descendant_vectors,
        "fresh_descendant_mask": fresh_descendant_mask,
        "carried_descendant_mask": carried_descendant_mask,
        "meaning": (
            "Ancestor strict dominance can become descendant equality. The "
            "quotient preserves objective metrics and the current state's root "
            "choice, but does not promise a recursively canonical full tree "
            "under node-identifier tie-breaking absent from Plan."
        ),
    }


def run() -> dict[str, object]:
    rows = []
    failures = []
    total_hereditary_comparisons = 0
    total_descendant_checks = 0
    for seed in range(SEED_START, SEED_START + TASK_COUNT):
        task, profile = response.random_task_and_profile(seed)
        profile_ok = profile_conforms(task, profile)
        partition_ok = all(
            partition_conforms(task, allowed, query)
            for allowed in response.descendants(task.full_mask)
            for query in range(task.query_count)
        )
        theorem = response.hereditary_pareto_theorem(task, profile)
        total_hereditary_comparisons += int(theorem["comparisons"])
        total_descendant_checks += int(theorem["descendant_checks"])
        remaining = (1 << task.query_count) - 1
        plain = response.PlainPlanner(
            task, profile, response.BUDGET
        ).result(task.full_mask, remaining)
        quotient = response.ParetoPlanner(
            task, profile, response.BUDGET
        ).result(task.full_mask, remaining)
        plan_match = exact_plan(plain.plan) == exact_plan(quotient.plan)
        passed = profile_ok and partition_ok and theorem["passed"] and plan_match
        row = {
            "seed": seed,
            "profile_conforms": profile_ok,
            "partitions_conform": partition_ok,
            "hereditary_passed": theorem["passed"],
            "hereditary_comparisons": theorem["comparisons"],
            "descendant_checks": theorem["descendant_checks"],
            "plan_match": plan_match,
            "plain_plan": exact_plan(plain.plan),
            "quotient_plan": exact_plan(quotient.plan),
            "plain_expansions": plain.stats.query_expansions,
            "quotient_expansions": quotient.stats.query_expansions,
            "passed": passed,
        }
        rows.append(row)
        if not passed:
            failures.append(row)

    lower = lower_bound.random_certificate()
    tie_scope = tie_scope_witness()
    gate = (
        len(rows) == TASK_COUNT
        and not failures
        and all(row["passed"] for row in rows)
        and lower["passed"]
        and int(lower["task_count"]) == 96
        and int(lower["exact_match_count"]) == 96
        and int(lower["mismatch_count"]) == 0
        and tie_scope["passed"]
    )
    result = {
        "status": (
            "theorem_obligation_audit_v75_pass"
            if gate else "theorem_obligation_audit_v75_rejected"
        ),
        "development_gate": gate,
        "claim_scope": (
            "Executable conformance checks connecting the v0.75 mathematical "
            "statements to the current Python implementations. This is internal "
            "machine evidence, not a proof-assistant derivation, novelty proof, "
            "external reproduction, peer review, or world-level claim."
        ),
        "task_count": len(rows),
        "task_pass_count": sum(int(row["passed"]) for row in rows),
        "failure_count": len(failures),
        "failures": failures[:20],
        "total_hereditary_comparisons": total_hereditary_comparisons,
        "total_descendant_checks": total_descendant_checks,
        "lower_bound_certificate": {
            key: lower[key]
            for key in (
                "task_count",
                "exact_match_count",
                "mismatch_count",
                "tasks_with_bound_pruning",
                "passed",
            )
        },
        "tie_scope_witness": tie_scope,
        "rows": rows,
    }
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "tasks": report["task_count"],
        "task_passes": report["task_pass_count"],
        "hereditary_comparisons": report["total_hereditary_comparisons"],
        "descendant_checks": report["total_descendant_checks"],
        "lower_bound_matches": report["lower_bound_certificate"]["exact_match_count"],
        "tie_scope": report["tie_scope_witness"]["passed"],
    }, indent=2))
    if not report["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
