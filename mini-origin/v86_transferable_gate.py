#!/usr/bin/env python3
"""Frozen source-agnostic acceptance gate for Mini-ORIGIN v0.86.

This module is intentionally data-source independent. It copies every numeric and
exactness threshold from the preregistered v0.82 gate, while replacing only the
PMLB-name-specific ``previously_zero_datasets`` clause with the preregistered
v0.86 rule that all seven immutable external tasks contribute at least one base
state.

It does not fetch OpenML metadata or task contents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_V82_THRESHOLDS: dict[str, Any] = {
    "contributing_datasets": 7,
    "minimum_states_from_each_dataset": 6,
    "minimum_base_states": 60,
    "minimum_profiled_states": 180,
    "minimum_bounded_solve_fraction": 0.9,
    "minimum_both_plain_bounded": 40,
    "plain_bounded_objective_mismatches": 0,
    "minimum_bounded_only": 25,
    "current_bounded_plan_mismatches": 0,
    "bounded_expansion_regressions": 0,
    "minimum_states_with_lower_bound_pruning": 30,
    "minimum_aggregate_bounded_reduction_fraction": 0.1,
    "minimum_dominated_queries_removed": 1000,
    "minimum_root_incomparable_classes": 1,
    "minimum_median_plain_bounded_ratio": 10.0,
    "minimum_p90_plain_bounded_ratio": 30.0,
    "minimum_50k_solve_advantage": 20,
    "rust_mismatches": 0,
    "label_independence_mismatches": 0,
}

NONTRANSFERABLE_V82_KEYS = {
    "minimum_states_from_each_previously_zero_dataset",
    "previously_zero_datasets",
}


def verify_v82_lock(v82_campaign: Mapping[str, Any]) -> None:
    """Fail closed if any transferable v0.82 threshold has drifted."""
    locked = v82_campaign.get("locked_gate")
    if not isinstance(locked, Mapping):
        raise ValueError("v0.82 campaign has no locked_gate mapping")

    for key, expected in EXPECTED_V82_THRESHOLDS.items():
        actual = locked.get(key)
        if actual != expected:
            raise ValueError(
                f"v0.82 threshold drift for {key}: expected {expected!r}, got {actual!r}"
            )

    missing_nontransferable = NONTRANSFERABLE_V82_KEYS.difference(locked)
    if missing_nontransferable:
        raise ValueError(
            "v0.82 nontransferable clause is missing keys: "
            + ", ".join(sorted(missing_nontransferable))
        )


def evaluate(summary: Mapping[str, Any], selected_task_ids: Sequence[int]) -> dict[str, Any]:
    """Apply the frozen v0.86 gate and return an auditable decision record."""
    task_ids = [int(task_id) for task_id in selected_task_ids]
    failures: list[str] = []

    if len(task_ids) != 7 or len(set(task_ids)) != 7:
        failures.append("selected_task_ids_must_contain_exactly_seven_unique_ids")

    per_task = summary.get("base_states_by_task", {})
    if not isinstance(per_task, Mapping):
        failures.append("base_states_by_task_must_be_a_mapping")
        per_task = {}

    normalized_per_task = {str(key): int(value) for key, value in per_task.items()}
    for task_id in task_ids:
        if normalized_per_task.get(str(task_id), 0) < 1:
            failures.append(f"task_{task_id}_contributed_zero_base_states")

    checks = {
        "contributing_datasets": (int(summary.get("contributing_datasets", -1)), "==", 7),
        "minimum_states_from_each_dataset": (int(summary.get("minimum_states_from_each_dataset", -1)), ">=", 6),
        "minimum_base_states": (int(summary.get("base_states", -1)), ">=", 60),
        "minimum_profiled_states": (int(summary.get("profiled_states", -1)), ">=", 180),
        "minimum_bounded_solve_fraction": (float(summary.get("bounded_solve_fraction", -1.0)), ">=", 0.9),
        "minimum_both_plain_bounded": (int(summary.get("both_plain_bounded", -1)), ">=", 40),
        "plain_bounded_objective_mismatches": (int(summary.get("plain_bounded_objective_mismatches", -1)), "==", 0),
        "minimum_bounded_only": (int(summary.get("bounded_only", -1)), ">=", 25),
        "current_bounded_plan_mismatches": (int(summary.get("current_bounded_plan_mismatches", -1)), "==", 0),
        "bounded_expansion_regressions": (int(summary.get("bounded_expansion_regressions", -1)), "==", 0),
        "minimum_states_with_lower_bound_pruning": (int(summary.get("states_with_lower_bound_pruning", -1)), ">=", 30),
        "minimum_aggregate_bounded_reduction_fraction": (float(summary.get("aggregate_bounded_reduction_fraction", -1.0)), ">=", 0.1),
        "minimum_dominated_queries_removed": (int(summary.get("dominated_queries_removed", -1)), ">=", 1000),
        "minimum_root_incomparable_classes": (int(summary.get("root_incomparable_classes", -1)), ">=", 1),
        "minimum_median_plain_bounded_ratio": (float(summary.get("median_plain_bounded_ratio", -1.0)), ">=", 10.0),
        "minimum_p90_plain_bounded_ratio": (float(summary.get("p90_plain_bounded_ratio", -1.0)), ">=", 30.0),
        "minimum_50k_solve_advantage": (int(summary.get("solve_advantage_50k", -1)), ">=", 20),
        "rust_mismatches": (int(summary.get("rust_mismatches", -1)), "==", 0),
        "label_independence_mismatches": (int(summary.get("label_independence_mismatches", -1)), "==", 0),
    }

    for name, (actual, operator, threshold) in checks.items():
        passed = actual == threshold if operator == "==" else actual >= threshold
        if not passed:
            failures.append(f"{name}: actual={actual!r} required {operator} {threshold!r}")

    return {
        "campaign": "mini-origin-v86-untouched-external-validation",
        "gate": "frozen-transferable-v82-thresholds-with-preregistered-v86-task-contribution-rule",
        "selected_task_ids": task_ids,
        "passed": not failures,
        "failures": failures,
        "negative_results_preserved": True,
        "claim_boundary": (
            "A pass is first-party untouched external validation only; it is not "
            "independent replication, peer review, or a world-class breakthrough."
        ),
    }


def _self_test() -> None:
    passing = {
        "contributing_datasets": 7,
        "minimum_states_from_each_dataset": 6,
        "base_states": 60,
        "profiled_states": 180,
        "bounded_solve_fraction": 0.9,
        "both_plain_bounded": 40,
        "plain_bounded_objective_mismatches": 0,
        "bounded_only": 25,
        "current_bounded_plan_mismatches": 0,
        "bounded_expansion_regressions": 0,
        "states_with_lower_bound_pruning": 30,
        "aggregate_bounded_reduction_fraction": 0.1,
        "dominated_queries_removed": 1000,
        "root_incomparable_classes": 1,
        "median_plain_bounded_ratio": 10.0,
        "p90_plain_bounded_ratio": 30.0,
        "solve_advantage_50k": 20,
        "rust_mismatches": 0,
        "label_independence_mismatches": 0,
        "base_states_by_task": {str(task_id): 1 for task_id in range(1, 8)},
    }
    assert evaluate(passing, range(1, 8))["passed"] is True

    zero_task = dict(passing)
    zero_task["base_states_by_task"] = dict(passing["base_states_by_task"])
    zero_task["base_states_by_task"]["7"] = 0
    result = evaluate(zero_task, range(1, 8))
    assert result["passed"] is False
    assert "task_7_contributed_zero_base_states" in result["failures"]

    mismatch = dict(passing)
    mismatch["rust_mismatches"] = 1
    assert evaluate(mismatch, range(1, 8))["passed"] is False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v82-campaign", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--selected-task-ids", nargs="*", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        print("v0.86 transferable gate self-test: PASS")
        return 0

    if not args.v82_campaign or not args.summary or not args.selected_task_ids:
        parser.error("evaluation requires --v82-campaign, --summary, and --selected-task-ids")

    v82_campaign = json.loads(args.v82_campaign.read_text(encoding="utf-8"))
    verify_v82_lock(v82_campaign)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = evaluate(summary, args.selected_task_ids)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
