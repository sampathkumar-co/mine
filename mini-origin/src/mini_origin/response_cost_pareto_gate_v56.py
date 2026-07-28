from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import response_cost_pareto_v56 as base


def run() -> dict[str, object]:
    report = base.run()
    rows = report["rows"]
    solved = [row for row in rows if row["pareto_solved"]]
    both = [row for row in solved if row["plain_solved"]]
    pareto_only = [
        row for row in solved if not row["plain_solved"]
    ]
    ratios = [
        float(row["expansion_ratio_lower_bound"])
        for row in solved
    ]
    root_dominated = sum(
        int(row["root_pareto_certificate"][
            "dominated_queries_removed"
        ])
        for row in rows
    )
    operational_removed = sum(
        int(row["pareto_stats"][
            "dominated_queries_removed"
        ])
        for row in solved
    )
    ladder = report["budget_ladder_results"]
    gate = (
        report["incomparable_counterexample"]["passed"]
        and report["theorem_certificate"]["passed"]
        and report["theorem_certificate"][
            "tasks_with_dominance_reduction"
        ] >= 20
        and report["archive_verification"]["v39"][
            "all_hashes_match"
        ]
        and report["archive_verification"]["v41"][
            "all_hashes_match"
        ]
        and report["base_state_count"] == 65
        and report["response_cost_state_count"] == 195
        and len(solved) >= 180
        and len(both) >= 120
        and len(pareto_only) >= 25
        and all(row["matched_if_both"] for row in both)
        and all(
            row["root_pareto_certificate"]["passed"]
            for row in rows
        )
        and root_dominated >= 100
        and report["root_incomparable_pareto_classes"] > 0
        and float(np.median(ratios)) >= 3.0
        and float(np.quantile(ratios, 0.9)) >= 10.0
        and ladder["50000"]["pareto_solved"] >= (
            ladder["50000"]["plain_solved"] + 20
        )
    )
    report["status"] = (
        "response_cost_pareto_candidate"
        if gate else "not_yet"
    )
    report["development_gate"] = gate
    report["pareto_solved_count"] = len(solved)
    report["plain_solved_count"] = len(both)
    report["pareto_only_solved_count"] = len(pareto_only)
    report["both_plan_match_count"] = sum(
        int(row["matched_if_both"]) for row in both
    )
    report["dominated_queries_removed_at_roots"] = (
        root_dominated
    )
    report[
        "operational_queries_removed_including_noninformative"
    ] = operational_removed
    report.pop("dominated_queries_removed", None)
    report["accounting_note"] = (
        "The scientific dominance gate counts only queries explicitly "
        "certified componentwise dominated inside root partition classes. "
        "The separate operational count also includes non-informative tests "
        "discarded during descendant canonicalisation and is not used to "
        "satisfy the Pareto-reduction threshold."
    )
    return report


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
        "counterexample_passed": report[
            "incomparable_counterexample"
        ]["passed"],
        "theorem_passed": report[
            "theorem_certificate"
        ]["passed"],
        "states": report["response_cost_state_count"],
        "pareto_solved": report["pareto_solved_count"],
        "plain_solved": report["plain_solved_count"],
        "pareto_only": report["pareto_only_solved_count"],
        "root_dominated": report[
            "dominated_queries_removed_at_roots"
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
