from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import exhaustive_tree_certificate_v62 as certificate


def scalar_collapse_counterexample() -> dict[str, object]:
    costs = ((1, 9), (9, 1))
    first_prior = (9, 1)
    second_prior = (1, 9)
    first_weighted = tuple(
        sum(mass * cost for mass, cost in zip(first_prior, row))
        for row in costs
    )
    second_weighted = tuple(
        sum(mass * cost for mass, cost in zip(second_prior, row))
        for row in costs
    )
    first_optimum = min(range(2), key=lambda query: (first_weighted[query], query))
    second_optimum = min(range(2), key=lambda query: (second_weighted[query], query))
    return {
        "passed": (
            first_optimum == 0
            and second_optimum == 1
            and not certificate.vector_dominates(costs[0], costs[1])
            and not certificate.vector_dominates(costs[1], costs[0])
        ),
        "cost_vectors": costs,
        "first_prior": first_prior,
        "first_weighted_costs": first_weighted,
        "first_optimal_query": first_optimum,
        "second_prior": second_prior,
        "second_weighted_costs": second_weighted,
        "second_optimal_query": second_optimum,
        "meaning": "Equivalent tests with incomparable response-cost vectors require different representatives under different priors, so one scalar representative is unsound.",
    }


def run() -> dict[str, object]:
    certificate.scalar_collapse_counterexample = scalar_collapse_counterexample
    return certificate.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "cases": report["case_count"],
        "matches": report["frontier_match_count"],
        "reductions": report["cases_with_root_reduction"],
    }, indent=2))
    if not report["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
