from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import refinement_dominance_certificate_v65 as certificate


def corrected_profile(
    partitions: tuple[tuple[int, ...], ...], profile_index: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    size = len(partitions[0])
    masses = tuple(1 + ((index + profile_index) % 2) for index in range(size))
    max_cells = max(len(set(partition)) for partition in partitions)
    rows = []
    for query, partition in enumerate(partitions):
        by_response: dict[int, int] = {}
        for response in sorted(set(partition)):
            if profile_index == 0:
                value = 1
            elif profile_index == 1:
                value = 1 + max_cells - len(set(partition))
            elif profile_index == 2:
                value = 1 + max_cells - len(set(partition)) + ((query + response) % 2)
            else:
                value = 1 + hashlib.sha256(
                    f"v65:{query}:{response}:{partition}".encode("utf-8")
                ).digest()[0] % 3
            by_response[response] = value
        row = tuple(by_response[response] for response in partition)
        for response in set(partition):
            observed = {row[index] for index, value in enumerate(partition) if value == response}
            if len(observed) != 1:
                raise AssertionError("response cell received multiple test costs")
        rows.append(row)
    return masses, tuple(rows)


def corrected_counterexamples() -> dict[str, object]:
    coarse = (0, 0, 1)
    fine = (0, 1, 2)
    allowed = 0b111
    equal_costs = ((1, 1, 1), (1, 1, 1))
    fine_more_expensive = ((1, 1, 1), (2, 2, 2))
    fine_strictly_cheaper = ((2, 2, 2), (1, 1, 1))
    strict_required = not certificate.query_dominates(
        (coarse, fine), equal_costs, allowed, 1, 0
    )[0]
    cost_required = not certificate.query_dominates(
        (coarse, fine), fine_more_expensive, allowed, 1, 0
    )[0]
    positive = certificate.query_dominates(
        (coarse, fine), fine_strictly_cheaper, allowed, 1, 0
    ) == (True, "strict-refinement")
    return {
        "passed": strict_required and cost_required and positive,
        "strict_cost_improvement_required": strict_required,
        "pointwise_no_worse_cost_required": cost_required,
        "positive_refinement_example": positive,
        "meaning": (
            "Strict refinement is removed only with pointwise no-worse cost and a "
            "strict improvement on at least one positive-mass hypothesis. Equal-cost "
            "strict refinement remains to preserve deterministic query tie-breaking."
        ),
    }


def run() -> dict[str, object]:
    certificate.profile = corrected_profile
    certificate.counterexamples = corrected_counterexamples
    report = certificate.run()
    report["profile_conformance"] = "one static cost per query response cell"
    return report


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
        "refinement_cases": report["cases_with_root_refinement_reduction"],
        "refinement_removed": report["root_refinement_queries_removed"],
    }, indent=2))
    if not report["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
