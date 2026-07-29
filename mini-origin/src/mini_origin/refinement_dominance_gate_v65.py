from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import refinement_dominance_certificate_v65 as certificate


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
    certificate.counterexamples = corrected_counterexamples
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
        "refinement_cases": report["cases_with_root_refinement_reduction"],
        "refinement_removed": report["root_refinement_queries_removed"],
    }, indent=2))
    if not report["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
