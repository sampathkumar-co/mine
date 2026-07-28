from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import attainable_envelope_v38 as v38
from . import local_quotient_v40 as v40
from . import safe_portfolio_v37 as v37
from . import state_policy_v34 as v34


def shared_candidate_rows(
    task: object,
    threshold: int,
) -> tuple[v38.AttainableRow, ...]:
    """Reuse one exact cache; the planner is fallback-independent."""
    exact = v40.LocalQuotientPlanner(task)
    rows = []
    for objective in v34.OBJECTIVE_NAMES:
        fallback = v37.FallbackPlanner(task, objective)
        result = v40.evaluate(
            task,
            v40.QuotientPolicy(threshold, objective),
            exact,
            fallback,
        )
        rows.append(v38.AttainableRow(
            objective=objective,
            diagnosed_fraction=result.diagnosed_fraction,
            mean_queries=result.mean_queries,
            worst_queries=result.worst_queries,
            candidates=result.candidates,
            exact_query_uses=result.exact_query_uses,
        ))
    return tuple(rows)


def run() -> dict[str, object]:
    original = v40.candidate_rows
    try:
        v40.candidate_rows = shared_candidate_rows
        return v40.run()
    finally:
        v40.candidate_rows = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "selected_threshold": report["selected_threshold"],
        "profile": report["profile"],
    }, indent=2))


if __name__ == "__main__":
    main()
