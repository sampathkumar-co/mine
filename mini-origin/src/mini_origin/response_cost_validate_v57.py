from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import average_odt_frontier_v44 as frontier
from . import exact_quotient_certificate_v42 as corpus
from . import response_cost_export_v57 as export
from . import response_cost_pareto_v56 as base


def run(compiled: Path, output: Path) -> dict[str, object]:
    cpp = json.loads(compiled.read_text(encoding="utf-8"))
    report = base.run()
    reference_rows = report["rows"]
    tasks, _ = corpus.load_all_opened_tasks()
    digests: list[str] = []
    for task in tasks:
        selected, _ = frontier.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            for seed in base.PROFILE_SEEDS:
                digests.append(str(export.compact_state(task, allowed, remaining, seed)["digest"]))
    if len(digests) != len(reference_rows):
        raise AssertionError("reference/export state count mismatch")
    reference = {digest: row for digest, row in zip(digests, reference_rows)}
    mismatches: list[dict[str, object]] = []
    for row in cpp["rows"]:
        expected = reference.get(row["digest"])
        if expected is None:
            mismatches.append({"digest": row["digest"], "kind": "unknown-state"})
            continue
        # JSON has no tuple type. Normalize the frozen Python plan to the same
        # three-element list representation before comparing values. This is a
        # validator-only representation fix; no solver or gate input changes.
        expected_plan = list(expected["pareto_plan"])
        fields = {
            "plan": expected_plan,
            "query_expansions": expected["pareto_stats"]["query_expansions"],
            "calls": expected["pareto_stats"]["calls"],
            "memo_entries": expected["pareto_stats"]["memo_entries"],
            "memo_hits": expected["pareto_stats"]["memo_hits"],
            "raw_queries_considered": expected["pareto_stats"]["raw_queries_considered"],
            "representative_queries_considered": expected["pareto_stats"]["representative_queries_considered"],
            "dominated_queries_removed": expected["pareto_stats"]["dominated_queries_removed"],
        }
        for key, value in fields.items():
            if row.get(key) != value:
                mismatches.append({"digest": row["digest"], "kind": key, "expected": value, "actual": row.get(key)})
    gate = len(cpp["rows"]) == 195 and not mismatches and all(row.get("solved") for row in cpp["rows"])
    result = {
        "status": "rust_response_cost_reproduction_pass" if gate else "rejected",
        "development_gate": gate,
        "state_count": len(cpp["rows"]),
        "exact_match_count": len(cpp["rows"]) - len({m["digest"] for m in mismatches}),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "total_milliseconds": cpp.get("total_milliseconds"),
        "frozen_parent_digest": report["frozen_response_cost_digest"],
        "claim_scope": "Independent Rust reproduction of the frozen v0.56 Pareto exact planner. This is internal independent-implementation evidence, not external peer validation or a world-class claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.compiled, args.output)
    print(json.dumps(result, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
