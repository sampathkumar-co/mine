from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cleanroom_path: Path, reference_path: Path) -> dict[str, object]:
    clean = load(cleanroom_path)
    reference = load(reference_path)
    clean_rows = {
        (row["task"], row["state_digest"]): row
        for row in clean["rows"]
    }
    reference_rows = {
        (row["task"], row["state_digest"]): row
        for row in reference["rows"]
    }
    keys_match = clean_rows.keys() == reference_rows.keys()
    comparisons = []
    for key in sorted(clean_rows.keys() & reference_rows.keys()):
        clean_row = clean_rows[key]
        ref_row = reference_rows[key]
        exact_plan_match = list(clean_row["exact_plan"]) == list(
            ref_row["quotient_plan"]
        )
        greedy_plan_match = list(clean_row["greedy_plan"]) == list(
            ref_row["greedy_plan"]
        )
        saving_match = clean_row["total_query_saving"] == ref_row[
            "total_query_saving"
        ]
        expansion_match = clean_row["stats"]["expanded_tests"] == ref_row[
            "quotient_stats"
        ]["query_expansions"]
        comparisons.append({
            "task": key[0],
            "state_digest": key[1],
            "exact_plan_match": exact_plan_match,
            "greedy_plan_match": greedy_plan_match,
            "saving_match": saving_match,
            "expansion_match": expansion_match,
        })
    exact_matches = sum(int(row["exact_plan_match"]) for row in comparisons)
    greedy_matches = sum(int(row["greedy_plan_match"]) for row in comparisons)
    saving_matches = sum(int(row["saving_match"]) for row in comparisons)
    expansion_matches = sum(int(row["expansion_match"]) for row in comparisons)
    state_count = len(comparisons)
    aggregate_match = (
        clean["state_count"] == reference["frontier_state_count"]
        and clean["strict_exact_gain_count"]
        == reference["strict_exact_gain_count"]
        and clean["aggregate_total_query_saving_vs_greedy"]
        == reference["aggregate_total_query_saving_vs_greedy"]
        and clean["budget_ladder_exact_solved"]["50000"]
        == reference["budget_ladder_results"]["50000"][
            "quotient_solved"
        ]
    )
    gate = (
        clean["cleanroom_gate"]
        and reference["development_gate"]
        and keys_match
        and state_count == 65
        and exact_matches == state_count
        and greedy_matches == state_count
        and saving_matches == state_count
        and expansion_matches == state_count
        and aggregate_match
    )
    return {
        "status": (
            "independent_reproduction_passed"
            if gate else "independent_reproduction_failed"
        ),
        "independent_reproduction_gate": gate,
        "state_keys_match": keys_match,
        "state_count": state_count,
        "exact_plan_match_count": exact_matches,
        "greedy_plan_match_count": greedy_matches,
        "query_saving_match_count": saving_matches,
        "expansion_count_match_count": expansion_matches,
        "aggregate_summary_match": aggregate_match,
        "cleanroom_solver_independence": clean["solver_independence"],
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanroom", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cleanroom, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "states": report["state_count"],
        "exact_matches": report["exact_plan_match_count"],
        "expansion_matches": report["expansion_count_match_count"],
    }, indent=2))
    if not report["independent_reproduction_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
