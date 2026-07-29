from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COUNTERS = (
    "query_expansions",
    "calls",
    "memo_entries",
    "memo_hits",
    "raw_queries_considered",
    "representative_queries_considered",
    "dominated_queries_removed",
    "bound_evaluations",
    "bound_pruned_queries",
    "impossible_full_diagnosis_prunes",
)
LOCK_DIGEST = "9abc52a7e83255498c84b802d432306b5ff15dece032469968b8db3501d0a385"
REGISTRY_DIGEST = "b88fcb352c2f80af8bc89a3a7576b9cd384800b67d1b168534ad26df9985b6c1"
V66_DIGEST = "3b2bb026556ff9f6321ad6a8375854ae46931e64080329c76f86f31d12c0d643"


def validate(reference_path: Path, rust_path: Path, output_path: Path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rust = json.loads(rust_path.read_text(encoding="utf-8"))
    rust_by_digest = {row["digest"]: row for row in rust["rows"]}
    mismatches = []
    exact_matches = 0
    for expected in reference["rows"]:
        digest = expected["state_digest"]
        actual = rust_by_digest.get(digest)
        if actual is None:
            mismatches.append({"digest": digest, "kind": "missing-rust-row"})
            continue
        if bool(actual.get("solved")) != bool(expected["bounded_solved"]):
            mismatches.append({
                "digest": digest,
                "kind": "solved-status",
                "python": expected["bounded_solved"],
                "rust": actual.get("solved"),
            })
            continue
        if not expected["bounded_solved"]:
            exact_matches += 1
            continue
        if actual["plan"] != expected["bounded_plan"]:
            mismatches.append({
                "digest": digest,
                "kind": "plan",
                "python": expected["bounded_plan"],
                "rust": actual["plan"],
            })
            continue
        bad = {
            field: {
                "python": expected["bounded_stats"][field],
                "rust": actual.get(field),
            }
            for field in COUNTERS
            if int(expected["bounded_stats"][field])
            != int(actual.get(field, -1))
        }
        if bad:
            mismatches.append({
                "digest": digest,
                "kind": "counters",
                "fields": bad,
            })
            continue
        exact_matches += 1

    expected_digests = {row["state_digest"] for row in reference["rows"]}
    unexpected = sorted(set(rust_by_digest) - expected_digests)
    if unexpected:
        mismatches.append({
            "kind": "unexpected-rust-rows",
            "digests": unexpected[:20],
        })

    profiled = int(reference["profiled_state_count"])
    bounded_solved = int(reference["bounded_solved_count"])
    both = int(reference["both_plain_bounded_count"])
    bounded_only = int(reference["bounded_only_count"])
    ladder = reference["budget_ladder_summary"]
    median = reference["expansion_ratio_median"]
    p90 = reference["expansion_ratio_p90"]
    gate = (
        reference["archive_lock_digest"] == LOCK_DIGEST
        and reference["repository_registry_digest"] == REGISTRY_DIGEST
        and reference["parent_v66_evidence_digest"] == V66_DIGEST
        and reference["archive_verification"]["all_hashes_match"]
        and int(reference["contributing_dataset_count"]) >= 5
        and int(reference["base_state_count"]) >= 50
        and profiled >= 150
        and bounded_solved >= int(0.9 * profiled)
        and both >= 40
        and int(reference["plain_bounded_objective_mismatch_count"]) == 0
        and bounded_only >= 25
        and int(reference["current_bounded_plan_mismatch_count"]) == 0
        and int(reference["bounded_expansion_regression_count"]) == 0
        and int(reference["states_with_lower_bound_pruning"]) >= 30
        and float(reference["aggregate_bounded_reduction_fraction"]) >= 0.10
        and not mismatches
        and exact_matches == profiled
        and int(reference["dominated_queries_removed"]) >= 1_000
        and int(reference["root_incomparable_classes"]) > 0
        and median is not None and float(median) >= 10.0
        and p90 is not None and float(p90) >= 30.0
        and int(ladder["50000"]["bounded_solved"])
        >= int(ladder["50000"]["plain_solved"]) + 20
    )
    result = {
        "status": (
            "clean_lower_bound_external_blind_pass"
            if gate else "clean_lower_bound_external_blind_rejected"
        ),
        "development_gate": gate,
        "claim_scope": (
            "Repository-audited unused official UCI archives, committed hash lock, "
            "unchanged solver-independent state generator, exact plain/current/bounded "
            "Python comparisons and independently implemented Rust lower-bound replay. "
            "A pass is strong clean external algorithmic evidence, not outside-human "
            "reproduction, publication novelty, peer review or a world-level claim."
        ),
        "archive_lock_digest": reference["archive_lock_digest"],
        "repository_registry_digest": reference["repository_registry_digest"],
        "parent_v66_evidence_digest": reference["parent_v66_evidence_digest"],
        "frozen_external_digest": reference["frozen_external_digest"],
        "contributing_dataset_count": reference["contributing_dataset_count"],
        "base_state_count": reference["base_state_count"],
        "profiled_state_count": profiled,
        "bounded_solved_count": bounded_solved,
        "both_plain_bounded_count": both,
        "bounded_only_count": bounded_only,
        "plain_bounded_objective_mismatch_count": reference["plain_bounded_objective_mismatch_count"],
        "both_current_bounded_count": reference["both_current_bounded_count"],
        "current_bounded_plan_mismatch_count": reference["current_bounded_plan_mismatch_count"],
        "bounded_expansion_regression_count": reference["bounded_expansion_regression_count"],
        "states_with_lower_bound_pruning": reference["states_with_lower_bound_pruning"],
        "current_query_expansions": reference["current_query_expansions"],
        "bounded_query_expansions": reference["bounded_query_expansions"],
        "aggregate_bounded_reduction_fraction": reference["aggregate_bounded_reduction_fraction"],
        "dominated_queries_removed": reference["dominated_queries_removed"],
        "root_incomparable_classes": reference["root_incomparable_classes"],
        "expansion_ratio_median": median,
        "expansion_ratio_p90": p90,
        "budget_ladder_summary": ladder,
        "rust_total_milliseconds": rust.get("total_milliseconds"),
        "rust_exact_match_count": exact_matches,
        "rust_mismatch_count": len(mismatches),
        "rust_mismatches": mismatches,
        "dataset_summaries": reference["dataset_summaries"],
        "archive_verification": reference["archive_verification"],
        "protocol": reference["protocol"],
    }
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.reference, args.rust, args.output)
    print(json.dumps({
        "status": result["status"],
        "gate": result["development_gate"],
        "datasets": result["contributing_dataset_count"],
        "base_states": result["base_state_count"],
        "profiled_states": result["profiled_state_count"],
        "bounded_solved": result["bounded_solved_count"],
        "plain_solved": result["both_plain_bounded_count"],
        "bounded_only": result["bounded_only_count"],
        "median": result["expansion_ratio_median"],
        "rust_mismatches": result["rust_mismatch_count"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
