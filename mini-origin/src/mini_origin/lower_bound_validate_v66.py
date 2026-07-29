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
PARENT_DIGEST = "3a1ef2c86c07600a591a900caf920456acecacb158b8faeb8bdb4c6d97d0550a"


def validate(
    manifest_path: Path,
    reference_path: Path,
    rust_path: Path,
    output_path: Path,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rust = json.loads(rust_path.read_text(encoding="utf-8"))

    if manifest["parent_v65_evidence_digest"] != PARENT_DIGEST:
        raise RuntimeError("unexpected manifest parent digest")
    if reference["parent_v65_evidence_digest"] != PARENT_DIGEST:
        raise RuntimeError("unexpected reference parent digest")
    if reference["manifest_digest"] != manifest["manifest_digest"]:
        raise RuntimeError("manifest/reference digest mismatch")

    rust_by_digest = {row["digest"]: row for row in rust["rows"]}
    mismatches = []
    exact_matches = 0
    for expected in reference["rows"]:
        digest = expected["digest"]
        actual = rust_by_digest.get(digest)
        if actual is None:
            mismatches.append({"digest": digest, "kind": "missing-rust-row"})
            continue
        if not actual.get("solved", False):
            mismatches.append({"digest": digest, "kind": "rust-unsolved"})
            continue
        if actual["plan"] != expected["plan"]:
            mismatches.append({
                "digest": digest,
                "kind": "plan",
                "python": expected["plan"],
                "rust": actual["plan"],
            })
            continue
        bad = {
            field: {
                "python": expected[field],
                "rust": actual.get(field),
            }
            for field in COUNTERS
            if int(expected[field]) != int(actual.get(field, -1))
        }
        if bad:
            mismatches.append({
                "digest": digest,
                "kind": "counters",
                "fields": bad,
            })
            continue
        exact_matches += 1

    expected_digests = {row["digest"] for row in reference["rows"]}
    unexpected = sorted(set(rust_by_digest) - expected_digests)
    if unexpected:
        mismatches.append({
            "kind": "unexpected-rust-rows",
            "digests": unexpected[:20],
        })

    gate = (
        int(manifest["state_count"]) == 180
        and int(reference["state_count"]) == 180
        and len(rust["rows"]) == 180
        and exact_matches == 180
        and not mismatches
    )
    result = {
        "status": (
            "rust_lower_bound_reproduction_pass"
            if gate else "rust_lower_bound_reproduction_rejected"
        ),
        "development_gate": gate,
        "claim_scope": (
            "Independent Rust implementation reproducing the opened-data v0.65 "
            "lower-bound Pareto planner from compact state tables. This is compiled "
            "implementation evidence, not outside-human reproduction or fresh external validation."
        ),
        "parent_v65_evidence_digest": PARENT_DIGEST,
        "manifest_digest": manifest["manifest_digest"],
        "reference_digest": reference["reference_digest"],
        "state_input_sha256": manifest["input_sha256"],
        "state_count": manifest["state_count"],
        "rust_total_milliseconds": rust.get("total_milliseconds"),
        "exact_match_count": exact_matches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "counter_fields": list(COUNTERS),
    }
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.manifest,
        args.reference,
        args.rust,
        args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "states": result["state_count"],
        "matches": result["exact_match_count"],
        "mismatches": result["mismatch_count"],
        "rust_milliseconds": result["rust_total_milliseconds"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
