from __future__ import annotations

import argparse
import json
from pathlib import Path


STAT_KEYS = (
    "calls",
    "memo_entries",
    "query_expansions",
    "memo_hits",
    "raw_queries_considered",
    "representative_queries_considered",
)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_digest(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["digest"]: row for row in report["rows"]}


def run(
    reference_path: Path,
    manifest_path: Path,
    local_path: Path,
    plain_path: Path,
) -> dict[str, object]:
    reference = load(reference_path)
    manifest = load(manifest_path)
    local = load(local_path)
    plain = load(plain_path)
    reference_rows = {
        row["weighted_state_digest"]: row
        for row in reference["rows"]
    }
    local_rows = rows_by_digest(local)
    plain_rows = rows_by_digest(plain)
    digests = set(reference_rows)
    digest_sets_match = (
        digests == set(manifest["state_digests"])
        == set(local_rows)
        == set(plain_rows)
    )

    local_plan_matches = 0
    local_stat_matches = 0
    plain_solved_matches = 0
    plain_plan_matches = 0
    plain_stat_matches = 0
    mismatches = []
    for digest in sorted(digests):
        expected = reference_rows[digest]
        compiled_local = local_rows[digest]
        compiled_plain = plain_rows[digest]
        expected_local_plan = expected["local_plan"]
        if (
            compiled_local["solved"]
            and compiled_local["plan"] == expected_local_plan
        ):
            local_plan_matches += 1
        else:
            mismatches.append({
                "digest": digest,
                "kind": "local-plan",
                "expected": expected_local_plan,
                "actual": compiled_local["plan"],
                "solved": compiled_local["solved"],
            })
        expected_local_stats = expected["local_stats"]
        if all(
            compiled_local[key] == expected_local_stats[key]
            for key in STAT_KEYS
        ):
            local_stat_matches += 1
        else:
            mismatches.append({
                "digest": digest,
                "kind": "local-stats",
                "expected": {
                    key: expected_local_stats[key] for key in STAT_KEYS
                },
                "actual": {
                    key: compiled_local[key] for key in STAT_KEYS
                },
            })

        expected_plain_solved = bool(expected["plain_solved"])
        if bool(compiled_plain["solved"]) == expected_plain_solved:
            plain_solved_matches += 1
        else:
            mismatches.append({
                "digest": digest,
                "kind": "plain-solved",
                "expected": expected_plain_solved,
                "actual": compiled_plain["solved"],
            })
        if expected_plain_solved:
            if compiled_plain["plan"] == expected["plain_plan"]:
                plain_plan_matches += 1
            else:
                mismatches.append({
                    "digest": digest,
                    "kind": "plain-plan",
                    "expected": expected["plain_plan"],
                    "actual": compiled_plain["plan"],
                })
            expected_plain_stats = expected["plain_stats"]
            if all(
                compiled_plain[key] == expected_plain_stats[key]
                for key in STAT_KEYS
            ):
                plain_stat_matches += 1
            else:
                mismatches.append({
                    "digest": digest,
                    "kind": "plain-stats",
                    "expected": {
                        key: expected_plain_stats[key]
                        for key in STAT_KEYS
                    },
                    "actual": {
                        key: compiled_plain[key] for key in STAT_KEYS
                    },
                })

    speedup = (
        float(plain["total_milliseconds"])
        / max(1e-9, float(local["total_milliseconds"]))
    )
    gate = (
        reference["development_gate"]
        and reference["frozen_weighted_digest"]
        == "cfdbb2d072d426d7b4a684986af23d573976809fc1fa0949e4edb5579b6012b0"
        and manifest["state_count"] == 195
        and manifest["base_state_count"] == 65
        and manifest["parent_weighted_digest"]
        == reference["frozen_weighted_digest"]
        and manifest["archive_verification"]["v39"]["all_hashes_match"]
        and manifest["archive_verification"]["v41"]["all_hashes_match"]
        and digest_sets_match
        and local["state_count"] == 195
        and local["solved_count"] == 195
        and plain["state_count"] == 195
        and plain["solved_count"] == 141
        and local_plan_matches == 195
        and local_stat_matches == 195
        and plain_solved_matches == 195
        and plain_plan_matches == 141
        and plain_stat_matches == 141
        and not mismatches
        and speedup >= 2.0
        and int(local["peak_rss_kb"]) <= 512000
        and int(plain["peak_rss_kb"]) <= 1024000
    )
    return {
        "status": (
            "compiled_reproduction_passed" if gate else "not_yet"
        ),
        "compiled_reproduction_gate": gate,
        "state_count": len(digests),
        "digest_sets_match": digest_sets_match,
        "local_solved_count": local["solved_count"],
        "plain_solved_count": plain["solved_count"],
        "local_plan_match_count": local_plan_matches,
        "local_stat_match_count": local_stat_matches,
        "plain_solved_match_count": plain_solved_matches,
        "plain_plan_match_count": plain_plan_matches,
        "plain_stat_match_count": plain_stat_matches,
        "compiled_local_total_milliseconds": local[
            "total_milliseconds"
        ],
        "compiled_plain_total_milliseconds": plain[
            "total_milliseconds"
        ],
        "compiled_plain_over_local_speedup": speedup,
        "compiled_local_peak_rss_kb": local["peak_rss_kb"],
        "compiled_plain_peak_rss_kb": plain["peak_rss_kb"],
        "compiled_local_query_expansions": local[
            "total_query_expansions"
        ],
        "compiled_plain_query_expansions": plain[
            "total_query_expansions"
        ],
        "input_sha256": manifest["input_sha256"],
        "parent_weighted_digest": manifest["parent_weighted_digest"],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "claim_scope": (
            "A standalone C++20 exact solver independently reproduces the "
            "Python v0.48 weighted local and plain objectives and operation "
            "counts on all frozen states. The dataset/task exporter is Python, "
            "but the compiled planner, canonicalisation, memoisation, objective "
            "ordering and budget accounting do not call or embed the Python "
            "solver."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.reference, args.manifest, args.local, args.plain
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
