from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import average_odt_frontier_v44 as frontier
from . import exact_quotient_certificate_v42 as corpus
from . import response_cost_export_v57 as export
from . import response_cost_pareto_v56 as response


SCHEMA = "mini-origin-response-cost-replication-output-v1"
OUTPUT_FIELDS = (
    "digest",
    "solved",
    "plan",
    "query_expansions",
    "calls",
    "memo_entries",
    "memo_hits",
    "raw_queries_considered",
    "representative_queries_considered",
    "dominated_queries_removed",
)


def canonical_bytes(rows: list[dict[str, object]]) -> bytes:
    normalized = [
        {field: row[field] for field in OUTPUT_FIELDS}
        for row in sorted(rows, key=lambda item: str(item["digest"]))
    ]
    return json.dumps(
        {"schema": SCHEMA, "rows": normalized},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def expected_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    report = response.run()
    reference_rows = report["rows"]
    tasks, archive_verification = corpus.load_all_opened_tasks()
    digests: list[str] = []
    for task in tasks:
        selected, _ = frontier.select_frontier_states(task)
        for allowed, remaining, _ in selected:
            for seed in response.PROFILE_SEEDS:
                digests.append(
                    str(export.compact_state(task, allowed, remaining, seed)["digest"])
                )
    if len(digests) != len(reference_rows):
        raise RuntimeError("reference/export state count mismatch")

    rows: list[dict[str, object]] = []
    for digest, expected in zip(digests, reference_rows):
        if not expected["pareto_solved"]:
            raise RuntimeError(f"frozen quotient did not solve state {digest}")
        stats = expected["pareto_stats"]
        rows.append({
            "digest": digest,
            "solved": True,
            "plan": list(expected["pareto_plan"]),
            "query_expansions": int(stats["query_expansions"]),
            "calls": int(stats["calls"]),
            "memo_entries": int(stats["memo_entries"]),
            "memo_hits": int(stats["memo_hits"]),
            "raw_queries_considered": int(stats["raw_queries_considered"]),
            "representative_queries_considered": int(
                stats["representative_queries_considered"]
            ),
            "dominated_queries_removed": int(stats["dominated_queries_removed"]),
        })
    return rows, {
        "frozen_response_cost_digest": report["frozen_response_cost_digest"],
        "archive_verification": archive_verification,
    }


def build(
    states_path: Path,
    input_manifest_path: Path,
    challenge_path: Path,
) -> dict[str, object]:
    input_manifest = export.run(states_path, input_manifest_path)
    rows, reference = expected_rows()
    commitment = hashlib.sha256(canonical_bytes(rows)).hexdigest()
    sorted_digests = sorted(str(row["digest"]) for row in rows)
    if sorted_digests != sorted(str(value) for value in input_manifest["state_digests"]):
        raise RuntimeError("challenge/reference digest set differs from public input")

    challenge = {
        "status": "external_replication_challenge_v76_frozen",
        "schema": SCHEMA,
        "input_format": input_manifest["format"],
        "state_count": len(rows),
        "base_state_count": input_manifest["base_state_count"],
        "input_sha256": input_manifest["input_sha256"],
        "state_digests": sorted_digests,
        "required_output_fields": list(OUTPUT_FIELDS),
        "expected_canonical_output_sha256": commitment,
        "canonicalization": (
            "Keep exactly required_output_fields, sort rows by digest, then encode "
            "{'schema': schema, 'rows': rows} as UTF-8 JSON with sort_keys=true "
            "and separators=(',', ':')."
        ),
        "objective": (
            "For every compact state, reproduce the exact v0.56 descendant-local "
            "response-cost Pareto quotient plan metrics and all seven locked "
            "operation counters."
        ),
        "independence_requirement": (
            "A qualifying outside reproduction must provide separately written "
            "solver source and must not import or call Mini-ORIGIN's Python, C++ "
            "or Rust planners. Reading the mathematical specification and input "
            "format is permitted."
        ),
        "reference_commitments": reference,
        "claim_boundary": (
            "Publishing this kit is not external reproduction. The commitment is "
            "evidence only after an outside party supplies a matching output and "
            "their independent source is reviewed."
        ),
    }
    challenge["challenge_digest"] = hashlib.sha256(
        json.dumps(challenge, sort_keys=True).encode("utf-8")
    ).hexdigest()
    challenge_path.parent.mkdir(parents=True, exist_ok=True)
    challenge_path.write_text(json.dumps(challenge, indent=2), encoding="utf-8")
    return challenge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    args = parser.parse_args()
    challenge = build(args.states, args.input_manifest, args.challenge)
    print(json.dumps({
        "status": challenge["status"],
        "states": challenge["state_count"],
        "input_sha256": challenge["input_sha256"],
        "expected_output_sha256": challenge["expected_canonical_output_sha256"],
        "challenge_digest": challenge["challenge_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
