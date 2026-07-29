from __future__ import annotations

import hashlib
import json

from mini_origin import external_replication_kit_v76 as kit
from mini_origin import validate_external_replication_v76 as validator


def sample_rows() -> list[dict[str, object]]:
    return [
        {
            "digest": "b",
            "solved": True,
            "plan": [2, 4, 3],
            "query_expansions": 7,
            "calls": 11,
            "memo_entries": 5,
            "memo_hits": 2,
            "raw_queries_considered": 17,
            "representative_queries_considered": 9,
            "dominated_queries_removed": 8,
        },
        {
            "digest": "a",
            "solved": True,
            "plan": [3, 6, 4],
            "query_expansions": 8,
            "calls": 12,
            "memo_entries": 6,
            "memo_hits": 3,
            "raw_queries_considered": 19,
            "representative_queries_considered": 10,
            "dominated_queries_removed": 9,
        },
    ]


def test_canonical_commitment_is_order_independent() -> None:
    rows = sample_rows()
    assert kit.canonical_bytes(rows) == kit.canonical_bytes(list(reversed(rows)))


def test_validator_accepts_exact_preimage_and_rejects_mutation(tmp_path) -> None:
    rows = sample_rows()
    commitment = hashlib.sha256(kit.canonical_bytes(rows)).hexdigest()
    challenge = {
        "schema": kit.SCHEMA,
        "state_count": len(rows),
        "state_digests": sorted(row["digest"] for row in rows),
        "expected_canonical_output_sha256": commitment,
        "challenge_digest": "test-challenge",
    }
    challenge_path = tmp_path / "challenge.json"
    submission_path = tmp_path / "submission.json"
    challenge_path.write_text(json.dumps(challenge), encoding="utf-8")
    submission_path.write_text(
        json.dumps({"schema": kit.SCHEMA, "rows": rows}),
        encoding="utf-8",
    )
    assert validator.validate(challenge_path, submission_path)["passed"]

    rows[0]["calls"] = int(rows[0]["calls"]) + 1
    submission_path.write_text(
        json.dumps({"schema": kit.SCHEMA, "rows": rows}),
        encoding="utf-8",
    )
    rejected = validator.validate(challenge_path, submission_path)
    assert not rejected["passed"]
    assert rejected["errors"][0]["kind"] == "commitment-mismatch"
