from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .external_replication_kit_v76 import OUTPUT_FIELDS, SCHEMA, canonical_bytes


def validate(challenge_path: Path, submission_path: Path) -> dict[str, object]:
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    errors: list[dict[str, object]] = []

    if challenge.get("schema") != SCHEMA:
        raise RuntimeError("unsupported challenge schema")
    if submission.get("schema") != SCHEMA:
        errors.append({"kind": "schema", "expected": SCHEMA, "actual": submission.get("schema")})

    rows = submission.get("rows")
    if not isinstance(rows, list):
        rows = []
        errors.append({"kind": "rows-not-list"})

    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append({"kind": "row-not-object", "index": index})
            continue
        missing = [field for field in OUTPUT_FIELDS if field not in row]
        extras = sorted(set(row) - set(OUTPUT_FIELDS))
        if missing:
            errors.append({"kind": "missing-fields", "index": index, "fields": missing})
            continue
        if extras:
            errors.append({"kind": "extra-fields", "index": index, "fields": extras})
            continue
        digest = str(row["digest"])
        if digest in seen:
            errors.append({"kind": "duplicate-digest", "digest": digest})
            continue
        seen.add(digest)
        plan = row["plan"]
        integer_fields = OUTPUT_FIELDS[3:]
        if row["solved"] is not True:
            errors.append({"kind": "unsolved", "digest": digest})
        if not (
            isinstance(plan, list)
            and len(plan) == 3
            and all(isinstance(value, int) and not isinstance(value, bool) for value in plan)
        ):
            errors.append({"kind": "invalid-plan", "digest": digest, "plan": plan})
        for field in integer_fields:
            value = row[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append({"kind": "invalid-counter", "digest": digest, "field": field, "value": value})
        normalized.append({field: row[field] for field in OUTPUT_FIELDS})

    expected_digests = set(str(value) for value in challenge["state_digests"])
    missing_digests = sorted(expected_digests - seen)
    unexpected_digests = sorted(seen - expected_digests)
    if missing_digests:
        errors.append({"kind": "missing-digests", "count": len(missing_digests), "examples": missing_digests[:20]})
    if unexpected_digests:
        errors.append({"kind": "unexpected-digests", "count": len(unexpected_digests), "examples": unexpected_digests[:20]})
    if len(normalized) != int(challenge["state_count"]):
        errors.append({"kind": "state-count", "expected": challenge["state_count"], "actual": len(normalized)})

    actual_commitment = (
        hashlib.sha256(canonical_bytes(normalized)).hexdigest()
        if not errors else None
    )
    expected_commitment = str(challenge["expected_canonical_output_sha256"])
    commitment_match = actual_commitment == expected_commitment
    if not errors and not commitment_match:
        errors.append({
            "kind": "commitment-mismatch",
            "expected": expected_commitment,
            "actual": actual_commitment,
        })

    passed = not errors and commitment_match
    return {
        "status": "external_replication_submission_matches" if passed else "rejected",
        "passed": passed,
        "schema": SCHEMA,
        "challenge_digest": challenge["challenge_digest"],
        "expected_commitment": expected_commitment,
        "actual_commitment": actual_commitment,
        "row_count": len(normalized),
        "error_count": len(errors),
        "errors": errors[:50],
        "claim_scope": (
            "A matching commitment proves output equality for the frozen bundle. "
            "It counts as outside reproduction only after the submitter's source "
            "and independence attestation are reviewed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.challenge, args.submission)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
