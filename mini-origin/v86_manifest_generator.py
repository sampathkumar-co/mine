#!/usr/bin/env python3
"""Deterministic metadata-only manifest generator for Mini-ORIGIN v0.86.

The generator consumes a normalized OpenML-CC18 registry metadata snapshot. It
never fetches task contents, feature values, labels, folds, ARFF files, or data
files. Candidate ordering and seven-task selection are fixed by the v0.86
preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SELECTION_PREFIX = "mini-origin-v86:"
SELECTED_TASK_COUNT = 7
REQUIRED_FIELDS = (
    "task_id",
    "task_version",
    "dataset_id",
    "dataset_version",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selection_digest(task_id: int) -> str:
    return sha256_hex(f"{SELECTION_PREFIX}{task_id}".encode("utf-8"))


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"metadata record missing required fields: {', '.join(missing)}")

    normalized = {
        "task_id": int(record["task_id"]),
        "task_version": int(record["task_version"]),
        "dataset_id": int(record["dataset_id"]),
        "dataset_version": int(record["dataset_version"]),
        "dataset_name": str(record.get("dataset_name", "")),
        "status": str(record.get("status", "active")),
        "license": str(record.get("license", "")),
        "exclusion_reason": record.get("exclusion_reason"),
    }
    if normalized["task_id"] <= 0 or normalized["dataset_id"] <= 0:
        raise ValueError("task_id and dataset_id must be positive")
    if normalized["task_version"] <= 0 or normalized["dataset_version"] <= 0:
        raise ValueError("task_version and dataset_version must be positive")
    if normalized["exclusion_reason"] is not None:
        normalized["exclusion_reason"] = str(normalized["exclusion_reason"]).strip()
        if not normalized["exclusion_reason"]:
            raise ValueError("exclusion_reason must be non-empty when supplied")
    return normalized


def generate_manifest(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [_normalize_record(record) for record in records]
    task_ids = [record["task_id"] for record in normalized]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("metadata snapshot contains duplicate task_id values")

    snapshot = sorted(normalized, key=lambda record: record["task_id"])
    eligible = [record for record in snapshot if record["exclusion_reason"] is None]
    excluded = [record for record in snapshot if record["exclusion_reason"] is not None]
    if len(eligible) < SELECTED_TASK_COUNT:
        raise ValueError(
            f"need at least {SELECTED_TASK_COUNT} eligible tasks, found {len(eligible)}"
        )

    ranked = sorted(
        eligible,
        key=lambda record: (selection_digest(record["task_id"]), record["task_id"]),
    )
    selected = ranked[:SELECTED_TASK_COUNT]

    manifest_core = {
        "campaign": "mini-origin-v86-untouched-external-validation",
        "selection_prefix": SELECTION_PREFIX,
        "selection_rule": "ascending SHA-256('mini-origin-v86:' || decimal task_id), then ascending task_id tie-break",
        "selected_task_count": SELECTED_TASK_COUNT,
        "metadata_only": True,
        "eligible_tasks_by_task_id": eligible,
        "excluded_tasks_by_task_id": excluded,
        "ranked_eligible_task_ids": [record["task_id"] for record in ranked],
        "selected_tasks": [
            {**record, "selection_digest": selection_digest(record["task_id"])}
            for record in selected
        ],
    }
    snapshot_bytes = canonical_json_bytes(snapshot)
    manifest_core["metadata_snapshot_sha256"] = sha256_hex(snapshot_bytes)
    manifest_core["manifest_sha256"] = sha256_hex(canonical_json_bytes(manifest_core))
    return manifest_core


def self_test() -> None:
    records = [
        {
            "task_id": task_id,
            "task_version": 1,
            "dataset_id": 1000 + task_id,
            "dataset_version": 1,
            "dataset_name": f"metadata-only-{task_id}",
            "exclusion_reason": "pre-access test exclusion" if task_id == 4 else None,
        }
        for task_id in range(1, 11)
    ]
    forward = generate_manifest(records)
    reverse = generate_manifest(list(reversed(records)))
    assert forward == reverse
    assert len(forward["selected_tasks"]) == 7
    assert 4 not in [row["task_id"] for row in forward["selected_tasks"]]
    assert forward["manifest_sha256"] == sha256_hex(
        canonical_json_bytes({key: value for key, value in forward.items() if key != "manifest_sha256"})
    )

    mutated = [dict(record) for record in records]
    mutated[0]["dataset_version"] = 2
    changed = generate_manifest(mutated)
    assert changed["metadata_snapshot_sha256"] != forward["metadata_snapshot_sha256"]
    assert changed["manifest_sha256"] != forward["manifest_sha256"]
    print("v0.86 metadata-only manifest generator self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.metadata is None or args.output is None:
        parser.error("--metadata and --output are required unless --self-test is used")

    records = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("metadata snapshot must be a JSON list")
    manifest = generate_manifest(records)
    args.output.write_bytes(canonical_json_bytes(manifest))
    print(f"wrote {args.output} with manifest_sha256={manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
