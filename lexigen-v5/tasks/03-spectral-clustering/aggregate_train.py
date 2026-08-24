from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from candidates import PROVENANCE

EXPECTED_RECORDS = 100
EXPECTED_ROWS = 3000
VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05


def harmonic(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def provenance_map() -> dict[tuple[str, str], dict[str, object]]:
    result: dict[tuple[str, str], dict[str, object]] = {}
    for arm, rows in PROVENANCE.items():
        for row in rows:
            result[(arm, str(row["candidate"]))] = row
    if len(result) != 30:
        raise RuntimeError(f"expected 30 provenance rows, got {len(result)}")
    return result


def summarise(rows: list[dict[str, object]], arm: str, candidate: str, prov: dict[str, object]) -> dict[str, object]:
    selected = sorted(
        (row for row in rows if row["arm"] == arm and row["candidate"] == candidate),
        key=lambda row: int(row["index"]),
    )
    if len(selected) != EXPECTED_RECORDS or len({int(row["index"]) for row in selected}) != EXPECTED_RECORDS:
        raise RuntimeError(f"{arm}/{candidate} does not contain 100 unique training records")
    speeds = [float(row["speedup"]) for row in selected]
    valid_count = sum(bool(row["valid"]) for row in selected)
    retries = sum(int(row["invalid_output_retries"]) for row in selected)
    result: dict[str, object] = {
        "arm": arm,
        "candidate": candidate,
        "count": EXPECTED_RECORDS,
        "valid": valid_count,
        "invalid_outputs": EXPECTED_RECORDS - valid_count,
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "invalid_output_retries": retries,
        "candidate_executions": sum(int(row["candidate_executions"]) for row in selected),
        "proposal_id": prov["proposal_id"],
        "rank": prov["rank"],
        "operators": prov["operators"],
        "transfer_ids": prov["transfer_ids"],
        "learned_template": prov["learned_template"],
        "implementation_class": prov["implementation_class"],
        "semantic_signature": prov["semantic_signature"],
    }
    result["eligible_for_blind_selection"] = bool(valid_count == VALID_REQUIRED and retries == 0)
    result["passes_training_performance_gate"] = bool(
        result["eligible_for_blind_selection"]
        and float(result["harmonic_speedup"]) >= HARMONIC_REQUIRED
        and float(result["minimum_speedup"]) >= MINIMUM_REQUIRED
    )
    return result


def choose(rows: list[dict[str, object]]) -> dict[str, object] | None:
    eligible = [row for row in rows if bool(row["eligible_for_blind_selection"])]
    if not eligible:
        return None
    eligible.sort(key=lambda row: (
        -int(bool(row["passes_training_performance_gate"])),
        -float(row["harmonic_speedup"]),
        -float(row["minimum_speedup"]),
        -float(row["median_speedup"]),
        str(row["candidate"]),
    ))
    return eligible[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard_files = sorted(args.input.rglob("train-shard-*.jsonl"))
    if len(shard_files) != 10:
        raise RuntimeError(f"expected 10 training shard files, got {len(shard_files)}")
    rows: list[dict[str, object]] = []
    for path in shard_files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} training rows, got {len(rows)}")

    manifest_ids = {(
        str(row["train_manifest_name"]),
        str(row["train_manifest_tree_oid"]),
        str(row["train_manifest_git_blob_sha1"]),
        str(row["train_manifest_sha256"]),
        str(row["expected_test_manifest_name"]),
        str(row["expected_test_manifest_tree_oid"]),
        str(row["authoritative_source_git_blob_sha1"]),
    ) for row in rows}
    if len(manifest_ids) != 1:
        raise RuntimeError("training/source identity differs across shards")
    train_name, train_oid, train_blob, train_sha256, test_name, test_oid, source_blob = next(iter(manifest_ids))
    if train_oid != train_blob or train_oid != "a437767fb704bb4bcccaee3240de0f1080426c90":
        raise RuntimeError("frozen training manifest identity mismatch")
    if test_oid != "dc9082ea69c205b712e98da4b62fb387998e54df":
        raise RuntimeError("frozen test metadata identity mismatch")
    if source_blob != "7e8055db9c069388e4d3fe7468c4a6d4f33c02e8":
        raise RuntimeError("frozen authoritative source identity mismatch")
    if any(bool(row["test_manifest_contents_opened"]) or int(row["test_payloads_opened"]) != 0 for row in rows):
        raise RuntimeError("test boundary crossed during official training")

    prov = provenance_map()
    summaries: list[dict[str, object]] = []
    for arm in sorted(PROVENANCE):
        for prow in PROVENANCE[arm]:
            candidate = str(prow["candidate"])
            summaries.append(summarise(rows, arm, candidate, prov[(arm, candidate)]))

    by_arm: dict[str, list[dict[str, object]]] = {}
    for row in summaries:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    selected: dict[str, dict[str, object] | None] = {arm: choose(group) for arm, group in by_arm.items()}
    complete_selection = all(value is not None for value in selected.values())

    selected_compact = {
        arm: None if row is None else {
            "candidate": row["candidate"],
            "proposal_id": row["proposal_id"],
            "operators": row["operators"],
            "transfer_ids": row["transfer_ids"],
            "learned_template": row["learned_template"],
            "implementation_class": row["implementation_class"],
            "semantic_signature": row["semantic_signature"],
            "valid": row["valid"],
            "harmonic_speedup": row["harmonic_speedup"],
            "median_speedup": row["median_speedup"],
            "minimum_speedup": row["minimum_speedup"],
            "passes_training_performance_gate": row["passes_training_performance_gate"],
        }
        for arm, row in selected.items()
    }

    full = selected.get("v5_full")
    no_transfer = selected.get("v5_no_transfer")
    full_uses_transfer = bool(full and full.get("transfer_ids"))
    full_no_transfer_semantically_non_equivalent = bool(
        full and no_transfer and full.get("semantic_signature") != no_transfer.get("semantic_signature")
    )

    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 3,
        "task": "spectral_clustering",
        "stage": "official_training_r1",
        "training_records": EXPECTED_RECORDS,
        "raw_record_count": len(rows),
        "candidate_count": 30,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_oid,
        "train_manifest_git_blob_sha1": train_blob,
        "train_manifest_sha256": train_sha256,
        "expected_test_manifest_name": test_name,
        "expected_test_manifest_tree_oid": test_oid,
        "authoritative_source_git_blob_sha1": source_blob,
        "verifier": "exact_frozen_source_is_solution",
        "selection_policy": "require 100/100 validity and zero retries; sort by training-gate pass, harmonic speedup, minimum speedup, median speedup, stable candidate name",
        "frozen_training_performance_gate": {"valid_required": 100, "harmonic_speedup_minimum": 1.5, "minimum_speedup": 1.05, "invalid_output_retries": 0},
        "candidate_summaries": summaries,
        "selected_by_arm": selected_compact,
        "complete_five_arm_selection": complete_selection,
        "selected_v5_uses_learned_transfer": full_uses_transfer,
        "selected_full_no_transfer_semantically_non_equivalent": full_no_transfer_semantically_non_equivalent,
        "test_manifest_contents_opened": False,
        "test_payloads_opened": 0,
        "thresholds_changed": False,
        "human_task_specific_solver_design": False,
        "training_reruns": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "train-results.jsonl").write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["arm"]), str(row["candidate"])))) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    if not complete_selection:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
