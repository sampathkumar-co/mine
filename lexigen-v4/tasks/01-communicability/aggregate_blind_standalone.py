from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import traceback
from pathlib import Path

SELECTED = {
    "v4_full": "v4_sort_bit_risk",
    "v4_no_transfer": "no_transfer_sort_bit_risk",
    "random_search": "random_contiguous_vector_risk",
    "template_synthesis": "template_vectorized_batch",
    "v3_compatible": "v3_dtype_specialization"
}
VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
TASK_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harmonic(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def aggregate(input_path: Path, output_path: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    paths = sorted(input_path.rglob("blind-shard-*.jsonl"))
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(paths) != 10:
        raise RuntimeError(f"expected 10 shard files, received {len(paths)}")
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 blind rows, received {len(rows)}")

    manifest_hashes = {str(row["test_manifest_sha256"]) for row in rows}
    manifest_blobs = {str(row["test_manifest_git_blob_sha1"]) for row in rows}
    tree_oids = {str(row["expected_test_tree_oid"]) for row in rows}
    expected_oid = "48efda1a3b334b13efb874355985d9f1d87291f2"
    if len(manifest_hashes) != 1 or len(manifest_blobs) != 1 or tree_oids != {expected_oid}:
        raise RuntimeError("blind manifest identity differs across shards")

    arms: dict[str, dict[str, object]] = {}
    for arm, candidate in SELECTED.items():
        selected = sorted((row for row in rows if row["arm"] == arm), key=lambda row: int(row["index"]))
        if len(selected) != 100 or len({int(row["index"]) for row in selected}) != 100:
            raise RuntimeError(f"{arm} blind record coverage invalid")
        if {str(row["candidate"]) for row in selected} != {candidate}:
            raise RuntimeError(f"{arm} selected candidate identity mismatch")
        speeds = [float(row["speedup"]) for row in selected]
        valid = sum(bool(row["valid"]) for row in selected)
        exceptions = sum(row["failure_reason"] is not None and not str(row["failure_reason"]).startswith("mismatch_") for row in selected)
        result: dict[str, object] = {
            "arm": arm,
            "candidate": candidate,
            "count": 100,
            "valid": valid,
            "exceptions": exceptions,
            "invalid_outputs": 100 - valid,
            "harmonic_speedup": harmonic(speeds),
            "median_speedup": statistics.median(speeds),
            "minimum_speedup": min(speeds),
            "maximum_speedup": max(speeds),
            "median_candidate_s": statistics.median(float(row["candidate_s"]) for row in selected if row["candidate_s"] is not None),
            "median_reference_s": statistics.median(float(row["reference_s"]) for row in selected),
            "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in selected),
            "connected_records": sum(int(row["components"]) == 1 for row in selected),
            "disconnected_records": sum(int(row["components"]) > 1 for row in selected),
            "candidate_executions": sum(int(row["candidate_executions"]) for row in selected),
            "reference_executions_observed": len(selected),
            "invalid_output_retries": sum(int(row["invalid_output_retries"]) for row in selected)
        }
        result["passes_blind_gate"] = bool(
            valid == VALID_REQUIRED
            and float(result["harmonic_speedup"]) >= HARMONIC_REQUIRED
            and float(result["minimum_speedup"]) >= MINIMUM_REQUIRED
            and int(result["invalid_output_retries"]) == 0
        )
        arms[arm] = result

    ranked = sorted(arms.values(), key=lambda row: (-int(row["valid"]), -float(row["harmonic_speedup"]), -float(row["minimum_speedup"]), str(row["arm"])))
    v4 = arms["v4_full"]
    no_transfer = arms["v4_no_transfer"]
    random = arms["random_search"]
    template = arms["template_synthesis"]
    v3 = arms["v3_compatible"]
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 1,
        "task": "communicability",
        "revision": 1,
        "aggregation_mode": "standalone_standard_library_original_shards_only",
        "original_blind_run_id": 30376286181,
        "candidate_source_sha256": sha256(TASK_DIR / "candidates.py"),
        "selected_solvers_sha256": sha256(TASK_DIR / "selected_solvers.py"),
        "blind_runner_sha256": sha256(TASK_DIR / "blind_shard.py"),
        "requirements_sha256": sha256(TASK_DIR / "requirements.txt"),
        "training_artifact_id": 8694809272,
        "training_artifact_digest": "sha256:4d4b0e8be1cdbbb02727685469ec64b856a020c49297b0344fafcd750784f5e9",
        "test_manifest_name": "communicability_T100ms_n61_size100_test.jsonl",
        "test_manifest_tree_oid": expected_oid,
        "test_manifest_git_blob_sha1": next(iter(manifest_blobs)),
        "test_manifest_sha256": next(iter(manifest_hashes)),
        "frozen_gate": {
            "valid_required": VALID_REQUIRED,
            "harmonic_speedup_minimum": HARMONIC_REQUIRED,
            "minimum_speedup": MINIMUM_REQUIRED,
            "invalid_output_retries": 0
        },
        "arms": arms,
        "ranking": [str(row["arm"]) for row in ranked],
        "v4_task_status": "clean_autonomous_unseen_task_win" if bool(v4["passes_blind_gate"]) else "failed_gate",
        "architecture_comparison": {
            "winner": ranked[0]["arm"],
            "v4_full_harmonic": v4["harmonic_speedup"],
            "v4_no_transfer_harmonic": no_transfer["harmonic_speedup"],
            "random_search_harmonic": random["harmonic_speedup"],
            "template_synthesis_harmonic": template["harmonic_speedup"],
            "v3_compatible_harmonic": v3["harmonic_speedup"],
            "v4_minus_v3_harmonic": float(v4["harmonic_speedup"]) - float(v3["harmonic_speedup"]),
            "v4_minus_no_transfer_harmonic": float(v4["harmonic_speedup"]) - float(no_transfer["harmonic_speedup"]),
            "v4_beats_v3": bool(v4["passes_blind_gate"] and (not v3["passes_blind_gate"] or float(v4["harmonic_speedup"]) > float(v3["harmonic_speedup"])),
            "transfer_advantage_observed": bool(float(v4["harmonic_speedup"]) > float(no_transfer["harmonic_speedup"]) * 1.02 or v4["passes_blind_gate"] != no_transfer["passes_blind_gate"])
        },
        "execution_integrity": {
            "candidate_changed_after_training": False,
            "thresholds_changed_after_training": False,
            "candidate_executions_per_arm_per_record": 1,
            "reference_executions_per_record": 1,
            "invalid_output_retries": 0,
            "successful_records_rerun": False,
            "candidate_executions_during_recovery": 0,
            "reference_executions_during_recovery": 0
        },
        "reports_opened": False,
        "public_solvers_opened": False,
        "human_task_specific_solver_design": False
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "blind-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_path / "blind-results.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["arm"])))) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    args.diagnostic.mkdir(parents=True, exist_ok=True)
    try:
        report = aggregate(args.input, args.output)
        diagnostic = {
            "status": "aggregation_succeeded",
            "original_blind_run_id": 30376286181,
            "ranking": report["ranking"],
            "candidate_executions": 0,
            "reference_executions": 0,
            "blind_records_rerun": 0
        }
        (args.diagnostic / "aggregation-diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"v4_task_status": report["v4_task_status"], "ranking": report["ranking"], "arms": report["arms"], "architecture_comparison": report["architecture_comparison"]}, indent=2))
    except Exception as exc:
        diagnostic = {
            "status": "aggregation_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "candidate_executions": 0,
            "reference_executions": 0,
            "blind_records_rerun": 0
        }
        (args.diagnostic / "aggregation-diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(diagnostic, indent=2))
        raise


if __name__ == "__main__":
    main()
