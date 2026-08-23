from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
EXPECTED = {
    "v4_full": [
        "v4_active_vector_risk",
        "v4_active_partition_vector",
        "v4_active_partition_risk",
        "v4_zero_active_vector",
        "v4_zero_active_risk",
        "v4_dtype_active_risk",
    ],
    "v4_no_transfer": [
        "no_transfer_active_vector_risk",
        "no_transfer_active_partition_vector",
        "no_transfer_active_partition_risk",
        "no_transfer_partition_vector_risk",
        "no_transfer_zero_active_vector",
        "no_transfer_dtype_active_risk",
    ],
    "random_search": [
        "random_zero_active",
        "random_dtype_risk",
        "random_sort_partition",
        "random_zero_dtype_risk",
        "random_zero_dtype_vector",
        "random_contiguous_dtype_active",
    ],
    "template_synthesis": [
        "template_active_set",
        "template_vectorized_batch",
        "template_risk_stage",
        "template_sort_partition",
        "template_zero_copy",
        "template_dtype",
    ],
    "v3_compatible": [
        "v3_vectorized_batch",
        "v3_zero_copy_representation",
        "v3_dtype_specialization",
        "v3_contiguous_layout",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    selected = sorted(
        (r for r in rows if r["arm"] == arm and r["candidate"] == candidate),
        key=lambda r: int(r["index"]),
    )
    if len(selected) != 100 or len({int(r["index"]) for r in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} does not contain exactly 100 unique records")
    speeds = [float(r["speedup"]) for r in selected]
    valid_rows = [r for r in selected if bool(r["valid"])]
    valid = len(valid_rows)
    candidate_times = [float(r["candidate_s"]) for r in selected if r["candidate_s"] is not None]
    reference_times = [float(r["reference_s"]) for r in selected]
    summary: dict[str, object] = {
        "arm": arm,
        "candidate": candidate,
        "count": 100,
        "valid": valid,
        "invalid_outputs": 100 - valid,
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(candidate_times) if candidate_times else None,
        "median_reference_s": statistics.median(reference_times),
        "total_candidate_s": sum(candidate_times),
        "maximum_abs_error_to_reference": max((float(r["max_abs_error_to_reference"]) for r in valid_rows), default=float("inf")),
        "maximum_simplex_sum_error": max((float(r["simplex_sum_error"]) for r in valid_rows), default=float("inf")),
        "candidate_executions": sum(int(r["candidate_executions"]) for r in selected),
        "invalid_output_retries": sum(int(r["invalid_output_retries"]) for r in selected),
    }
    summary["passes_training_gate"] = bool(
        valid == VALID_REQUIRED
        and float(summary["harmonic_speedup"]) >= HARMONIC_REQUIRED
        and float(summary["minimum_speedup"]) >= MINIMUM_REQUIRED
        and int(summary["invalid_output_retries"]) == 0
    )
    return summary


def select_arm(summaries: list[dict[str, object]], arm: str) -> dict[str, object]:
    arm_rows = [r for r in summaries if r["arm"] == arm]
    eligible = [r for r in arm_rows if bool(r["passes_training_gate"])]
    pool = eligible if eligible else arm_rows
    selected = min(
        pool,
        key=lambda r: (-int(r["valid"]), -float(r["harmonic_speedup"]), -float(r["minimum_speedup"]), str(r["candidate"])),
    )
    return {
        "arm": arm,
        "passes_training_gate": bool(eligible),
        "selected": selected,
        "eligible_candidate_count": len(eligible),
        "candidate_count": len(arm_rows),
        "discovery_cost_total_candidate_s": sum(float(r["total_candidate_s"]) for r in arm_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard_files = sorted(args.input.rglob("train-shard-*.jsonl"))
    if len(shard_files) != 10:
        raise RuntimeError(f"expected exactly 10 training shard files, got {len(shard_files)}")
    rows: list[dict[str, object]] = []
    for path in shard_files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if sum(len(v) for v in EXPECTED.values()) != 28 or len(rows) != 2800:
        raise RuntimeError(f"expected 28 candidates and 2800 rows, got {sum(len(v) for v in EXPECTED.values())} and {len(rows)}")
    seen = {(str(r["arm"]), str(r["candidate"])) for r in rows}
    expected_pairs = {(arm, name) for arm, names in EXPECTED.items() for name in names}
    if seen != expected_pairs:
        raise RuntimeError(f"candidate identity mismatch: missing={sorted(expected_pairs-seen)} extra={sorted(seen-expected_pairs)}")
    per_index: dict[int, int] = {}
    for r in rows:
        idx = int(r["index"])
        per_index[idx] = per_index.get(idx, 0) + 1
    if set(per_index) != set(range(1, 101)) or any(v != 28 for v in per_index.values()):
        raise RuntimeError("each of 100 records must contain exactly 28 candidate rows")

    summaries = [summarise(rows, arm, name) for arm, names in EXPECTED.items() for name in names]
    arms = {arm: select_arm(summaries, arm) for arm in EXPECTED}
    v4 = arms["v4_full"]["selected"]
    nt = arms["v4_no_transfer"]["selected"]
    rnd = arms["random_search"]["selected"]
    tmpl = arms["template_synthesis"]["selected"]
    v3 = arms["v3_compatible"]["selected"]
    comparison = {
        "v4_full_harmonic": v4["harmonic_speedup"],
        "v4_no_transfer_harmonic": nt["harmonic_speedup"],
        "random_search_harmonic": rnd["harmonic_speedup"],
        "template_synthesis_harmonic": tmpl["harmonic_speedup"],
        "v3_compatible_harmonic": v3["harmonic_speedup"],
        "v4_minus_v3_harmonic": float(v4["harmonic_speedup"]) - float(v3["harmonic_speedup"]),
        "v4_minus_template_harmonic": float(v4["harmonic_speedup"]) - float(tmpl["harmonic_speedup"]),
        "v4_minus_random_harmonic": float(v4["harmonic_speedup"]) - float(rnd["harmonic_speedup"]),
        "v4_minus_no_transfer_harmonic": float(v4["harmonic_speedup"]) - float(nt["harmonic_speedup"]),
        "v4_beats_v3_on_training": bool(arms["v4_full"]["passes_training_gate"] and (not arms["v3_compatible"]["passes_training_gate"] or float(v4["harmonic_speedup"]) > float(v3["harmonic_speedup"]))),
        "v4_beats_template_on_training": bool(arms["v4_full"]["passes_training_gate"] and (not arms["template_synthesis"]["passes_training_gate"] or float(v4["harmonic_speedup"]) > float(tmpl["harmonic_speedup"]))),
        "transfer_advantage_observed": bool(float(v4["harmonic_speedup"]) > float(nt["harmonic_speedup"]) * 1.02 or bool(arms["v4_full"]["passes_training_gate"]) != bool(arms["v4_no_transfer"]["passes_training_gate"])),
    }
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 6,
        "task": "unit_simplex_projection",
        "revision": 1,
        "candidate_source_sha256": sha256(Path(__file__).resolve().parent / "candidates.py"),
        "training_runner_sha256": sha256(Path(__file__).resolve().parent / "train_shard.py"),
        "aggregator_sha256": sha256(Path(__file__)),
        "requirements_sha256": sha256(Path(__file__).resolve().parent / "requirements.txt"),
        "train_manifest_name": "unit_simplex_projection_T100ms_n982958_size100_train.jsonl",
        "train_manifest_tree_oid": "ce1828ef37b2c3c3dde19d4ed9842aec33ed6b79",
        "train_manifest_sha256": "fbb47d089bbe5ffeab8f930947e14aa3afc968f5159eeca427677dc0a9e39419",
        "expected_test_manifest_name": "unit_simplex_projection_T100ms_n982958_size100_test.jsonl",
        "expected_test_manifest_tree_oid": "33358f9cad20ab1d145ccf51f63caaad091cc35f",
        "vector_length": 982958,
        "frozen_gate": {"valid_required": 100, "harmonic_speedup_minimum": 1.5, "minimum_speedup": 1.05, "invalid_output_retries": 0},
        "candidate_count": 28,
        "raw_record_count": len(rows),
        "all_candidates": summaries,
        "arms": arms,
        "task_training_status": "passed" if arms["v4_full"]["passes_training_gate"] else "failed",
        "architecture_comparison": comparison,
        "test_manifest_opened": False,
        "test_payloads_opened": 0,
        "reports_opened": False,
        "public_solvers_opened": False,
        "training_revision_consumed": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "training-results.jsonl").write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in sorted(rows, key=lambda r: (int(r["index"]), str(r["arm"]), str(r["candidate"])))) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"task_training_status": report["task_training_status"], "selected_by_arm": {a: x["selected"] for a, x in arms.items()}, "architecture_comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
