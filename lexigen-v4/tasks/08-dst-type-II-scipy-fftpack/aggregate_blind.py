from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
EXPECTED = {
    "v4_full": "v4_zero_dtype_vector",
    "v4_no_transfer": "no_transfer_contiguous_dtype_vector",
    "random_search": "random_zero_dtype_vector",
    "template_synthesis": "template_dtype",
    "v3_compatible": "v3_dtype_specialization",
}


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    selected = sorted((r for r in rows if r["arm"] == arm and r["candidate"] == candidate), key=lambda r: int(r["index"]))
    if len(selected) != 100 or len({int(r["index"]) for r in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} does not contain exactly 100 unique blind records")
    speeds = [float(r["speedup"]) for r in selected]
    valid_rows = [r for r in selected if bool(r["valid"])]
    candidate_times = [float(r["candidate_s"]) for r in selected if r["candidate_s"] is not None]
    reference_times = [float(r["reference_s"]) for r in selected]
    summary: dict[str, object] = {
        "arm": arm,
        "candidate": candidate,
        "count": 100,
        "valid": len(valid_rows),
        "invalid_outputs": 100 - len(valid_rows),
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(candidate_times) if candidate_times else None,
        "median_reference_s": statistics.median(reference_times),
        "maximum_relative_error_to_reference": max((float(r["relative_error_to_reference"]) for r in valid_rows), default=float("inf")),
        "candidate_executions": sum(int(r["candidate_executions"]) for r in selected),
        "invalid_output_retries": sum(int(r["invalid_output_retries"]) for r in selected),
    }
    summary["passes_blind_gate"] = bool(
        int(summary["valid"]) == VALID_REQUIRED
        and float(summary["harmonic_speedup"]) >= HARMONIC_REQUIRED
        and float(summary["minimum_speedup"]) >= MINIMUM_REQUIRED
        and int(summary["invalid_output_retries"]) == 0
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard_files = sorted(args.input.rglob("blind-shard-*.jsonl"))
    if len(shard_files) != 10:
        raise RuntimeError(f"expected exactly 10 blind shard files, got {len(shard_files)}")
    rows: list[dict[str, object]] = []
    for path in shard_files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 raw blind rows, got {len(rows)}")
    expected_pairs = set(EXPECTED.items())
    seen = {(str(r["arm"]), str(r["candidate"])) for r in rows}
    if seen != expected_pairs:
        raise RuntimeError(f"blind candidate identity mismatch: missing={sorted(expected_pairs-seen)} extra={sorted(seen-expected_pairs)}")
    per_index: dict[int, int] = {}
    for row in rows:
        idx = int(row["index"])
        per_index[idx] = per_index.get(idx, 0) + 1
    if set(per_index) != set(range(1, 101)) or any(v != 5 for v in per_index.values()):
        raise RuntimeError("each of 100 blind records must contain exactly five arm rows")
    manifest_ids = {(str(r["test_manifest_name"]), str(r["test_manifest_git_blob_sha1"]), str(r["test_manifest_sha256"])) for r in rows}
    if len(manifest_ids) != 1:
        raise RuntimeError("blind manifest identity differs across shards")
    manifest_name, manifest_blob, manifest_sha256 = next(iter(manifest_ids))

    arms = {arm: summarise(rows, arm, candidate) for arm, candidate in EXPECTED.items()}
    v4 = arms["v4_full"]
    nt = arms["v4_no_transfer"]
    rnd = arms["random_search"]
    tmpl = arms["template_synthesis"]
    v3 = arms["v3_compatible"]
    architecture = {
        "v4_minus_v3_harmonic": float(v4["harmonic_speedup"]) - float(v3["harmonic_speedup"]),
        "v4_minus_template_harmonic": float(v4["harmonic_speedup"]) - float(tmpl["harmonic_speedup"]),
        "v4_minus_random_harmonic": float(v4["harmonic_speedup"]) - float(rnd["harmonic_speedup"]),
        "v4_minus_no_transfer_harmonic": float(v4["harmonic_speedup"]) - float(nt["harmonic_speedup"]),
        "v4_beats_v3_blind_raw_timing": float(v4["harmonic_speedup"]) > float(v3["harmonic_speedup"]),
        "v4_beats_template_blind_raw_timing": float(v4["harmonic_speedup"]) > float(tmpl["harmonic_speedup"]),
        "raw_timing_transfer_threshold_crossed": bool(float(v4["harmonic_speedup"]) > float(nt["harmonic_speedup"]) * 1.02 or bool(v4["passes_blind_gate"]) != bool(nt["passes_blind_gate"])),
        "causal_transfer_credit": False,
        "causal_transfer_credit_reason": "v4_full and v4_no_transfer selected winners use the same modern_float32 implementation; random_search also uses that implementation. Blind timing differences cannot establish transfer causality.",
        "implementation_equivalent_modern_float32_arms": ["v4_full", "v4_no_transfer", "random_search"]
    }
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 8,
        "task": "dst_type_II_scipy_fftpack",
        "revision": 1,
        "stage": "blind",
        "test_manifest_name": manifest_name,
        "test_manifest_git_blob_sha1": manifest_blob,
        "test_manifest_sha256": manifest_sha256,
        "blind_records": 100,
        "raw_record_count": len(rows),
        "matrix_shape": [2054, 2054],
        "selected_by_arm": EXPECTED,
        "frozen_gate": {"valid_required": 100, "harmonic_speedup_minimum": 1.5, "minimum_speedup": 1.05, "invalid_output_retries": 0},
        "arms": arms,
        "task_blind_status": "passed" if bool(v4["passes_blind_gate"]) else "failed",
        "architecture_comparison": architecture,
        "clean_unseen_task_win": bool(v4["passes_blind_gate"]),
        "transfer_credit": False,
        "invalid_output_retries": sum(int(r["invalid_output_retries"]) for r in rows),
        "blind_reruns": 0
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "blind-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "blind-results.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in sorted(rows, key=lambda r: (int(r["index"]), str(r["arm"])))) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
