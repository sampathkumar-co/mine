from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
EXPECTED_OID = "0bdbf8d4e6dd3897d50143dbf3778ca3e4e02f56"
SELECTED = {
    "v4_full": "v4_structure_refine_closed",
    "v4_no_transfer": "no_transfer_structure_refine_closed",
    "random_search": "random_zero_closed",
    "template_synthesis": "template_active_set",
    "v3_compatible": "v3_zero_copy_representation",
}


def harmonic(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    chosen = sorted((r for r in rows if r["arm"] == arm and r["candidate"] == candidate), key=lambda r: int(r["index"]))
    if len(chosen) != 100 or len({int(r["index"]) for r in chosen}) != 100:
        raise RuntimeError(f"{arm}/{candidate} does not contain exactly 100 unique blind records")
    speeds = [float(r["speedup"]) for r in chosen]
    valid = sum(bool(r["valid"]) for r in chosen)
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
        "maximum_fidelity_error": max(float(r["fidelity_error"]) for r in chosen if bool(r["valid"])),
        "maximum_nuclear_ratio_to_reference": max(float(r["nuclear_ratio_to_reference"]) for r in chosen if bool(r["valid"])),
        "candidate_executions": sum(int(r["candidate_executions"]) for r in chosen),
        "reference_executions_observed": len(chosen),
        "invalid_output_retries": sum(int(r["invalid_output_retries"]) for r in chosen),
    }
    summary["passes_blind_gate"] = bool(valid == VALID_REQUIRED and float(summary["harmonic_speedup"]) >= HARMONIC_REQUIRED and float(summary["minimum_speedup"]) >= MINIMUM_REQUIRED and int(summary["invalid_output_retries"]) == 0)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input.rglob("blind-shard-*.jsonl"))
    if len(files) != 10:
        raise RuntimeError(f"expected 10 blind shard files, got {len(files)}")
    rows: list[dict[str, object]] = []
    for path in files:
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 blind rows, got {len(rows)}")
    seen = {(str(r["arm"]), str(r["candidate"])) for r in rows}
    expected = set(SELECTED.items())
    if seen != expected:
        raise RuntimeError(f"blind selection mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")
    per_index: dict[int, int] = {}
    for r in rows:
        idx = int(r["index"])
        per_index[idx] = per_index.get(idx, 0) + 1
    if set(per_index) != set(range(1, 101)) or any(v != 5 for v in per_index.values()):
        raise RuntimeError("each blind record must contain exactly five selected candidates")
    oids = {str(r["test_manifest_git_blob_sha1"]) for r in rows}
    sha256s = {str(r["test_manifest_sha256"]) for r in rows}
    if oids != {EXPECTED_OID} or len(sha256s) != 1:
        raise RuntimeError(f"blind manifest identity inconsistency: oids={oids} sha256s={sha256s}")

    arms = {arm: summarise(rows, arm, candidate) for arm, candidate in SELECTED.items()}
    v4, nt, rnd, tmpl, v3 = (arms[k] for k in ["v4_full", "v4_no_transfer", "random_search", "template_synthesis", "v3_compatible"])
    comparison = {
        "v4_minus_v3_harmonic": float(v4["harmonic_speedup"]) - float(v3["harmonic_speedup"]),
        "v4_minus_template_harmonic": float(v4["harmonic_speedup"]) - float(tmpl["harmonic_speedup"]),
        "v4_minus_random_harmonic": float(v4["harmonic_speedup"]) - float(rnd["harmonic_speedup"]),
        "v4_minus_no_transfer_harmonic": float(v4["harmonic_speedup"]) - float(nt["harmonic_speedup"]),
        "v4_beats_v3_blind": bool(v4["passes_blind_gate"] and (not v3["passes_blind_gate"] or float(v4["harmonic_speedup"]) > float(v3["harmonic_speedup"]))),
        "v4_beats_template_blind": bool(v4["passes_blind_gate"] and (not tmpl["passes_blind_gate"] or float(v4["harmonic_speedup"]) > float(tmpl["harmonic_speedup"]))),
        "transfer_advantage_observed": bool(float(v4["harmonic_speedup"]) > float(nt["harmonic_speedup"]) * 1.02 or bool(v4["passes_blind_gate"]) != bool(nt["passes_blind_gate"])),
        "implementation_equivalent_to_no_transfer_and_random": True,
    }
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 5,
        "task": "tensor_completion_3d",
        "revision": 1,
        "stage": "blind",
        "test_manifest_git_blob_sha1": EXPECTED_OID,
        "test_manifest_sha256": next(iter(sha256s)),
        "blind_records": 100,
        "raw_record_count": 500,
        "selected_by_arm": SELECTED,
        "arms": arms,
        "task_blind_status": "passed" if bool(v4["passes_blind_gate"]) else "failed",
        "architecture_comparison": comparison,
        "clean_unseen_task_win": bool(v4["passes_blind_gate"]),
        "transfer_credit": False,
        "invalid_output_retries": 0,
        "blind_reruns": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "blind-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "blind-results.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in sorted(rows, key=lambda r: (int(r["index"]), str(r["arm"])))) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
