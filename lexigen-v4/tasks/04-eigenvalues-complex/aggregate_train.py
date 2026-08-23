from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from candidates import CANDIDATES_BY_ARM

TASK_DIR = Path(__file__).resolve().parent
VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harmonic(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def names_for_arm(arm: str) -> list[str]:
    return [name for name, _ in CANDIDATES_BY_ARM[arm]]


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    selected = sorted((r for r in rows if r["arm"] == arm and r["candidate"] == candidate), key=lambda r: int(r["index"]))
    if len(selected) != 100 or len({int(r["index"]) for r in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} does not contain exactly 100 unique records")
    speeds = [float(r["speedup"]) for r in selected]
    candidate_times = [float(r["candidate_s"]) for r in selected if r["candidate_s"] is not None]
    reference_times = [float(r["reference_s"]) for r in selected]
    valid = sum(bool(r["valid"]) for r in selected)
    summary = {
        "arm": arm,
        "candidate": candidate,
        "count": 100,
        "valid": valid,
        "exceptions": sum(r["failure_reason"] is not None and str(r["failure_reason"]) not in {"relative_error", "not_sorted", "solution_shape_or_type", "non_complex_value", "nonfinite_value"} for r in selected),
        "invalid_outputs": 100 - valid,
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(candidate_times),
        "median_reference_s": statistics.median(reference_times),
        "total_candidate_s": sum(candidate_times),
        "maximum_relative_error": max(float(r["maximum_relative_error"]) for r in selected if bool(r["valid"])),
        "candidate_executions": sum(int(r["candidate_executions"]) for r in selected),
        "reference_executions_observed": len(selected),
        "invalid_output_retries": sum(int(r["invalid_output_retries"]) for r in selected),
    }
    summary["passes_training_gate"] = bool(valid == VALID_REQUIRED and summary["harmonic_speedup"] >= HARMONIC_REQUIRED and summary["minimum_speedup"] >= MINIMUM_REQUIRED and summary["invalid_output_retries"] == 0)
    return summary


def select_arm(summaries: list[dict[str, object]], arm: str) -> dict[str, object]:
    arm_rows = [r for r in summaries if r["arm"] == arm]
    eligible = [r for r in arm_rows if r["passes_training_gate"]]
    pool = eligible if eligible else arm_rows
    selected = min(pool, key=lambda r: (-int(r["valid"]), -float(r["harmonic_speedup"]), -float(r["minimum_speedup"]), str(r["candidate"])))
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
    rows: list[dict[str, object]] = []
    for path in sorted(args.input.rglob("train-shard-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    expected_candidates = sum(len(items) for items in CANDIDATES_BY_ARM.values())
    if len(rows) != expected_candidates * 100:
        raise RuntimeError(f"expected {expected_candidates * 100} rows, received {len(rows)}")

    summaries = [summarise(rows, arm, name) for arm in CANDIDATES_BY_ARM for name in names_for_arm(arm)]
    arms = {arm: select_arm(summaries, arm) for arm in CANDIDATES_BY_ARM}
    v4 = arms["v4_full"]["selected"]
    no_transfer = arms["v4_no_transfer"]["selected"]
    random = arms["random_search"]["selected"]
    template = arms["template_synthesis"]["selected"]
    v3 = arms["v3_compatible"]["selected"]
    comparison = {
        "v4_full_harmonic": v4["harmonic_speedup"],
        "v4_no_transfer_harmonic": no_transfer["harmonic_speedup"],
        "random_search_harmonic": random["harmonic_speedup"],
        "template_synthesis_harmonic": template["harmonic_speedup"],
        "v3_compatible_harmonic": v3["harmonic_speedup"],
        "v4_minus_v3_harmonic": float(v4["harmonic_speedup"]) - float(v3["harmonic_speedup"]),
        "v4_minus_template_harmonic": float(v4["harmonic_speedup"]) - float(template["harmonic_speedup"]),
        "v4_minus_random_harmonic": float(v4["harmonic_speedup"]) - float(random["harmonic_speedup"]),
        "v4_minus_no_transfer_harmonic": float(v4["harmonic_speedup"]) - float(no_transfer["harmonic_speedup"]),
        "v4_beats_v3_on_training": bool(arms["v4_full"]["passes_training_gate"] and (not arms["v3_compatible"]["passes_training_gate"] or float(v4["harmonic_speedup"]) > float(v3["harmonic_speedup"]))),
        "v4_beats_template_on_training": bool(arms["v4_full"]["passes_training_gate"] and (not arms["template_synthesis"]["passes_training_gate"] or float(v4["harmonic_speedup"]) > float(template["harmonic_speedup"]))),
        "transfer_advantage_observed": bool(float(v4["harmonic_speedup"]) > float(no_transfer["harmonic_speedup"]) * 1.02 or arms["v4_full"]["passes_training_gate"] != arms["v4_no_transfer"]["passes_training_gate"]),
    }
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 4,
        "task": "eigenvalues_complex",
        "revision": 1,
        "candidate_source_sha256": sha256(TASK_DIR / "candidates.py"),
        "training_runner_sha256": sha256(TASK_DIR / "train_shard.py"),
        "aggregator_sha256": sha256(TASK_DIR / "aggregate_train.py"),
        "requirements_sha256": sha256(TASK_DIR / "requirements.txt"),
        "train_manifest_name": "eigenvalues_complex_T100ms_n474_size100_train.jsonl",
        "train_manifest_tree_oid": "9e1372505b483e1ecdd24179d5577c710dfec4dd",
        "train_manifest_sha256": "4d1eb81b05e772d1238cd693dcb5c2463cac3521b3a4a79f5d9b3e2139c09270",
        "expected_test_manifest_name": "eigenvalues_complex_T100ms_n474_size100_test.jsonl",
        "expected_test_manifest_tree_oid": "2587b20b27c657ac64a952bb991c0e07db462858",
        "frozen_gate": {"valid_required": 100, "harmonic_speedup_minimum": 1.5, "minimum_speedup": 1.05, "invalid_output_retries": 0},
        "candidate_count": expected_candidates,
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
        "human_task_specific_solver_design": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "training-results.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in sorted(rows, key=lambda r: (int(r["index"]), str(r["arm"]), str(r["candidate"])))) + "\n", encoding="utf-8")
    print(json.dumps({"task_training_status": report["task_training_status"], "selected_by_arm": {arm: result["selected"] for arm, result in arms.items()}, "architecture_comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
