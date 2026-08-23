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


def summarise(rows: list[dict[str, object]], arm: str, candidate: str) -> dict[str, object]:
    selected = sorted(
        (row for row in rows if row["arm"] == arm and row["candidate"] == candidate),
        key=lambda row: int(row["index"]),
    )
    if len(selected) != 100:
        raise RuntimeError(f"{arm}/{candidate} has {len(selected)} rows")
    if len({int(row["index"]) for row in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} has duplicate training records")
    speeds = [float(row["speedup"]) for row in selected]
    candidate_times = [float(row["candidate_s"]) for row in selected if row["candidate_s"] is not None]
    reference_times = [float(row["reference_s"]) for row in selected]
    valid = sum(bool(row["valid"]) for row in selected)
    exceptions = sum(
        row["failure_reason"] is not None and not str(row["failure_reason"]).startswith("mismatch_")
        for row in selected
    )
    summary = {
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
        "median_candidate_s": statistics.median(candidate_times),
        "median_reference_s": statistics.median(reference_times),
        "total_candidate_s": sum(candidate_times),
        "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in selected),
        "connected_records": sum(int(row["components"]) == 1 for row in selected),
        "disconnected_records": sum(int(row["components"]) > 1 for row in selected),
        "candidate_executions": sum(int(row["candidate_executions"]) for row in selected),
        "reference_executions_observed": len(selected),
        "invalid_output_retries": sum(int(row["invalid_output_retries"]) for row in selected),
    }
    summary["passes_training_gate"] = bool(
        valid == VALID_REQUIRED
        and summary["harmonic_speedup"] >= HARMONIC_REQUIRED
        and summary["minimum_speedup"] >= MINIMUM_REQUIRED
        and summary["invalid_output_retries"] == 0
    )
    return summary


def select_arm(summaries: list[dict[str, object]], arm: str) -> dict[str, object]:
    arm_rows = [row for row in summaries if row["arm"] == arm]
    eligible = [row for row in arm_rows if row["passes_training_gate"]]
    pool = eligible if eligible else arm_rows
    selected = min(
        pool,
        key=lambda row: (
            -int(row["valid"]),
            -float(row["harmonic_speedup"]),
            -float(row["minimum_speedup"]),
            float(row["maximum_absolute_error"]),
            str(row["candidate"]),
        ),
    )
    return {
        "arm": arm,
        "passes_training_gate": bool(eligible),
        "selected": selected,
        "eligible_candidate_count": len(eligible),
        "candidate_count": len(arm_rows),
        "discovery_cost_total_candidate_s": sum(float(row["total_candidate_s"]) for row in arm_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.input.rglob("train-shard-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    expected_candidates = sum(len(candidates) for candidates in CANDIDATES_BY_ARM.values())
    expected_rows = expected_candidates * 100
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} candidate rows, received {len(rows)}")

    summaries = [
        summarise(rows, arm, candidate)
        for arm, candidates in CANDIDATES_BY_ARM.items()
        for candidate in candidates
    ]
    arm_results = {arm: select_arm(summaries, arm) for arm in CANDIDATES_BY_ARM}
    v4 = arm_results["v4_full"]["selected"]
    no_transfer = arm_results["v4_no_transfer"]["selected"]
    random = arm_results["random_search"]["selected"]
    template = arm_results["template_synthesis"]["selected"]
    v3 = arm_results["v3_compatible"]["selected"]

    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 1,
        "task": "communicability",
        "revision": 1,
        "candidate_source_sha256": sha256(TASK_DIR / "candidates.py"),
        "training_runner_sha256": sha256(TASK_DIR / "train_shard.py"),
        "aggregator_sha256": sha256(TASK_DIR / "aggregate_train.py"),
        "requirements_sha256": sha256(TASK_DIR / "requirements.txt"),
        "train_manifest_name": "communicability_T100ms_n61_size100_train.jsonl",
        "train_manifest_tree_oid": "a906d00ab84131d86772b8ff78c8f0650ec53b60",
        "train_manifest_sha256": "c4495df9c72e78a4fecb79f88cd712036b587b3ccbec23f83bfc0918f5eb4c38",
        "expected_test_manifest_name": "communicability_T100ms_n61_size100_test.jsonl",
        "expected_test_manifest_tree_oid": "48efda1a3b334b13efb874355985d9f1d87291f2",
        "frozen_gate": {
            "valid_required": VALID_REQUIRED,
            "harmonic_speedup_minimum": HARMONIC_REQUIRED,
            "minimum_speedup": MINIMUM_REQUIRED,
            "invalid_output_retries": 0
        },
        "candidate_count": expected_candidates,
        "raw_record_count": len(rows),
        "all_candidates": summaries,
        "arms": arm_results,
        "task_training_status": "passed" if arm_results["v4_full"]["passes_training_gate"] else "failed",
        "architecture_comparison": {
            "v4_full_harmonic": v4["harmonic_speedup"],
            "v4_no_transfer_harmonic": no_transfer["harmonic_speedup"],
            "random_search_harmonic": random["harmonic_speedup"],
            "template_synthesis_harmonic": template["harmonic_speedup"],
            "v3_compatible_harmonic": v3["harmonic_speedup"],
            "v4_minus_v3_harmonic": float(v4["harmonic_speedup"]) - float(v3["harmonic_speedup"]),
            "v4_minus_no_transfer_harmonic": float(v4["harmonic_speedup"]) - float(no_transfer["harmonic_speedup"]),
            "v4_beats_v3_on_training": bool(
                arm_results["v4_full"]["passes_training_gate"]
                and (not arm_results["v3_compatible"]["passes_training_gate"] or float(v4["harmonic_speedup"]) > float(v3["harmonic_speedup"]))
            ),
            "transfer_advantage_observed": bool(
                float(v4["harmonic_speedup"]) > float(no_transfer["harmonic_speedup"]) * 1.02
                or arm_results["v4_full"]["passes_training_gate"] != arm_results["v4_no_transfer"]["passes_training_gate"]
            )
        },
        "test_manifest_opened": False,
        "test_payloads_opened": 0,
        "reports_opened": False,
        "public_solvers_opened": False,
        "training_revision_consumed": True,
        "human_task_specific_solver_design": False
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "training-results.jsonl").write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["arm"]), str(row["candidate"])))) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "task_training_status": report["task_training_status"],
        "selected_by_arm": {arm: result["selected"] for arm, result in arm_results.items()},
        "architecture_comparison": report["architecture_comparison"]
    }, indent=2))


if __name__ == "__main__":
    main()
