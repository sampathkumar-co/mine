from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from selected_solvers import SELECTED_NAMES

VALID_REQUIRED = 100
HARMONIC_REQUIRED = 1.50
MINIMUM_REQUIRED = 1.05
EXPECTED_TEST_BLOB = "cee4dc02b968ffeac9fbfcd8d5157521ead9d3dc"


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
        raise RuntimeError(f"{arm}/{candidate} has {len(selected)} blind rows")
    if len({int(row["index"]) for row in selected}) != 100:
        raise RuntimeError(f"{arm}/{candidate} has duplicate blind records")
    if {str(row["blind_manifest_git_blob_sha1"]) for row in selected} != {EXPECTED_TEST_BLOB}:
        raise RuntimeError(f"{arm}/{candidate} blind manifest identity mismatch")
    manifest_sha256_values = {str(row["blind_manifest_sha256"]) for row in selected}
    if len(manifest_sha256_values) != 1:
        raise RuntimeError(f"{arm}/{candidate} observed inconsistent blind manifest SHA-256")

    speeds = [float(row["speedup"]) for row in selected]
    candidate_times = [float(row["candidate_s"]) for row in selected if row["candidate_s"] is not None]
    reference_times = [float(row["reference_s"]) for row in selected]
    valid = sum(bool(row["valid"]) for row in selected)
    result = {
        "arm": arm,
        "candidate": candidate,
        "count": 100,
        "valid": valid,
        "invalid_outputs": 100 - valid,
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(candidate_times),
        "median_reference_s": statistics.median(reference_times),
        "total_candidate_s": sum(candidate_times),
        "optimum_size_min": min(int(row["optimum_size"]) for row in selected),
        "optimum_size_max": max(int(row["optimum_size"]) for row in selected),
        "candidate_executions": sum(int(row["candidate_executions"]) for row in selected),
        "reference_executions_observed": len(selected),
        "invalid_output_retries": sum(int(row["invalid_output_retries"]) for row in selected),
        "test_manifest_git_blob_sha1": EXPECTED_TEST_BLOB,
        "test_manifest_sha256": next(iter(manifest_sha256_values)),
    }
    result["passes_blind_gate"] = bool(
        valid == VALID_REQUIRED
        and result["harmonic_speedup"] >= HARMONIC_REQUIRED
        and result["minimum_speedup"] >= MINIMUM_REQUIRED
        and result["invalid_output_retries"] == 0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.input.rglob("blind-shard-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    expected_rows = len(SELECTED_NAMES) * 100
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} blind rows, received {len(rows)}")

    arms = {
        arm: summarise(rows, arm, candidate)
        for arm, candidate in SELECTED_NAMES.items()
    }
    v4 = arms["v4_full"]
    no_transfer = arms["v4_no_transfer"]
    random = arms["random_search"]
    template = arms["template_synthesis"]
    v3 = arms["v3_compatible"]

    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 3,
        "task": "max_common_subgraph",
        "revision": 1,
        "blind_status": "passed" if v4["passes_blind_gate"] else "failed",
        "frozen_gate": {
            "valid_required": VALID_REQUIRED,
            "harmonic_speedup_minimum": HARMONIC_REQUIRED,
            "minimum_speedup": MINIMUM_REQUIRED,
            "invalid_output_retries": 0
        },
        "arms": arms,
        "architecture_comparison": {
            "v4_full_harmonic": v4["harmonic_speedup"],
            "v4_no_transfer_harmonic": no_transfer["harmonic_speedup"],
            "random_search_harmonic": random["harmonic_speedup"],
            "template_synthesis_harmonic": template["harmonic_speedup"],
            "v3_compatible_harmonic": v3["harmonic_speedup"],
            "v4_minus_v3_harmonic": float(v4["harmonic_speedup"]) - float(v3["harmonic_speedup"]),
            "v4_minus_template_harmonic": float(v4["harmonic_speedup"]) - float(template["harmonic_speedup"]),
            "v4_minus_random_harmonic": float(v4["harmonic_speedup"]) - float(random["harmonic_speedup"]),
            "v4_minus_no_transfer_harmonic": float(v4["harmonic_speedup"]) - float(no_transfer["harmonic_speedup"]),
            "v4_task_win": bool(v4["passes_blind_gate"]),
            "v4_beats_v3_by_task_win": bool(v4["passes_blind_gate"] and not v3["passes_blind_gate"]),
            "v4_beats_template_by_task_win": bool(v4["passes_blind_gate"] and not template["passes_blind_gate"]),
            "v4_beats_random_by_task_win": bool(v4["passes_blind_gate"] and not random["passes_blind_gate"]),
            "transfer_task_win_advantage": bool(v4["passes_blind_gate"] and not no_transfer["passes_blind_gate"]),
            "transfer_speed_advantage_over_2pct": bool(float(v4["harmonic_speedup"]) > float(no_transfer["harmonic_speedup"]) * 1.02),
        },
        "blind_record_count": 100,
        "successful_record_reruns": 0,
        "test_manifest_opened_after_lock_only": True,
        "reports_opened": False,
        "public_solvers_opened": False,
        "human_task_specific_solver_design": False
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "blind-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "blind-results.jsonl").write_text(
        "\n".join(
            json.dumps(row, separators=(",", ":"))
            for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["arm"])))
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"blind_status": report["blind_status"], "arms": arms, "architecture_comparison": report["architecture_comparison"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
