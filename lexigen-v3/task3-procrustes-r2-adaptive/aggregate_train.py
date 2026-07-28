from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harmonic(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.input.rglob("train-shard-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    candidates = sorted({str(row["candidate"]) for row in rows})
    if len(rows) != len(candidates) * 100:
        raise RuntimeError(f"unexpected row count: {len(rows)}")

    summaries: list[dict[str, object]] = []
    for candidate in candidates:
        selected = sorted((row for row in rows if row["candidate"] == candidate), key=lambda row: int(row["index"]))
        if [int(row["index"]) for row in selected] != list(range(1, 101)):
            raise RuntimeError(f"{candidate} does not contain records 1..100")
        speeds = [float(row["speedup"]) for row in selected]
        summaries.append({
            "candidate": candidate,
            "count": 100,
            "valid": sum(bool(row["valid"]) for row in selected),
            "exceptions": sum(row.get("candidate_s") is None for row in selected),
            "harmonic_speedup": harmonic(speeds),
            "median_speedup": statistics.median(speeds),
            "minimum_speedup": min(speeds),
            "maximum_speedup": max(speeds),
            "median_candidate_s": statistics.median(float(row["candidate_s"]) for row in selected),
            "median_reference_s": statistics.median(float(row["reference_s"]) for row in selected),
            "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in selected),
            "fallback_count": sum(bool(row["fallback_expected"]) for row in selected),
            "minimum_condition_ratio": min(float(row["condition_ratio"]) for row in selected),
            "maximum_condition_ratio": max(float(row["condition_ratio"]) for row in selected),
        })

    eligible = [
        item for item in summaries
        if item["valid"] == 100
        and item["exceptions"] == 0
        and item["harmonic_speedup"] >= 1.50
        and item["minimum_speedup"] >= 1.05
    ]
    eligible.sort(key=lambda item: (-float(item["harmonic_speedup"]), -float(item["minimum_speedup"]), float(item["maximum_absolute_error"]), str(item["candidate"])))
    selected = eligible[0] if eligible else max(
        summaries,
        key=lambda item: (int(item["valid"]), float(item["harmonic_speedup"]), float(item["minimum_speedup"])),
    )
    report = {
        "campaign": "LEXIGEN autonomous unseen-task discovery v3",
        "task_index": 3,
        "task": "procrustes",
        "candidate_revision": 2,
        "architecture": "condition_aware_precision_escalation",
        "condition_threshold": 4.76837158203125e-7,
        "candidate_family_sha256": sha256(TASK_DIR / "candidates.py"),
        "runner_sha256": sha256(TASK_DIR / "train_shard.py"),
        "aggregator_sha256": sha256(TASK_DIR / "aggregate_train.py"),
        "train_manifest_name": "procrustes_T100ms_n585_size100_train.jsonl",
        "train_manifest_tree_oid": "289fc7dd4db5638412b7de37c1703a576754f65a",
        "train_manifest_sha256": "61a55f5731fa9ba6e3b83b5cce05bb91f5d82edd81c1ea273ab8b7681bbef59f",
        "expected_test_manifest_name": "procrustes_T100ms_n585_size100_test.jsonl",
        "expected_test_manifest_tree_oid": "fe35d3885bf346ceb089d3c0e233ef69c0f1c260",
        "selected": selected,
        "all_candidates": summaries,
        "passes_training_gate": bool(eligible),
        "test_data_opened": False,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "train-results.jsonl"
    summary_path = args.output / "train-summary.json"
    results_path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["candidate"])))) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "verification.json").write_text(
        json.dumps({"results_sha256": sha256(results_path), "summary_sha256": sha256(summary_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
