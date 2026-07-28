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
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    candidates = sorted({str(row["candidate"]) for row in rows})
    expected = len(candidates) * 100
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, received {len(rows)}")

    summaries: list[dict[str, object]] = []
    for candidate in candidates:
        selected = sorted(
            (row for row in rows if row["candidate"] == candidate),
            key=lambda row: int(row["index"]),
        )
        if len(selected) != 100:
            raise RuntimeError(f"{candidate} has {len(selected)} rows")
        speeds = [float(row["speedup"]) for row in selected]
        summaries.append({
            "candidate": candidate,
            "count": 100,
            "valid": sum(bool(row["valid"]) for row in selected),
            "exceptions": sum(row["failure_reason"] not in (None, "feasibility_or_objective") for row in selected),
            "harmonic_speedup": harmonic(speeds),
            "median_speedup": statistics.median(speeds),
            "minimum_speedup": min(speeds),
            "maximum_speedup": max(speeds),
            "median_candidate_s": statistics.median(float(row["candidate_s"]) for row in selected if row["candidate_s"] is not None),
            "median_reference_s": statistics.median(float(row["reference_s"]) for row in selected),
            "maximum_cvar_violation": max(float(row["cvar"]) - float(row["kappa"]) for row in selected),
            "maximum_objective_ratio": max(float(row["objective_ratio"]) for row in selected),
        })

    eligible = [
        summary
        for summary in summaries
        if summary["valid"] == 100
        and summary["harmonic_speedup"] >= 1.50
        and summary["minimum_speedup"] >= 1.05
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["harmonic_speedup"]),
            -float(row["minimum_speedup"]),
            str(row["candidate"]),
        )
    )
    selected = eligible[0] if eligible else max(
        summaries,
        key=lambda row: (
            int(row["valid"]),
            float(row["harmonic_speedup"]),
            float(row["minimum_speedup"]),
        ),
    )
    report = {
        "task": "cvar_projection",
        "candidate_revision": 1,
        "candidate_family_sha256": sha256(TASK_DIR / "candidates.py"),
        "runner_sha256": sha256(TASK_DIR / "train_shard.py"),
        "aggregator_sha256": sha256(TASK_DIR / "aggregate_train.py"),
        "train_manifest_name": "cvar_projection_T100ms_n9_size100_train.jsonl",
        "train_manifest_tree_oid": "3555d6025f202cb060e01473112717351bc2d829",
        "train_manifest_sha256": "f425911dee9a17939392f6f01fd4622a33ab8365c95e956e03c13351af099eb8",
        "expected_test_manifest_name": "cvar_projection_T100ms_n9_size100_test.jsonl",
        "expected_test_manifest_tree_oid": "194db2639d41ee0c91b2f171932e7984ad4aed3a",
        "selected": selected,
        "all_candidates": summaries,
        "passes_training_gate": bool(eligible),
        "test_data_opened": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train-summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "train-results.jsonl").write_text(
        "\n".join(
            json.dumps(row, separators=(",", ":"))
            for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["candidate"])))
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
