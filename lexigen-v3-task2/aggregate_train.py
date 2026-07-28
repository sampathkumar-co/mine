from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

CANDIDATES = ("block_elkan", "block_lloyd", "block_numpy")


def harmonic(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.input.rglob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(rows) != 300:
        raise RuntimeError(f"expected 300 candidate rows, received {len(rows)}")

    summaries = []
    for name in CANDIDATES:
        selected = sorted((row for row in rows if row["candidate"] == name), key=lambda row: int(row["index"]))
        if [int(row["index"]) for row in selected] != list(range(1, 101)):
            raise RuntimeError(f"{name} does not contain exactly records 1..100")
        speedups = [float(row["speedup"]) for row in selected]
        summary = {
            "candidate": name,
            "count": len(selected),
            "valid": sum(bool(row["valid"]) for row in selected),
            "exceptions": sum(row.get("candidate_s") is None for row in selected),
            "harmonic_speedup": harmonic(speedups),
            "median_speedup": statistics.median(speedups),
            "minimum_speedup": min(speedups),
            "maximum_speedup": max(speedups),
            "median_candidate_s": statistics.median(float(row["candidate_s"]) for row in selected),
            "median_reference_s": statistics.median(float(row["reference_s"]) for row in selected),
            "maximum_inertia_ratio": max(float(row["inertia_ratio"]) for row in selected),
        }
        summaries.append(summary)

    ranked = sorted(
        summaries,
        key=lambda item: (
            int(item["valid"]) == 100,
            float(item["harmonic_speedup"]),
            float(item["minimum_speedup"]),
            -len(str(item["candidate"])),
        ),
        reverse=True,
    )
    winner = ranked[0]
    passes = bool(
        winner["valid"] == 100
        and winner["exceptions"] == 0
        and winner["harmonic_speedup"] >= 1.50
        and winner["minimum_speedup"] >= 1.05
    )
    result = {
        "status": "passed_training_gate" if passes else "failed_training_gate",
        "task": "kmeans",
        "candidate_revision": 1,
        "selected": winner,
        "all_candidates": summaries,
        "passes_training_gate": passes,
        "test_data_opened": false,
        "candidate_executions_repeated_during_aggregation": 0,
        "source_training_workflow_run_id": 30362180027
    }

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "train-results.jsonl"
    summary_path = args.output / "train-summary.json"
    results_path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["candidate"])))) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "verification.json").write_text(
        json.dumps({"results_sha256": sha256(results_path), "summary_sha256": sha256(summary_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
