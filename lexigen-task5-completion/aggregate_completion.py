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


def read_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = read_rows(args.original)
    completion = read_rows(args.completion)
    if len(original) != 80:
        raise RuntimeError(f"expected 80 original rows, received {len(original)}")
    if len(completion) != 20:
        raise RuntimeError(f"expected 20 completion rows, received {len(completion)}")
    if {int(row["shard"]) for row in completion} != {2, 7}:
        raise RuntimeError("completion rows are not restricted to original HTTP-429 shards")

    rows = original + completion
    rows.sort(key=lambda row: int(row["index"]))
    indices = [int(row["index"]) for row in rows]
    if indices != list(range(1, 101)):
        raise RuntimeError("combined blind rows are incomplete or duplicated")
    if any(int(row.get("candidate_retries", 0)) != 0 for row in rows):
        raise RuntimeError("candidate retry detected")
    if any(int(row.get("candidate_executions", 1)) != 1 for row in rows):
        raise RuntimeError("candidate execution count changed")
    if any(int(row.get("reference_executions", 1)) != 1 for row in rows):
        raise RuntimeError("reference execution count changed")

    speeds = [float(row["speedup"]) for row in rows]
    valid = sum(bool(row["valid"]) for row in rows)
    exceptions = sum(row["failure_reason"] not in (None, "value_mismatch") for row in rows)
    summary = {
        "candidate": "native_parallel8",
        "count": 100,
        "valid": valid,
        "exceptions": exceptions,
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(float(row["candidate_s"]) for row in rows),
        "median_reference_s": statistics.median(float(row["reference_s"]) for row in rows),
        "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in rows),
    }
    passes = bool(
        valid == 100
        and summary["harmonic_speedup"] >= 1.50
        and summary["minimum_speedup"] >= 1.05
        and exceptions == 0
    )
    report = {
        "status": "blind_win" if passes else "blind_failure",
        "task": "outer_product",
        "candidate_revision": 1,
        "selected": summary,
        "blind_manifest_name": "outer_product_T100ms_n10630_size100_test.jsonl",
        "blind_manifest_lfs_sha256": "7c96c6cb4b391f0268625869217c50a5948eddd15a44d0d5f650e8f2adf04538",
        "original_blind_run_id": 30355933562,
        "original_completed_records": 80,
        "original_failed_shards": [2, 7],
        "original_failure_classification": "HTTP 429 before manifest bytes and before candidate execution",
        "completion_shards": [2, 7],
        "completion_records": 20,
        "candidate_changed": False,
        "thresholds_changed": False,
        "successful_records_rerun": False,
        "candidate_executions_per_record": 1,
        "candidate_retries_after_invalid_output": 0,
        "candidate_sha256": sha256(TASK_DIR / "candidate.py"),
        "native_source_sha256": sha256(TASK_DIR / "native_outer.c"),
        "completion_runner_sha256": sha256(TASK_DIR / "blind_completion_shard.py"),
        "completion_aggregator_sha256": sha256(TASK_DIR / "aggregate_completion.py"),
        "completion_lock_sha256": sha256(TASK_DIR / "COMPLETION_LOCK.json"),
        "passes_blind_gate": passes,
        "scientific_interpretation": "The 100-case blind result combines 80 records from the original locked run with 20 records from an infrastructure-only completion of two shards that had received HTTP 429 before downloading manifest bytes or executing the candidate. No successful record was rerun, and every blind record has exactly one candidate and one reference execution.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "blind-summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "blind-results.jsonl").write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not passes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
