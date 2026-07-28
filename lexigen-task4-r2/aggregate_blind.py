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
    for path in sorted(args.input.rglob("blind-shard-*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind rows, received {len(rows)}")
    ordered = sorted(rows, key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in ordered] != list(range(1, 101)):
        raise RuntimeError("blind result indices are not exactly 1 through 100")
    if any(int(row.get("candidate_executions", 0)) != 1 for row in ordered):
        raise RuntimeError("one or more blind records did not use exactly one candidate execution")
    if any(str(row["candidate"]) != "native_parallel4" for row in ordered):
        raise RuntimeError("unexpected candidate in blind evidence")

    speeds = [float(row["speedup"]) for row in ordered]
    valid_count = sum(bool(row["valid"]) for row in ordered)
    exceptions = sum(row["failure_reason"] not in (None, "output_mismatch") for row in ordered)
    summary = {
        "candidate": "native_parallel4",
        "count": 100,
        "valid": valid_count,
        "exceptions": exceptions,
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(
            float(row["candidate_s"])
            for row in ordered
            if row["candidate_s"] is not None
        ),
        "median_reference_s": statistics.median(float(row["reference_s"]) for row in ordered),
    }
    passes = bool(
        valid_count == 100
        and summary["harmonic_speedup"] >= 1.50
        and summary["minimum_speedup"] >= 1.05
    )
    report = {
        "status": "blind_win" if passes else "failed_blind_gate",
        "task": "chacha_encryption",
        "candidate_revision": 2,
        "selected": summary,
        "blind_manifest_name": "chacha_encryption_T100ms_n197380_size100_test.jsonl",
        "blind_manifest_git_oid": "b6903231c88291a8eff777a519a9af0736148a0f",
        "candidate_sha256": sha256(TASK_DIR / "candidate.py"),
        "native_source_sha256": sha256(TASK_DIR / "native_chacha.c"),
        "blind_runner_sha256": sha256(TASK_DIR / "blind_shard.py"),
        "blind_aggregator_sha256": sha256(TASK_DIR / "aggregate_blind.py"),
        "blind_lock_sha256": sha256(TASK_DIR / "BLIND_LOCK.json"),
        "passes_blind_gate": passes,
        "candidate_retries": 0,
        "test_data_opened_only_after_lock": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "blind-summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (args.output / "blind-results.jsonl").write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in ordered) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not passes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
