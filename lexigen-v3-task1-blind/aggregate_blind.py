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
    rows.sort(key=lambda row: int(row["index"]))
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind rows, received {len(rows)}")
    if [int(row["index"]) for row in rows] != list(range(1, 101)):
        raise RuntimeError("blind row indices are incomplete or duplicated")
    if any(int(row["candidate_executions"]) != 1 for row in rows):
        raise RuntimeError("candidate execution count changed")
    if any(int(row["reference_executions"]) != 1 for row in rows):
        raise RuntimeError("reference execution count changed")

    speeds = [float(row["speedup"]) for row in rows]
    valid = sum(bool(row["valid"]) for row in rows)
    exceptions = sum(row["failure_reason"] not in (None, "feasibility_or_objective") for row in rows)
    retries = sum(int(row["candidate_retries"]) for row in rows)
    summary = {
        "candidate": "pruned_hybrid6",
        "count": 100,
        "valid": valid,
        "exceptions": exceptions,
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(float(row["candidate_s"]) for row in rows),
        "median_reference_s": statistics.median(float(row["reference_s"]) for row in rows),
        "maximum_cvar_violation": max(float(row["cvar"]) - float(row["kappa"]) for row in rows),
        "maximum_objective_ratio": max(float(row["objective_ratio"]) for row in rows),
    }
    passes = bool(
        valid == 100
        and summary["harmonic_speedup"] >= 1.50
        and summary["minimum_speedup"] >= 1.05
        and retries == 0
    )
    manifest_hashes = sorted({str(row["manifest_content_sha256"]) for row in rows})
    if len(manifest_hashes) != 1:
        raise RuntimeError("shards observed different blind manifest content")

    report = {
        "status": "blind_win" if passes else "blind_failure",
        "task": "cvar_projection",
        "candidate_revision": 3,
        "selected": summary,
        "blind_manifest_name": "cvar_projection_T100ms_n9_size100_test.jsonl",
        "blind_manifest_git_blob_sha1": "194db2639d41ee0c91b2f171932e7984ad4aed3a",
        "blind_manifest_content_sha256": manifest_hashes[0],
        "candidate_sha256": sha256(TASK_DIR / "candidate.py"),
        "blind_runner_sha256": sha256(TASK_DIR / "blind_shard.py"),
        "blind_aggregator_sha256": sha256(TASK_DIR / "aggregate_blind.py"),
        "blind_lock_sha256": sha256(TASK_DIR / "BLIND_LOCK.json"),
        "passes_blind_gate": passes,
        "candidate_retries": retries,
        "candidate_executions_per_record": 1,
        "reference_executions_per_record": 1,
        "test_data_opened_only_after_lock": True,
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
