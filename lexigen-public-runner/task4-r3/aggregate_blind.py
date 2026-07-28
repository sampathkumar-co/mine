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
    if len({int(row["index"]) for row in rows}) != 100:
        raise RuntimeError("blind rows do not cover 100 unique indices")

    speeds = [float(row["speedup"]) for row in rows]
    summary = {
        "candidate": "direct4",
        "count": 100,
        "valid": sum(bool(row["valid"]) for row in rows),
        "exceptions": sum(
            row["failure_reason"] not in (None, "output_mismatch") for row in rows
        ),
        "harmonic_speedup": harmonic(speeds),
        "median_speedup": statistics.median(speeds),
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "median_candidate_s": statistics.median(
            float(row["candidate_s"]) for row in rows if row["candidate_s"] is not None
        ),
        "median_reference_s": statistics.median(float(row["reference_s"]) for row in rows),
    }
    passes = bool(
        summary["valid"] == 100
        and float(summary["harmonic_speedup"]) >= 1.50
        and float(summary["minimum_speedup"]) >= 1.05
    )
    report = {
        "task": "chacha_encryption",
        "candidate_revision": 3,
        "blind_evaluation": true,
        "selected_candidate": "direct4",
        "candidate_family_sha256": sha256(TASK_DIR / "candidates.py"),
        "selected_solver_sha256": sha256(TASK_DIR / "selected_solver.py"),
        "blind_runner_sha256": sha256(TASK_DIR / "blind_shard.py"),
        "test_manifest_name": "chacha_encryption_T100ms_n197380_size100_test.jsonl",
        "test_manifest_git_oid": "b6903231c88291a8eff777a519a9af0736148a0f",
        "result": summary,
        "passes_blind_gate": passes,
        "no_candidate_revisions_after_test_access": true,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "blind-summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output / "blind-results.jsonl").write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not passes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
