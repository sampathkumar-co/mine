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
    for path in sorted(args.input.rglob("shard-*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    candidates = sorted({str(row["candidate"]) for row in rows})
    expected = len(candidates) * 100
    if not candidates or len(rows) != expected:
        raise RuntimeError(
            f"expected {expected} rows for {len(candidates)} candidates, received {len(rows)}"
        )

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
            "exceptions": sum(
                row["failure_reason"] not in (None, "output_mismatch")
                for row in selected
            ),
            "harmonic_speedup": harmonic(speeds),
            "median_speedup": statistics.median(speeds),
            "minimum_speedup": min(speeds),
            "maximum_speedup": max(speeds),
            "median_candidate_s": statistics.median(
                float(row["candidate_s"])
                for row in selected
                if row["candidate_s"] is not None
            ),
            "median_reference_s": statistics.median(
                float(row["reference_s"]) for row in selected
            ),
        })

    eligible = [
        summary
        for summary in summaries
        if summary["valid"] == 100
        and float(summary["harmonic_speedup"]) >= 1.50
        and float(summary["minimum_speedup"]) >= 1.05
    ]
    eligible.sort(
        key=lambda row: (
            -float(row["harmonic_speedup"]),
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
        "task": "chacha_encryption",
        "candidate_revision": 2,
        "candidate_family_sha256": sha256(TASK_DIR / "candidates.py"),
        "runner_sha256": sha256(TASK_DIR / "train_shard.py"),
        "train_manifest_name": "chacha_encryption_T100ms_n197380_size100_train.jsonl",
        "train_manifest_git_oid": "9eb2847936888c7843b54dc94dd8e225a4fc5850",
        "expected_test_manifest_name": "chacha_encryption_T100ms_n197380_size100_test.jsonl",
        "expected_test_manifest_git_oid": "b6903231c88291a8eff777a519a9af0736148a0f",
        "selected": selected,
        "all_candidates": summaries,
        "passes_training_gate": bool(eligible),
        "test_data_opened": False,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train-summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output / "train-results.jsonl").write_text(
        "\n".join(
            json.dumps(row, separators=(",", ":"))
            for row in sorted(
                rows,
                key=lambda row: (int(row["index"]), str(row["candidate"])),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
