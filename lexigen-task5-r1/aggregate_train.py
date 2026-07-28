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
    expected_rows = len(candidates) * 100
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, received {len(rows)}")

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
                row["failure_reason"] not in (None, "value_mismatch")
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
            "maximum_absolute_error": max(
                float(row["maximum_absolute_error"]) for row in selected
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
        "task": "outer_product",
        "candidate_revision": 1,
        "candidate_family_sha256": sha256(TASK_DIR / "candidates.py"),
        "native_source_sha256": sha256(TASK_DIR / "native_outer.c"),
        "runner_sha256": sha256(TASK_DIR / "train_shard.py"),
        "train_manifest_name": "outer_product_T100ms_n10630_size100_train.jsonl",
        "train_manifest_tree_oid": "c4b2923db0aca1ca679ce5430c0c934c646bf947",
        "train_manifest_lfs_sha256": "a910e76b4058137a8d69213d72541a6ea55410a494c69f52058b63b22b020372",
        "expected_test_manifest_name": "outer_product_T100ms_n10630_size100_test.jsonl",
        "expected_test_manifest_tree_oid": "a4ab4cb2fc915f91a4cdc057ae078d652585e667",
        "expected_test_manifest_lfs_sha256": "7c96c6cb4b391f0268625869217c50a5948eddd15a44d0d5f650e8f2adf04538",
        "selected": selected,
        "all_candidates": summaries,
        "passes_training_gate": bool(eligible),
        "test_data_opened": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train-summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (args.output / "train-results.jsonl").write_text(
        "\n".join(
            json.dumps(row, separators=(",", ":"))
            for row in sorted(rows, key=lambda row: (int(row["index"]), str(row["candidate"])))
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
