from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

CANDIDATE = "block_lloyd"
EXPECTED_TREE_OID = "f5f3fded57e9ca0fc4d1369477fd2176eeac15e2"


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
    for path in sorted(args.input.rglob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    rows.sort(key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in rows] != list(range(1, 101)):
        raise RuntimeError("blind rows are not exactly records 1..100")
    if any(str(row["candidate"]) != CANDIDATE for row in rows):
        raise RuntimeError("candidate identity changed")
    if any(int(row.get("candidate_executions", 0)) != 1 for row in rows):
        raise RuntimeError("candidate execution count changed")
    if any(int(row.get("reference_executions", 0)) != 1 for row in rows):
        raise RuntimeError("reference execution count changed")
    if any(str(row.get("test_manifest_tree_oid")) != EXPECTED_TREE_OID for row in rows):
        raise RuntimeError("test object identity changed")

    manifest_hashes = sorted({str(row["manifest_sha256"]) for row in rows})
    if len(manifest_hashes) != 1:
        raise RuntimeError(f"multiple test manifest hashes observed: {manifest_hashes}")
    speedups = [float(row["speedup"]) for row in rows]
    valid = sum(bool(row["valid"]) for row in rows)
    exceptions = sum(row.get("candidate_s") is None for row in rows)
    report = {
        "campaign": "LEXIGEN World-Proof Campaign v3",
        "task_index": 2,
        "task": "kmeans",
        "candidate_revision": 1,
        "candidate": CANDIDATE,
        "count": 100,
        "valid": valid,
        "exceptions": exceptions,
        "harmonic_speedup": harmonic(speedups),
        "median_speedup": statistics.median(speedups),
        "minimum_speedup": min(speedups),
        "maximum_speedup": max(speedups),
        "median_candidate_s": statistics.median(float(row["candidate_s"]) for row in rows),
        "median_reference_s": statistics.median(float(row["reference_s"]) for row in rows),
        "maximum_loss_ratio": max(float(row["loss_ratio"]) for row in rows),
        "median_loss_ratio": statistics.median(float(row["loss_ratio"]) for row in rows),
        "test_manifest_tree_oid": EXPECTED_TREE_OID,
        "test_manifest_sha256": manifest_hashes[0],
        "candidate_retries": 0,
        "passes_blind_gate": bool(
            valid == 100
            and exceptions == 0
            and harmonic(speedups) >= 1.50
            and min(speedups) >= 1.05
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "blind-results.jsonl"
    summary_path = args.output / "blind-summary.json"
    results_path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "verification.json").write_text(
        json.dumps({
            "results_sha256": sha256(results_path),
            "summary_sha256": sha256(summary_path),
            "indices_exact": True,
            "candidate_identity_exact": True,
            "one_execution_each": True,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["passes_blind_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
