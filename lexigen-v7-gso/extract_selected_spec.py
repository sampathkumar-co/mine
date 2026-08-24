from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

HF_DATASET = "gso-bench/gso"
HF_REVISION = "c2e4f1a58427cccd15e0e542f136bd204fb19284"
PARQUET_PATH = "data/test-00000-of-00001.parquet"
ALLOWED_COLUMNS = ("instance_id", "prob_script", "tests")
FORBIDDEN_COLUMNS = {"opt_commit", "gt_commit_message", "gt_diff", "hints_text"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    parquet_url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{PARQUET_PATH}?download=true"
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_http_metadata_cache=false")
    quoted = ", ".join(f'"{name}"' for name in ALLOWED_COLUMNS)
    rows = con.execute(f"SELECT {quoted} FROM read_parquet(?) WHERE instance_id = ?", [parquet_url, args.instance_id]).fetchall()
    names = [d[0] for d in con.description]
    if tuple(names) != ALLOWED_COLUMNS:
        raise RuntimeError(f"unexpected projection: {names}")
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one selected GSO row, got {len(rows)}")
    record = dict(zip(names, rows[0]))
    if any(name in record for name in FORBIDDEN_COLUMNS):
        raise RuntimeError("forbidden expert metadata entered selected-task extractor")
    tests = record["tests"]
    if not isinstance(tests, list):
        tests = list(tests) if tests is not None else []
    report = {
        "instance_id": str(record["instance_id"]),
        "prob_script": str(record["prob_script"]),
        "tests": [str(x) for x in tests],
        "prob_script_sha256": hashlib.sha256(str(record["prob_script"]).encode()).hexdigest(),
        "test_sha256": [hashlib.sha256(str(x).encode()).hexdigest() for x in tests],
        "projection": list(ALLOWED_COLUMNS),
        "expert_opt_commit_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "task-spec.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"instance_id": report["instance_id"], "prob_script_sha256": report["prob_script_sha256"], "test_count": len(report["tests"]), "test_sha256": report["test_sha256"], "forbidden_values_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
