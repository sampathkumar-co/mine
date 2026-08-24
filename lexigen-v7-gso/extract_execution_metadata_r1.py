from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

HF_DATASET = "gso-bench/gso"
HF_REVISION = "c2e4f1a58427cccd15e0e542f136bd204fb19284"
PARQUET_PATH = "data/test-00000-of-00001.parquet"
ALLOWED_COLUMNS = ("instance_id", "repo", "install_commands")
FORBIDDEN_COLUMNS = {
    "opt_commit",
    "gt_commit_message",
    "gt_diff",
    "hints_text",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    parquet_url = (
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/"
        f"{HF_REVISION}/{PARQUET_PATH}?download=true"
    )
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_http_metadata_cache=false")
    quoted = ", ".join(f'"{name}"' for name in ALLOWED_COLUMNS)
    placeholders = ",".join("?" for _ in args.instance_id)
    sql = (
        f"SELECT {quoted} FROM read_parquet(?) "
        f"WHERE instance_id IN ({placeholders}) ORDER BY instance_id"
    )
    rows = con.execute(sql, [parquet_url, *args.instance_id]).fetchall()
    names = [d[0] for d in con.description]
    if tuple(names) != ALLOWED_COLUMNS:
        raise RuntimeError(f"unexpected projection: {names}")
    if len(rows) != len(set(args.instance_id)):
        raise RuntimeError(f"expected {len(set(args.instance_id))} rows, got {len(rows)}")

    records = []
    for row in rows:
        record = dict(zip(names, row))
        if any(name in record for name in FORBIDDEN_COLUMNS):
            raise RuntimeError("forbidden expert metadata entered execution projector")
        cmds = record["install_commands"]
        if cmds is None:
            cmds = []
        elif not isinstance(cmds, list):
            cmds = list(cmds)
        records.append({
            "instance_id": str(record["instance_id"]),
            "repo": str(record["repo"]),
            "install_commands": [str(x) for x in cmds if str(x).strip() != "git clean -xfd"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_revision": HF_REVISION,
        "projection": list(ALLOWED_COLUMNS),
        "records": records,
        "expert_opt_commit_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
