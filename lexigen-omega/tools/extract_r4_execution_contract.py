from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

HF_DATASET = "gso-bench/gso"
HF_REVISION = "c2e4f1a58427cccd15e0e542f136bd204fb19284"
PARQUET_PATH = "data/test-00000-of-00001.parquet"
ALLOWED_COLUMNS = (
    "instance_id",
    "repo",
    "base_commit",
    "api",
    "install_commands",
    "setup_commands",
    "prob_script",
    "tests",
)
FORBIDDEN_COLUMNS = {
    "opt_commit",
    "gt_commit_message",
    "gt_diff",
    "hints_text",
}


def normalize_commands(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        vals = value
    else:
        vals = list(value)
    return [str(x) for x in vals if str(x).strip() and str(x).strip() != "git clean -xfd"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--expected-repo", required=True)
    ap.add_argument("--expected-base", required=True)
    ap.add_argument("--expected-api", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    parquet_url = (
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/"
        f"{HF_REVISION}/{PARQUET_PATH}?download=true"
    )
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_http_metadata_cache=false")
    quoted = ", ".join(f'"{name}"' for name in ALLOWED_COLUMNS)
    rows = con.execute(
        f"SELECT {quoted} FROM read_parquet(?) WHERE instance_id = ?",
        [parquet_url, args.instance_id],
    ).fetchall()
    names = [d[0] for d in con.description]
    if tuple(names) != ALLOWED_COLUMNS:
        raise RuntimeError(f"unexpected projection: {names}")
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one target row, got {len(rows)}")
    record = dict(zip(names, rows[0]))
    if FORBIDDEN_COLUMNS.intersection(record):
        raise RuntimeError("forbidden expert metadata entered R4 projector")
    if str(record["repo"]) != args.expected_repo:
        raise RuntimeError("repo identity mismatch")
    if str(record["base_commit"]) != args.expected_base:
        raise RuntimeError("base identity mismatch")
    if str(record["api"]) != args.expected_api:
        raise RuntimeError("API identity mismatch")

    payload = {
        "project": "LEXIGEN OMEGA",
        "stage": "omega3_R4_execution_contract_after_proposal_prediction_scorer_freeze",
        "status": "permitted_execution_contract_projected",
        "dataset_revision": HF_REVISION,
        "projection": list(ALLOWED_COLUMNS),
        "instance_id": str(record["instance_id"]),
        "repo": str(record["repo"]),
        "base_commit": str(record["base_commit"]),
        "api": str(record["api"]),
        "install_commands": normalize_commands(record["install_commands"]),
        "setup_commands": normalize_commands(record["setup_commands"]),
        "prob_script": str(record["prob_script"] or ""),
        "tests": str(record["tests"] or ""),
        "expert_opt_commit_accessed": False,
        "expert_gt_commit_message_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
        "candidate_timing_accessed_before_contract": False,
        "candidate_outcome_accessed_before_contract": False,
        "claim_boundary": "Execution/performance contract projected only after R4 proposals, exact-sequence prediction, thresholds, and scorer were frozen.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "instance_id": payload["instance_id"],
        "repo": payload["repo"],
        "install_command_count": len(payload["install_commands"]),
        "setup_command_count": len(payload["setup_commands"]),
        "prob_script_bytes": len(payload["prob_script"].encode()),
        "tests_bytes": len(payload["tests"].encode()),
        "expert_fields_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
