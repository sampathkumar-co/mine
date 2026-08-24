from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import duckdb

HF_DATASET = "gso-bench/gso"
HF_REVISION = "c2e4f1a58427cccd15e0e542f136bd204fb19284"
PARQUET_PATH = "data/test-00000-of-00001.parquet"
EXPECTED_PARQUET_SHA256 = "bda458a7b5437c252f6cefdbc896f5f2868de51479e7a221ceda3f3ab74879bc"
EXPECTED_XET_HASH = "59cb48b81dd9033151c35123d78dea99ad1959ed33f737c8523f916643402373"
EXPECTED_ROW_COUNT = 102
SEED = "LEXIGEN-V7-GSO-REAL-TRANSFER-2026-08-24-A"
TASK_COUNT = 6
SAFE_COLUMNS = (
    "instance_id",
    "repo",
    "base_commit",
    "api",
    "created_at",
    "arch",
    "instance_image_tag",
)
FORBIDDEN_COLUMNS = {
    "opt_commit",
    "gt_commit_message",
    "gt_diff",
    "hints_text",
    "prob_script",
    "tests",
    "setup_commands",
    "install_commands",
}


def fetch_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LEXIGEN-v7-gso-selector", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def verify_remote_file_metadata() -> dict[str, object]:
    url = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/{HF_REVISION}/data?recursive=false&expand=true"
    payload = fetch_json(url)
    if not isinstance(payload, list):
        raise RuntimeError("Hugging Face tree metadata is not a list")
    entry = next((x for x in payload if isinstance(x, dict) and x.get("path") == PARQUET_PATH), None)
    if not isinstance(entry, dict):
        raise RuntimeError("pinned GSO parquet is absent from the pinned dataset revision")
    lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
    oid = str(lfs.get("oid", ""))
    xet_hash = str(entry.get("xetHash") or entry.get("xet_hash") or "")
    if oid and oid.startswith("sha256:"):
        oid = oid.split(":", 1)[1]
    if oid and oid != EXPECTED_PARQUET_SHA256:
        raise RuntimeError(f"pinned parquet sha mismatch: {oid}")
    if xet_hash and xet_hash != EXPECTED_XET_HASH:
        raise RuntimeError(f"pinned xet hash mismatch: {xet_hash}")
    # Some Hub API versions omit lfs/xet details for Xet-backed entries. The immutable
    # dataset revision + path remain hard-pinned; the expected hashes are still recorded
    # in the selection evidence and lock.
    return {
        "metadata_url": url,
        "path": PARQUET_PATH,
        "entry_size": entry.get("size"),
        "reported_lfs_sha256": oid or None,
        "reported_xet_hash": xet_hash or None,
        "expected_parquet_sha256": EXPECTED_PARQUET_SHA256,
        "expected_xet_hash": EXPECTED_XET_HASH,
        "revision": HF_REVISION,
        "file_contents_downloaded_for_identity_check": False,
    }


def read_safe_columns() -> tuple[list[dict[str, str]], str]:
    parquet_url = (
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/"
        f"{HF_REVISION}/{PARQUET_PATH}?download=true"
    )
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_http_metadata_cache=false")
    quoted = ", ".join(f'"{name}"' for name in SAFE_COLUMNS)
    # Projection pushdown is the privacy boundary: this SQL asks the Parquet reader only
    # for safe metadata columns. Forbidden expert/test columns are never referenced.
    sql = f"SELECT {quoted} FROM read_parquet(?)"
    rows = con.execute(sql, [parquet_url]).fetchall()
    names = [d[0] for d in con.description]
    if tuple(names) != SAFE_COLUMNS:
        raise RuntimeError(f"unexpected projected columns: {names}")
    if len(rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError(f"expected {EXPECTED_ROW_COUNT} GSO rows, got {len(rows)}")
    result = []
    for values in rows:
        row = {name: ("" if value is None else str(value)) for name, value in zip(names, values)}
        if any(key in row for key in FORBIDDEN_COLUMNS):
            raise RuntimeError("forbidden GSO column entered selector memory")
        result.append(row)
    return result, parquet_url


def selection_score(instance_id: str) -> str:
    return hashlib.sha256(f"{SEED}\0{instance_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("selection-evidence"))
    args = parser.parse_args()

    file_meta = verify_remote_file_metadata()
    rows, parquet_url = read_safe_columns()
    inventory = []
    for row in rows:
        instance_id = row["instance_id"]
        repo = row["repo"]
        if not instance_id or not repo:
            raise RuntimeError("GSO safe metadata contains blank instance_id/repo")
        inventory.append({**row, "selection_score": selection_score(instance_id)})
    inventory.sort(key=lambda r: (r["selection_score"], r["instance_id"]))

    selected = []
    repos: set[str] = set()
    for row in inventory:
        if row["repo"] in repos:
            continue
        selected.append(row)
        repos.add(row["repo"])
        if len(selected) == TASK_COUNT:
            break
    if len(selected) != TASK_COUNT or len(repos) != TASK_COUNT:
        raise RuntimeError(f"could not select {TASK_COUNT} distinct repositories")

    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "campaign": "LEXIGEN V7 GSO Real Causal-Transfer Pilot R1",
        "stage": "safe_metadata_only_selection",
        "selection_seed": SEED,
        "gso_harness_commit": "7074865b48123b30a2e61d7dbc4887fcd990e681",
        "hf_dataset": HF_DATASET,
        "hf_revision": HF_REVISION,
        "parquet_identity": file_meta,
        "projected_parquet_url": parquet_url,
        "safe_columns": list(SAFE_COLUMNS),
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "projection_method": "DuckDB remote Parquet projection pushdown over HTTP range requests",
        "inventory_row_count": len(inventory),
        "inventory_sha256": inventory_hash,
        "selected_count": len(selected),
        "selected_repo_count": len(repos),
        "selected": selected,
        "forbidden_column_values_accessed": False,
        "expert_opt_commit_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
        "performance_test_spec_accessed": False,
        "correctness_tests_accessed": False,
        "prior_submissions_accessed": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
