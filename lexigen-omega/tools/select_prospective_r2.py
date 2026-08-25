from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "evidence" / "PROSPECTIVE_R2_SELECTION_LOCK.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_json(url: str) -> object:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LEXIGEN-Omega-R2-selector", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def verify_remote_metadata(lock: dict) -> dict:
    dataset = lock["dataset"]
    url = (
        f"https://huggingface.co/api/datasets/{dataset['hf_dataset']}/tree/"
        f"{dataset['hf_revision']}/data?recursive=false&expand=true"
    )
    payload = fetch_json(url)
    if not isinstance(payload, list):
        raise RuntimeError("Hugging Face tree metadata is not a list")
    entry = next(
        (x for x in payload if isinstance(x, dict) and x.get("path") == dataset["parquet_path"]),
        None,
    )
    if not isinstance(entry, dict):
        raise RuntimeError("pinned GSO parquet absent from pinned revision")
    lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
    oid = str(lfs.get("oid", ""))
    xet_hash = str(entry.get("xetHash") or entry.get("xet_hash") or "")
    if oid.startswith("sha256:"):
        oid = oid.split(":", 1)[1]
    if oid and oid != dataset["expected_parquet_sha256"]:
        raise RuntimeError(f"pinned parquet sha mismatch: {oid}")
    if xet_hash and xet_hash != dataset["expected_xet_hash"]:
        raise RuntimeError(f"pinned xet hash mismatch: {xet_hash}")
    return {
        "metadata_url": url,
        "reported_lfs_sha256": oid or None,
        "reported_xet_hash": xet_hash or None,
        "entry_size": entry.get("size"),
    }


def read_safe_inventory(lock: dict) -> tuple[list[dict[str, str]], str]:
    dataset = lock["dataset"]
    safe = tuple(lock["safe_columns"])
    forbidden = set(lock["forbidden_columns"])
    url = (
        f"https://huggingface.co/datasets/{dataset['hf_dataset']}/resolve/"
        f"{dataset['hf_revision']}/{dataset['parquet_path']}?download=true"
    )
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_http_metadata_cache=false")
    quoted = ", ".join(f'"{name}"' for name in safe)
    rows = con.execute(f"SELECT {quoted} FROM read_parquet(?)", [url]).fetchall()
    names = [d[0] for d in con.description]
    if tuple(names) != safe:
        raise RuntimeError(f"unexpected projected columns: {names}")
    if len(rows) != int(dataset["expected_row_count"]):
        raise RuntimeError(f"expected {dataset['expected_row_count']} rows, got {len(rows)}")
    inventory: list[dict[str, str]] = []
    for values in rows:
        row = {name: ("" if value is None else str(value)) for name, value in zip(names, values)}
        if forbidden.intersection(row):
            raise RuntimeError("forbidden column entered selector memory")
        if not row["instance_id"] or not row["repo"]:
            raise RuntimeError("blank instance_id/repo in safe metadata")
        inventory.append(row)
    return inventory, url


def score(seed: str, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{instance_id}".encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--output", type=Path, default=Path("omega-r2-selection-evidence"))
    args = ap.parse_args()

    lock = load_json(args.lock)
    if lock["status"] != "frozen_before_selection_execution":
        raise RuntimeError("selection lock is not frozen-before-execution")
    boundary = lock["preselection_boundary"]
    if any(bool(value) for value in boundary.values()):
        raise RuntimeError("preselection boundary claims prior target access")
    if int(lock["task_count"]) != 1:
        raise RuntimeError("R2 selector requires exactly one target")

    remote = verify_remote_metadata(lock)
    inventory, parquet_url = read_safe_inventory(lock)
    excluded_ids = set(lock["exclude_instance_ids"])
    excluded_repos = set(lock["exclude_repositories"])
    seed = str(lock["selection_seed"])

    eligible = []
    for row in inventory:
        if row["instance_id"] in excluded_ids or row["repo"] in excluded_repos:
            continue
        eligible.append({**row, "selection_score": score(seed, row["instance_id"])})
    if not eligible:
        raise RuntimeError("no eligible prospective-development rows remain")
    eligible.sort(key=lambda row: (row["selection_score"], row["instance_id"]))
    selected = eligible[0]

    canonical_inventory = sorted(inventory, key=lambda row: (row["instance_id"], row["repo"]))
    inventory_sha256 = hashlib.sha256(
        json.dumps(canonical_inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    eligible_sha256 = hashlib.sha256(
        json.dumps(eligible, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    report = {
        "project": "LEXIGEN OMEGA",
        "stage": lock["stage"],
        "status": "selected_from_frozen_safe_metadata_rule",
        "scientific_role": lock["scientific_role"],
        "selection_seed": seed,
        "dataset": lock["dataset"],
        "safe_columns": lock["safe_columns"],
        "forbidden_columns": lock["forbidden_columns"],
        "projection_method": "DuckDB remote Parquet projection pushdown over HTTP range requests",
        "projected_parquet_url": parquet_url,
        "remote_metadata": remote,
        "inventory_row_count": len(inventory),
        "inventory_sha256": inventory_sha256,
        "excluded_instance_count": len(excluded_ids),
        "excluded_repository_count": len(excluded_repos),
        "eligible_row_count": len(eligible),
        "eligible_order_sha256": eligible_sha256,
        "selected": selected,
        "selection_rule": lock["selection_rule"],
        "postselection_policy": lock["postselection_policy"],
        "forbidden_column_values_accessed": False,
        "target_source_accessed": False,
        "target_tests_accessed": False,
        "target_hints_accessed": False,
        "target_performance_spec_accessed": False,
        "target_expert_patch_accessed": False,
        "target_outcome_accessed": False,
        "target_timing_accessed": False,
        "claim_boundary": lock["claim_boundary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
