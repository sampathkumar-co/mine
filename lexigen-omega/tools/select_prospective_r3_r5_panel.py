from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "evidence" / "PROSPECTIVE_R3_R5_PANEL_LOCK.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-Omega-panel-selector", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def verify_remote(lock: dict) -> dict:
    d = lock["dataset"]
    url = f"https://huggingface.co/api/datasets/{d['hf_dataset']}/tree/{d['hf_revision']}/data?recursive=false&expand=true"
    payload = fetch_json(url)
    entry = next((x for x in payload if isinstance(x, dict) and x.get("path") == d["parquet_path"]), None) if isinstance(payload, list) else None
    if not isinstance(entry, dict):
        raise RuntimeError("pinned parquet missing")
    lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
    oid = str(lfs.get("oid", ""))
    xet = str(entry.get("xetHash") or entry.get("xet_hash") or "")
    if oid.startswith("sha256:"):
        oid = oid.split(":", 1)[1]
    if oid and oid != d["expected_parquet_sha256"]:
        raise RuntimeError("parquet sha mismatch")
    if xet and xet != d["expected_xet_hash"]:
        raise RuntimeError("xet hash mismatch")
    return {"metadata_url": url, "reported_lfs_sha256": oid or None, "reported_xet_hash": xet or None, "entry_size": entry.get("size")}


def inventory(lock: dict) -> tuple[list[dict[str, str]], str]:
    d = lock["dataset"]
    safe = tuple(lock["safe_columns"])
    url = f"https://huggingface.co/datasets/{d['hf_dataset']}/resolve/{d['hf_revision']}/{d['parquet_path']}?download=true"
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_http_metadata_cache=false")
    names_sql = ", ".join(f'"{x}"' for x in safe)
    rows = con.execute(f"SELECT {names_sql} FROM read_parquet(?)", [url]).fetchall()
    names = [x[0] for x in con.description]
    if tuple(names) != safe or len(rows) != int(d["expected_row_count"]):
        raise RuntimeError("safe projection identity mismatch")
    out = [{k: ("" if v is None else str(v)) for k, v in zip(names, values)} for values in rows]
    if any(set(lock["forbidden_columns"]).intersection(row) for row in out):
        raise RuntimeError("forbidden column entered selector memory")
    return out, url


def digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--output", type=Path, default=Path("omega-r3-r5-panel-evidence"))
    args = ap.parse_args()
    lock = load(args.lock)
    if lock["status"] != "frozen_before_panel_selection_execution" or any(lock["preselection_boundary"].values()):
        raise RuntimeError("panel preselection boundary is not clean")
    if len(lock["slots"]) != int(lock["panel_policy"]["fixed_target_count"]):
        raise RuntimeError("slot count mismatch")

    remote = verify_remote(lock)
    rows, parquet_url = inventory(lock)
    excluded_ids = set(lock["initial_exclude_instance_ids"])
    excluded_repos = set(lock["initial_exclude_repositories"])
    chosen = []
    slot_summaries = []
    for slot in lock["slots"]:
        seed = slot["selection_seed"]
        eligible = []
        for row in rows:
            if row["instance_id"] in excluded_ids or row["repo"] in excluded_repos:
                continue
            score = hashlib.sha256(f"{seed}\0{row['instance_id']}".encode()).hexdigest()
            eligible.append({**row, "selection_score": score})
        eligible.sort(key=lambda r: (r["selection_score"], r["instance_id"]))
        if not eligible:
            raise RuntimeError(f"no eligible row for {slot['replication']}")
        selected = eligible[0]
        chosen.append({"replication": slot["replication"], **selected})
        slot_summaries.append({"replication": slot["replication"], "eligible_row_count": len(eligible), "eligible_order_sha256": digest(eligible)})
        excluded_ids.add(selected["instance_id"])
        excluded_repos.add(selected["repo"])

    canonical_inventory = sorted(rows, key=lambda r: (r["instance_id"], r["repo"]))
    report = {
        "project": "LEXIGEN OMEGA",
        "stage": lock["stage"],
        "status": "fixed_three_target_panel_selected_from_safe_metadata",
        "scientific_role": lock["scientific_role"],
        "dataset": lock["dataset"],
        "inventory_row_count": len(rows),
        "inventory_sha256": digest(canonical_inventory),
        "slot_summaries": slot_summaries,
        "selected": chosen,
        "selection_rule": lock["selection_rule"],
        "base_resolution_rule": lock["base_resolution_rule"],
        "panel_policy": lock["panel_policy"],
        "projection_method": "DuckDB remote Parquet projection pushdown over HTTP range requests",
        "projected_parquet_url": parquet_url,
        "remote_metadata": remote,
        "forbidden_column_values_accessed": False,
        "selected_sources_accessed": False,
        "selected_tests_accessed": False,
        "selected_hints_accessed": False,
        "selected_expert_patches_accessed": False,
        "selected_outcomes_accessed": False,
        "selected_timings_accessed": False,
        "claim_boundary": lock["claim_boundary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "panel.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
