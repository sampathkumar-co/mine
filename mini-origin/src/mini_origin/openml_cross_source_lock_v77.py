from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

REGISTRY = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v76-openml-preblind-registry.json"
PROTOCOL = Path(__file__).resolve().parents[2] / "campaigns" / "v77-openml-cross-source-lock.json"
REGISTRY_DIGEST = "d312c4f0b853237479d6be8a74b6bf47776722d7aea1ce00c7b9745be90d57d2"
SUITE_URL = "https://www.openml.org/api/v1/json/benchmarking/suite/99"
DATA_URL = "https://www.openml.org/api/v1/json/data/{dataset_id}"
QUALITIES_URL = "https://www.openml.org/api/v1/json/data/qualities/{dataset_id}"
DOWNLOAD_URL = "https://www.openml.org/data/v1/download/{file_id}"
USER_AGENT = "Mini-ORIGIN-v0.77-cross-source-lock/1"


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,*/*"})
    with urlopen(request, timeout=300) as response:
        return response.read()


def fetch_json(url: str) -> dict[str, object]:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def normalize_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text


def collect_dataset_ids(value: object) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"data_id", "dataset_id", "did"}:
                try:
                    found.add(int(item))
                except (TypeError, ValueError):
                    pass
            found.update(collect_dataset_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(collect_dataset_ids(item))
    return found


def description(payload: dict[str, object]) -> dict[str, object]:
    row = payload.get("data_set_description", payload)
    if not isinstance(row, dict):
        raise RuntimeError("invalid OpenML dataset description")
    return row


def quality_map(payload: dict[str, object]) -> dict[str, float]:
    root = payload.get("data_qualities", payload)
    rows = root.get("quality", []) if isinstance(root, dict) else []
    if isinstance(rows, dict):
        rows = [rows]
    result: dict[str, float] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            result[str(row["name"])] = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def metadata_text(row: dict[str, object]) -> str:
    keys = ("name", "description", "original_data_url", "paper_url", "creator", "citation")
    return " ".join(str(row.get(key, "")) for key in keys).lower()


def eligible(
    row: dict[str, object], qualities: dict[str, float], excluded_ids: set[int], excluded_names: set[str]
) -> tuple[bool, str]:
    dataset_id = int(row["id"])
    name = normalize_name(row.get("name", ""))
    if dataset_id in excluded_ids or name in excluded_names:
        return False, "registry-excluded"
    text = metadata_text(row)
    if "archive.ics.uci.edu" in text or re.search(r"\buci\b", text):
        return False, "uci-origin"
    if str(row.get("status", "active")).lower() != "active":
        return False, "inactive"
    if str(row.get("format", "")).upper() != "ARFF":
        return False, "format"
    required = ("NumberOfInstances", "NumberOfFeatures", "NumberOfClasses")
    if any(key not in qualities for key in required):
        return False, "missing-quality"
    instances = int(qualities["NumberOfInstances"])
    features = int(qualities["NumberOfFeatures"])
    classes = int(qualities["NumberOfClasses"])
    if not 500 <= instances <= 10000:
        return False, "instances"
    if not 4 <= features <= 500:
        return False, "features"
    if not 2 <= classes <= 20:
        return False, "classes"
    missing = qualities.get("NumberOfInstancesWithMissingValues", 0.0)
    if instances and missing / instances > 0.20:
        return False, "missing-fraction"
    if not row.get("file_id"):
        return False, "missing-file-id"
    return True, "eligible"


def rank_key(seed: str, row: dict[str, object]) -> str:
    token = "|".join((seed, str(row["id"]), normalize_name(row.get("name", "")), str(row.get("version", "")), str(row["file_id"])))
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def run(output: Path) -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if registry["registry_digest"] != REGISTRY_DIGEST:
        raise RuntimeError("v0.76 registry digest changed")
    if protocol["parent_v76_registry_digest"] != REGISTRY_DIGEST:
        raise RuntimeError("protocol parent changed")
    excluded_ids = {int(value) for value in registry["excluded_openml_dataset_ids"]}
    excluded_names = {str(value) for value in registry["excluded_dataset_names"]}
    suite_payload = fetch_json(SUITE_URL)
    suite_ids = sorted(collect_dataset_ids(suite_payload))
    if len(suite_ids) < 20:
        raise RuntimeError(f"unexpectedly small OpenML-CC18 suite: {len(suite_ids)}")
    considered = []
    eligible_rows = []
    seed = str(protocol["selection_seed"])
    for dataset_id in suite_ids:
        row = description(fetch_json(DATA_URL.format(dataset_id=dataset_id)))
        qualities = quality_map(fetch_json(QUALITIES_URL.format(dataset_id=dataset_id)))
        ok, reason = eligible(row, qualities, excluded_ids, excluded_names)
        summary = {
            "dataset_id": int(row["id"]), "name": str(row.get("name", "")),
            "normalized_name": normalize_name(row.get("name", "")), "version": int(row.get("version", 1)),
            "file_id": int(row["file_id"]) if row.get("file_id") else None, "eligible": ok, "reason": reason,
        }
        considered.append(summary)
        if ok:
            eligible_rows.append((rank_key(seed, row), row, qualities))
    eligible_rows.sort(key=lambda item: item[0])
    selected = []
    for rank, row, qualities in eligible_rows[: int(protocol["dataset_count"])]:
        raw = fetch_bytes(DOWNLOAD_URL.format(file_id=int(row["file_id"])))
        selected.append({
            "rank": rank, "dataset_id": int(row["id"]), "name": str(row["name"]),
            "normalized_name": normalize_name(row["name"]), "version": int(row.get("version", 1)),
            "file_id": int(row["file_id"]), "format": str(row.get("format", "")),
            "instances_metadata": int(qualities["NumberOfInstances"]),
            "features_metadata": int(qualities["NumberOfFeatures"]),
            "classes_metadata": int(qualities["NumberOfClasses"]),
            "download_url": DOWNLOAD_URL.format(file_id=int(row["file_id"])),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        })
    if len(selected) != int(protocol["dataset_count"]):
        raise RuntimeError(f"only {len(selected)} eligible datasets")
    result: dict[str, object] = {
        "status": "openml_cross_source_hash_lock_v77_complete",
        "protocol": "metadata-only deterministic selection then raw-byte hash lock; no record parsing",
        "parent_v76_registry_digest": REGISTRY_DIGEST,
        "parent_v75_evidence_digest": protocol["parent_v75_evidence_digest"],
        "suite_id": 99, "suite_dataset_count": len(suite_ids),
        "eligible_dataset_count": len(eligible_rows), "considered": considered, "datasets": selected,
        "candidate_overlap_ids": sorted({row["dataset_id"] for row in selected} & excluded_ids),
        "candidate_overlap_names": sorted({row["normalized_name"] for row in selected} & excluded_names),
    }
    result["lock_digest"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "suite": result["suite_dataset_count"], "eligible": result["eligible_dataset_count"], "selected": len(result["datasets"]), "lock_digest": result["lock_digest"]}, indent=2))


if __name__ == "__main__":
    main()
