from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v74-fresh-external-hash-lock.json"
)
V73_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v73-frozen-dataset-registry.json"
)
V67_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v67-updated-dataset-registry.json"
)
V73_REGISTRY_DIGEST = "3fde8f0138548928651937201cef66aa71ee41bbb792ade82b1a02337ae8b392"
V72_EVIDENCE_DIGEST = "b1fc70852a2ad35d91972889eb853856cde18bca0ed02db37cd37ac333639090"
FROZEN_V72_COMMIT = "dae02829efc4819935a4ec87c31ea5eee3305d83"
USER_AGENT = "Mini-ORIGIN-v0.74-fresh-hash-lock/1"


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as response:
        return response.read()


def fetch_json(url: str) -> dict[str, object]:
    payload = json.loads(download(url))
    if int(payload.get("status", 0)) != 200:
        raise RuntimeError(f"metadata request failed: {url}")
    return payload


def metadata_rejection_reasons(
    metadata: dict[str, object],
    contaminated_tokens: set[str],
    selection: dict[str, object],
) -> list[str]:
    reasons: list[str] = []
    tasks = {str(value).lower() for value in metadata.get("tasks") or []}
    if str(selection["required_task"]).lower() not in tasks:
        reasons.append("task")
    instances = metadata.get("num_instances")
    features_count = metadata.get("num_features")
    if not isinstance(instances, int) or not (
        int(selection["minimum_instances"])
        <= instances
        <= int(selection["maximum_instances"])
    ):
        reasons.append("instances")
    if not isinstance(features_count, int) or not (
        int(selection["minimum_features"])
        <= features_count
        <= int(selection["maximum_features"])
    ):
        reasons.append("features")
    data_url = metadata.get("data_url")
    if not isinstance(data_url, str) or not data_url.startswith(
        "https://archive.ics.uci.edu/static/public/"
    ):
        reasons.append("standardized_csv")
    variables = metadata.get("variables") or []
    target_count = sum(
        str(row.get("role", "")).lower() == "target"
        for row in variables
        if isinstance(row, dict)
    )
    feature_count = sum(
        str(row.get("role", "")).lower() == "feature"
        for row in variables
        if isinstance(row, dict)
    )
    if target_count != int(selection["required_target_columns"]):
        reasons.append("target_columns")
    if feature_count < int(selection["minimum_features"]):
        reasons.append("feature_roles")
    name = normalize_token(str(metadata.get("name", "")))
    if any(token and token in name for token in contaminated_tokens):
        reasons.append("contaminated_name_token")
    return sorted(set(reasons))


def run(output: Path) -> dict[str, object]:
    preregistration = json.loads(
        PREREGISTRATION.read_text(encoding="utf-8-sig")
    )
    if preregistration["status"] != "preregistered_before_metadata_access":
        raise RuntimeError("v0.74 preregistration status changed")
    if preregistration["parent_v73_registry_digest"] != V73_REGISTRY_DIGEST:
        raise RuntimeError("v0.73 registry commitment changed")
    if preregistration["frozen_v72_commit"] != FROZEN_V72_COMMIT:
        raise RuntimeError("frozen v0.72 commit changed")
    if preregistration["parent_v72_evidence_digest"] != V72_EVIDENCE_DIGEST:
        raise RuntimeError("v0.72 evidence commitment changed")
    for key in (
        "candidate_ids_names_urls_committed_before_execution",
        "data_bytes_access_before_preregistration",
        "record_or_label_parsing_during_lock",
        "solver_execution_during_lock",
    ):
        if preregistration[key] is not False:
            raise RuntimeError(f"hash-lock boundary violated: {key}")

    registry = json.loads(V73_REGISTRY.read_text(encoding="utf-8"))
    if registry["status"] != "repository_dataset_registry_v73_complete":
        raise RuntimeError("v0.73 registry must remain complete")
    if registry["registry_digest"] != V73_REGISTRY_DIGEST:
        raise RuntimeError("unexpected v0.73 registry digest")
    excluded_ids = {int(value) for value in registry["excluded_uci_ids"]}

    previous = json.loads(V67_REGISTRY.read_text(encoding="utf-8"))
    contaminated_tokens = {
        normalize_token(str(value))
        for value in previous["pystreed_dataset_tokens"]
    }

    list_payload = fetch_json(str(preregistration["metadata_list_url"]))
    list_digest = canonical_digest(list_payload)
    selection = preregistration["selection"]
    metadata_rows = []
    rejection_counts: Counter[str] = Counter()

    list_rows = list_payload.get("data") or []
    for listed in sorted(list_rows, key=lambda row: int(row["id"])):
        uci_id = int(listed["id"])
        if uci_id in excluded_ids:
            rejection_counts["excluded_uci_id"] += 1
            continue
        url = str(preregistration["metadata_dataset_url_template"]).format(
            uci_id=uci_id
        )
        metadata_payload = fetch_json(url)
        metadata = metadata_payload["data"]
        if int(metadata["uci_id"]) != uci_id:
            raise RuntimeError(f"metadata ID mismatch: {uci_id}")
        reasons = metadata_rejection_reasons(
            metadata,
            contaminated_tokens,
            selection,
        )
        if reasons:
            rejection_counts.update(reasons)
            continue
        metadata_digest = canonical_digest(metadata_payload)
        rank = hashlib.sha256(
            (
                f"{preregistration['selection_seed']}|{uci_id}|"
                f"{metadata['name']}|{metadata_digest}"
            ).encode("utf-8")
        ).hexdigest()
        metadata_rows.append({
            "rank": rank,
            "uci_id": uci_id,
            "name": metadata["name"],
            "repository_url": metadata["repository_url"],
            "data_url": metadata["data_url"],
            "dataset_doi": metadata.get("dataset_doi"),
            "num_instances": metadata["num_instances"],
            "num_features": metadata["num_features"],
            "target_columns": list(metadata.get("target_col") or []),
            "metadata_digest": metadata_digest,
        })

    metadata_rows.sort(key=lambda row: (str(row["rank"]), int(row["uci_id"])))
    selected = []
    byte_rejections = []
    maximum_bytes = int(selection["maximum_csv_bytes"])
    target_count = int(selection["dataset_count"])
    for row in metadata_rows:
        data = download(str(row["data_url"]))
        if not data or len(data) > maximum_bytes:
            byte_rejections.append({
                "uci_id": row["uci_id"],
                "bytes": len(data),
                "reason": "empty_or_over_maximum_csv_bytes",
            })
            continue
        selected.append({
            **row,
            "csv_sha256": hashlib.sha256(data).hexdigest(),
            "csv_bytes": len(data),
        })
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise RuntimeError(
            f"only {len(selected)} eligible hash-locked datasets; required {target_count}"
        )

    selected_ids = [int(row["uci_id"]) for row in selected]
    overlap = sorted(set(selected_ids) & excluded_ids)
    if overlap:
        raise RuntimeError(f"selected IDs overlap frozen registry: {overlap}")
    result = {
        "status": "fresh_external_hash_lock_v74_complete",
        "protocol": preregistration["protocol"],
        "claim_scope": preregistration["claim_boundary"],
        "parent_v73_registry_digest": V73_REGISTRY_DIGEST,
        "frozen_v72_commit": FROZEN_V72_COMMIT,
        "parent_v72_evidence_digest": V72_EVIDENCE_DIGEST,
        "metadata_list_digest": list_digest,
        "metadata_list_count": len(list_rows),
        "excluded_uci_id_count": len(excluded_ids),
        "contaminated_name_tokens": sorted(contaminated_tokens),
        "selection": selection,
        "selection_seed": preregistration["selection_seed"],
        "metadata_eligible_count": len(metadata_rows),
        "metadata_rejection_counts": dict(sorted(rejection_counts.items())),
        "byte_rejections": byte_rejections,
        "selected_overlap": overlap,
        "dataset_count": len(selected),
        "datasets": selected,
    }
    result["lock_digest"] = canonical_digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "status": result["status"],
        "metadata_list_count": result["metadata_list_count"],
        "excluded_uci_ids": result["excluded_uci_id_count"],
        "metadata_eligible": result["metadata_eligible_count"],
        "datasets": result["dataset_count"],
        "selected_overlap": result["selected_overlap"],
        "lock_digest": result["lock_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
