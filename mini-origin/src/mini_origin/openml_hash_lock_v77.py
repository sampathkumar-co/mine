from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v77-openml-hash-lock.json"
)
V76_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v76-openml-preblind-registry.json"
)
V75_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v75-fresh-external-blind-pass.json"
)
V76_REGISTRY_DIGEST = "d312c4f0b853237479d6be8a74b6bf47776722d7aea1ce00c7b9745be90d57d2"
V75_EVIDENCE_DIGEST = "db379850b2a517e16d5ea442047ac4933ad06fdcf4d6838d91fc36d72e75bc47"
FROZEN_V75_COMMIT = "d8aa4153b69b82ccb714cfbb50d12c5137186047"
OPENML_VERSION = "0.15.1"
USER_AGENT = "Mini-ORIGIN-v0.77-openml-hash-lock/1"
RETRY_ATTEMPTS = 6
RETRY_BASE_SECONDS = 2.0


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def retry(call: Callable[[], object], label: str) -> object:
    errors: list[str] = []
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return call()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}"[-500:])
            if attempt + 1 < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))
    raise RuntimeError(f"{label} failed after retries: {errors}")


def download(url: str) -> bytes:
    def fetch() -> bytes:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=300) as response:
            return response.read()

    return retry(fetch, f"download {url}")  # type: ignore[return-value]


def to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number.is_integer():
        return None
    return int(number)


def task_type_value(value: object) -> int | None:
    nested = getattr(value, "value", None)
    if nested is not None:
        return to_int(nested)
    text = str(value)
    if text.isdigit():
        return int(text)
    if "SUPERVISED_CLASSIFICATION" in text.upper():
        return 1
    return None


def metadata_row(task_id: int, task: object, dataset: object) -> dict[str, object]:
    qualities = getattr(dataset, "qualities", None) or {}
    row = {
        "task_id": int(task_id),
        "task_type_id": task_type_value(getattr(task, "task_type_id", None)),
        "dataset_id": int(getattr(dataset, "dataset_id")),
        "name": str(getattr(dataset, "name")),
        "normalized_name": normalize_name(str(getattr(dataset, "name"))),
        "version": to_int(getattr(dataset, "version", None)),
        "creator": str(getattr(dataset, "creator", "") or ""),
        "licence": str(getattr(dataset, "licence", "") or ""),
        "visibility": str(getattr(dataset, "visibility", "") or ""),
        "format": str(getattr(dataset, "format", "") or ""),
        "url": str(getattr(dataset, "url", "") or ""),
        "target_name": str(getattr(task, "target_name", "") or ""),
        "default_target_attribute": str(
            getattr(dataset, "default_target_attribute", "") or ""
        ),
        "row_id_attribute": getattr(dataset, "row_id_attribute", None),
        "ignore_attribute": getattr(dataset, "ignore_attribute", None),
        "original_data_url": str(
            getattr(dataset, "original_data_url", "") or ""
        ),
        "paper_url": str(getattr(dataset, "paper_url", "") or ""),
        "description": str(getattr(dataset, "description", "") or ""),
        "citation": str(getattr(dataset, "citation", "") or ""),
        "md5_checksum": str(getattr(dataset, "md5_checksum", "") or "").lower(),
        "num_instances": to_int(qualities.get("NumberOfInstances")),
        "num_features": to_int(qualities.get("NumberOfFeatures")),
        "num_classes": to_int(qualities.get("NumberOfClasses")),
        "num_missing_values": to_int(qualities.get("NumberOfMissingValues")),
    }
    row["metadata_digest"] = canonical_digest(row)
    return row


def uci_origin(row: dict[str, object]) -> bool:
    original = str(row.get("original_data_url", "")).lower()
    description = str(row.get("description", "")).lower()
    citation = str(row.get("citation", "")).lower()
    paper = str(row.get("paper_url", "")).lower()
    if "uci" in original or "archive.ics.uci.edu" in original:
        return True
    combined = "\n".join((description, citation, paper))
    indicators = (
        "archive.ics.uci.edu",
        "www.ics.uci.edu",
        "source: [uci]",
        "uci machine learning",
        "uci repository",
        "doi.org/10.24432",
    )
    return any(indicator in combined for indicator in indicators)


def metadata_rejection_reasons(
    row: dict[str, object],
    registry: dict[str, object],
    selection: dict[str, object],
) -> list[str]:
    reasons: list[str] = []
    if int(row["dataset_id"]) in {
        int(value) for value in registry["excluded_openml_dataset_ids"]
    }:
        reasons.append("frozen_openml_dataset_id")
    if str(row["normalized_name"]) in {
        str(value) for value in registry["excluded_dataset_names"]
    }:
        reasons.append("frozen_dataset_name")
    if int(row.get("task_type_id") or -1) != int(selection["required_task_type_id"]):
        reasons.append("task_type")
    instances = row.get("num_instances")
    if not isinstance(instances, int) or not (
        int(selection["minimum_instances"])
        <= instances
        <= int(selection["maximum_instances"])
    ):
        reasons.append("instances")
    features = row.get("num_features")
    if not isinstance(features, int) or not (
        int(selection["minimum_features"])
        <= features
        <= int(selection["maximum_features"])
    ):
        reasons.append("features")
    target = str(row.get("target_name", "")).strip()
    if not target or "," in target:
        reasons.append("target_columns")
    if str(row.get("default_target_attribute", "")).strip() != target:
        reasons.append("target_metadata_mismatch")
    data_format = str(row.get("format", "")).lower()
    if bool(selection["reject_sparse_datasets"]) and "sparse" in data_format:
        reasons.append("sparse")
    parsed = urlparse(str(row.get("url", "")))
    if parsed.scheme != "https" or not (parsed.hostname or "").endswith("openml.org"):
        reasons.append("raw_url")
    if bool(selection["reject_uci_origin_sources"]) and uci_origin(row):
        reasons.append("uci_origin")
    return sorted(set(reasons))


def load_frozen_inputs() -> tuple[dict[str, object], dict[str, object]]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_openml_suite_access":
        raise RuntimeError("v0.77 preregistration status changed")
    if preregistration["parent_v76_registry_digest"] != V76_REGISTRY_DIGEST:
        raise RuntimeError("v0.76 registry commitment changed")
    if preregistration["frozen_v75_commit"] != FROZEN_V75_COMMIT:
        raise RuntimeError("frozen v0.75 commit changed")
    if preregistration["parent_v75_evidence_digest"] != V75_EVIDENCE_DIGEST:
        raise RuntimeError("v0.75 evidence commitment changed")
    for key in (
        "candidate_task_or_dataset_ids_names_urls_committed_before_execution",
        "openml_suite_or_candidate_metadata_access_before_preregistration",
        "dataset_bytes_access_before_preregistration",
        "record_target_or_feature_parsing_during_lock",
        "solver_execution_during_lock",
    ):
        if preregistration[key] is not False:
            raise RuntimeError(f"hash-lock boundary violated: {key}")

    registry = json.loads(V76_REGISTRY.read_text(encoding="utf-8"))
    if registry["status"] != "openml_preblind_registry_v76_complete":
        raise RuntimeError("v0.76 registry must remain complete")
    if registry["registry_digest"] != V76_REGISTRY_DIGEST:
        raise RuntimeError("unexpected v0.76 registry digest")
    parent = json.loads(V75_EVIDENCE.read_text(encoding="utf-8"))
    if not parent["external_gate"] or parent["evidence_digest"] != V75_EVIDENCE_DIGEST:
        raise RuntimeError("unexpected v0.75 evidence")
    return preregistration, registry


def run(output: Path) -> dict[str, object]:
    preregistration, registry = load_frozen_inputs()
    try:
        import openml
    except ImportError as error:
        raise RuntimeError(f"openml=={OPENML_VERSION} is required") from error
    if str(openml.__version__) != OPENML_VERSION:
        raise RuntimeError(f"unexpected OpenML client version {openml.__version__}")

    suite = retry(
        lambda: openml.study.get_suite(int(preregistration["benchmark_suite_id"])),
        "OpenML benchmark suite",
    )
    task_ids = [int(value) for value in suite.tasks]
    suite_data_ids = {int(value) for value in suite.data}
    if len(task_ids) != len(set(task_ids)) or len(task_ids) != len(suite_data_ids):
        raise RuntimeError("unexpected OpenML suite task/dataset structure")

    selection = preregistration["selection"]
    metadata_rows: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    for task_id in sorted(task_ids):
        try:
            task = retry(
                lambda task_id=task_id: openml.tasks.get_task(
                    task_id,
                    download_data=False,
                    download_splits=False,
                    download_qualities=False,
                ),
                f"OpenML task {task_id}",
            )
            dataset_id = int(task.dataset_id)
            if dataset_id not in suite_data_ids:
                raise RuntimeError("task dataset absent from frozen suite")
            dataset = retry(
                lambda dataset_id=dataset_id: openml.datasets.get_dataset(
                    dataset_id,
                    download_data=False,
                    download_qualities=True,
                    download_features_meta_data=False,
                    force_refresh_cache=True,
                ),
                f"OpenML dataset metadata {dataset_id}",
            )
            row = metadata_row(task_id, task, dataset)
        except Exception as error:
            unavailable.append({
                "task_id": task_id,
                "error": f"{type(error).__name__}: {error}"[-1000:],
            })
            rejection_counts["metadata_unavailable"] += 1
            continue
        reasons = metadata_rejection_reasons(row, registry, selection)
        if reasons:
            rejection_counts.update(reasons)
            continue
        rank = hashlib.sha256(
            (
                f"{preregistration['selection_seed']}|{row['task_id']}|"
                f"{row['dataset_id']}|{row['normalized_name']}|"
                f"{row['metadata_digest']}"
            ).encode("utf-8")
        ).hexdigest()
        row["rank"] = rank
        metadata_rows.append(row)

    metadata_rows.sort(
        key=lambda row: (str(row["rank"]), int(row["dataset_id"]))
    )
    selected: list[dict[str, object]] = []
    byte_rejections: list[dict[str, object]] = []
    maximum_bytes = int(selection["maximum_dataset_bytes"])
    target_count = int(selection["dataset_count"])
    for row in metadata_rows:
        try:
            payload = download(str(row["url"]))
        except Exception as error:
            byte_rejections.append({
                "task_id": row["task_id"],
                "dataset_id": row["dataset_id"],
                "reason": "download_failed",
                "error": f"{type(error).__name__}: {error}"[-1000:],
            })
            continue
        actual_md5 = hashlib.md5(payload).hexdigest()
        expected_md5 = str(row.get("md5_checksum", ""))
        if not payload or len(payload) > maximum_bytes:
            byte_rejections.append({
                "task_id": row["task_id"],
                "dataset_id": row["dataset_id"],
                "bytes": len(payload),
                "reason": "empty_or_over_maximum_dataset_bytes",
            })
            continue
        if expected_md5 and actual_md5 != expected_md5:
            byte_rejections.append({
                "task_id": row["task_id"],
                "dataset_id": row["dataset_id"],
                "expected_md5": expected_md5,
                "actual_md5": actual_md5,
                "reason": "openml_md5_mismatch",
            })
            continue
        selected.append({
            **row,
            "raw_sha256": hashlib.sha256(payload).hexdigest(),
            "raw_md5": actual_md5,
            "raw_bytes": len(payload),
        })
        if len(selected) == target_count:
            break

    if len(selected) != target_count:
        raise RuntimeError(
            f"only {len(selected)} eligible hash-locked OpenML datasets; "
            f"required {target_count}"
        )
    selected_ids = [int(row["dataset_id"]) for row in selected]
    selected_names = [str(row["normalized_name"]) for row in selected]
    id_overlap = sorted(
        set(selected_ids)
        & {int(value) for value in registry["excluded_openml_dataset_ids"]}
    )
    name_overlap = sorted(
        set(selected_names)
        & {str(value) for value in registry["excluded_dataset_names"]}
    )
    if id_overlap or name_overlap:
        raise RuntimeError(
            f"selected OpenML suite overlaps frozen registry: ids={id_overlap}, "
            f"names={name_overlap}"
        )

    suite_commitment = {
        "suite_id": int(suite.suite_id),
        "alias": str(suite.alias),
        "name": str(suite.name),
        "status": str(suite.status),
        "task_ids": task_ids,
        "dataset_ids": sorted(suite_data_ids),
    }
    result = {
        "status": "openml_cross_source_hash_lock_v77_complete",
        "protocol": preregistration["protocol"],
        "claim_scope": preregistration["claim_boundary"],
        "openml_client_version": OPENML_VERSION,
        "parent_v76_registry_digest": V76_REGISTRY_DIGEST,
        "frozen_v75_commit": FROZEN_V75_COMMIT,
        "parent_v75_evidence_digest": V75_EVIDENCE_DIGEST,
        "suite": suite_commitment,
        "suite_digest": canonical_digest(suite_commitment),
        "selection": selection,
        "selection_seed": preregistration["selection_seed"],
        "metadata_eligible_count": len(metadata_rows),
        "metadata_rejection_counts": dict(sorted(rejection_counts.items())),
        "metadata_unavailable": unavailable,
        "byte_rejections": byte_rejections,
        "selected_id_overlap": id_overlap,
        "selected_name_overlap": name_overlap,
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
        "suite_tasks": len(result["suite"]["task_ids"]),
        "metadata_eligible": result["metadata_eligible_count"],
        "metadata_unavailable": len(result["metadata_unavailable"]),
        "datasets": result["dataset_count"],
        "selected_id_overlap": result["selected_id_overlap"],
        "selected_name_overlap": result["selected_name_overlap"],
        "lock_digest": result["lock_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
