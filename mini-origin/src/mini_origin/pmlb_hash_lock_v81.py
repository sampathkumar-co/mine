from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v81-pmlb-hash-lock.json"
)
REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v80-pmlb-preblind-registry.json"
)
SOURCE_RECORD = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v81-pmlb-source-commit.json"
)
SOURCE_REPOSITORY = "EpistasisLab/pmlb"
SOURCE_COMMIT = "7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68"
SOURCE_TREE = "ca5d36e9093c2f7360db57198c8c0586a3217a60"
PARENT_V80_COMMIT = "22cc53c9b22b1d6e15190e769462846992e28149"
V80_REGISTRY_DIGEST = "aa6bab47d8d2453b669eee2f7a36720e0eb798a79355b0d2c9a509d81959038c"
V80_REGISTRY_SHA256 = "242a3b4173786ab52435658c959ce6195c3f831c25cf3dff4bba2d784796a3f9"
PREREGISTRATION_COMMIT = "e150c73fb558d6083991e37d7aab9eacc81ca35b"
USER_AGENT = "Mini-ORIGIN-v0.81-pmlb-hash-lock/1"
RETRY_ATTEMPTS = 6
RETRY_BASE_SECONDS = 1.5
SUMMARY_PATH = "pmlb/all_summary_stats.tsv"
RAW_BASE = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_COMMIT}"
SUMMARY_URL = f"{RAW_BASE}/{SUMMARY_PATH}"
REQUIRED_SUMMARY_FIELDS = (
    "dataset",
    "n_instances",
    "n_features",
    "endpoint_type",
    "n_classes",
    "task",
)
OPEN_SOURCE_PATTERN = re.compile(r"\b(?:uci|openml)\b", re.IGNORECASE)
FIELD_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def download(url: str) -> bytes:
    errors: list[str] = []
    for attempt in range(RETRY_ATTEMPTS):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=300) as handle:
                return handle.read()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}"[-500:])
            if attempt + 1 < RETRY_ATTEMPTS:
                time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))
    raise RuntimeError(f"download failed after retries: {errors}")


def canonical_metadata(payload: bytes) -> tuple[str, str]:
    text = payload.decode("utf-8-sig", errors="strict")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    canonical = "\n".join(lines).rstrip() + "\n"
    return canonical, sha256_bytes(canonical.encode("utf-8"))


def top_level_fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith((" ", "\t", "#")) or not line.strip():
            continue
        match = FIELD_PATTERN.match(line)
        if match:
            result[match.group(1).lower()] = match.group(2).strip(" '\"")
    return result


def metadata_url(name: str) -> str:
    token = quote(name, safe="")
    return f"{RAW_BASE}/datasets/{token}/metadata.yaml"


def raw_url(name: str) -> str:
    token = quote(name, safe="")
    return f"{RAW_BASE}/datasets/{token}/{token}.tsv.gz"


def load_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_pmlb_catalogue_access":
        raise RuntimeError("v0.81 preregistration status changed")
    if preregistration["parent_v80_commit"] != PARENT_V80_COMMIT:
        raise RuntimeError("frozen v0.80 commit changed")
    if preregistration["parent_v80_registry_digest"] != V80_REGISTRY_DIGEST:
        raise RuntimeError("v0.80 registry digest changed")
    if preregistration["parent_v80_registry_sha256"] != V80_REGISTRY_SHA256:
        raise RuntimeError("v0.80 registry hash changed")
    for key in (
        "pmlb_catalogue_or_repository_tree_access_before_preregistration",
        "pmlb_candidate_names_ids_or_urls_committed_before_preregistration",
        "pmlb_dataset_bytes_access_before_preregistration",
        "record_or_label_access_before_preregistration",
        "solver_execution_before_preregistration",
    ):
        if preregistration[key] is not False:
            raise RuntimeError(f"preregistration boundary violated: {key}")

    registry_bytes = REGISTRY.read_bytes()
    if sha256_bytes(registry_bytes) != V80_REGISTRY_SHA256:
        raise RuntimeError("v0.80 registry bytes changed")
    registry = json.loads(registry_bytes.decode("utf-8"))
    if registry["status"] != "pmlb_preblind_registry_v80_complete":
        raise RuntimeError("v0.80 registry is not complete")
    if registry["registry_digest"] != V80_REGISTRY_DIGEST:
        raise RuntimeError("unexpected v0.80 registry digest")

    source = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
    if source["status"] != "pmlb_source_commit_frozen_after_preregistration":
        raise RuntimeError("PMLB source record status changed")
    if source["preregistration_commit"] != PREREGISTRATION_COMMIT:
        raise RuntimeError("PMLB source was not frozen from the preregistered commit")
    if source["source_commit"] != SOURCE_COMMIT:
        raise RuntimeError("PMLB source commit changed")
    if source["source_tree"] != SOURCE_TREE:
        raise RuntimeError("PMLB source tree changed")
    if source["dataset_records_or_labels_accessed"] is not False:
        raise RuntimeError("records were accessed before the hash lock")
    if source["dataset_bytes_accessed"] is not False:
        raise RuntimeError("dataset bytes were accessed before the hash lock")
    if source["solver_executed"] is not False:
        raise RuntimeError("solver ran before the hash lock")
    return preregistration, registry, source


def parse_float(value: str, field: str, dataset: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise RuntimeError(f"invalid {field} for {dataset}: {value}") from error


def parse_summary(payload: bytes) -> list[dict[str, object]]:
    text = payload.decode("utf-8-sig", errors="strict")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if not reader.fieldnames:
        raise RuntimeError("PMLB summary has no header")
    missing = [field for field in REQUIRED_SUMMARY_FIELDS if field not in reader.fieldnames]
    if missing:
        raise RuntimeError(f"PMLB summary fields missing: {missing}")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in reader:
        name = str(raw["dataset"]).strip()
        normalized = normalize_name(name)
        if not name or not normalized or normalized in seen:
            raise RuntimeError(f"missing or duplicate PMLB dataset: {name}")
        seen.add(normalized)
        instances = int(parse_float(str(raw["n_instances"]), "instances", name))
        features = int(parse_float(str(raw["n_features"]), "features", name))
        classes = int(parse_float(str(raw["n_classes"]), "classes", name))
        rows.append({
            "name": name,
            "normalized_name": normalized,
            "instances": instances,
            "features": features,
            "classes": classes,
            "endpoint_type": str(raw["endpoint_type"]).strip().lower(),
            "task": str(raw["task"]).strip().lower(),
        })
    return rows


def eligible_summary(
    row: dict[str, object],
    preregistration: dict[str, object],
) -> bool:
    rules = preregistration["metadata_only_eligibility"]
    return (
        row["task"] == rules["problem_type"]
        and int(rules["minimum_instances"]) <= int(row["instances"])
        <= int(rules["maximum_instances"])
        and int(rules["minimum_features"]) <= int(row["features"])
        <= int(rules["maximum_features"])
        and int(rules["minimum_classes"]) <= int(row["classes"])
        <= int(rules["maximum_classes"])
        and row["endpoint_type"] == "categorical"
    )


def mentioned_excluded_names(
    metadata_text: str,
    excluded_names: set[str],
    own_name: str,
) -> list[str]:
    haystack = f"-{normalize_name(metadata_text)}-"
    result = []
    for name in excluded_names:
        if name == own_name or len(name) < 4:
            continue
        if f"-{name}-" in haystack:
            result.append(name)
    return sorted(result)


def metadata_candidate(
    row: dict[str, object],
    metadata_payload: bytes,
    excluded_names: set[str],
    seed: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    text, metadata_digest = canonical_metadata(metadata_payload)
    fields = top_level_fields(text)
    name = str(row["name"])
    normalized = str(row["normalized_name"])
    rejection: dict[str, object] = {
        "name": name,
        "normalized_name": normalized,
    }
    if normalize_name(fields.get("dataset", "")) != normalized:
        rejection["reason"] = "metadata_dataset_mismatch"
        return None, rejection
    if fields.get("task", "").lower() != "classification":
        rejection["reason"] = "metadata_task_mismatch"
        return None, rejection
    if normalized in excluded_names:
        rejection["reason"] = "name_overlap"
        return None, rejection
    if OPEN_SOURCE_PATTERN.search(text):
        rejection["reason"] = "metadata_mentions_uci_or_openml"
        return None, rejection
    overlaps = mentioned_excluded_names(text, excluded_names, normalized)
    if overlaps:
        rejection["reason"] = "metadata_mentions_excluded_name"
        rejection["overlaps"] = overlaps[:20]
        return None, rejection
    rank = hashlib.sha256(
        json.dumps(
            [seed, SOURCE_COMMIT, normalized, metadata_digest],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    candidate = dict(row)
    candidate.update({
        "metadata_path": f"datasets/{name}/metadata.yaml",
        "metadata_url": metadata_url(name),
        "metadata_raw_sha256": sha256_bytes(metadata_payload),
        "metadata_bytes": len(metadata_payload),
        "canonical_metadata_digest": metadata_digest,
        "rank": rank,
    })
    return candidate, None


def selected_byte_lock(
    candidates: list[dict[str, object]],
    count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (str(row["rank"]), str(row["normalized_name"])),
    ):
        if len(selected) >= count:
            break
        name = str(candidate["name"])
        url = raw_url(name)
        try:
            payload = download(url)
        except Exception as error:
            rejections.append({
                "name": name,
                "reason": "raw_download_failure",
                "error": str(error)[-500:],
            })
            continue
        if not payload.startswith(b"\x1f\x8b"):
            rejections.append({
                "name": name,
                "reason": "raw_not_gzip",
                "bytes": len(payload),
            })
            continue
        locked = dict(candidate)
        locked.update({
            "raw_path": f"datasets/{name}/{name}.tsv.gz",
            "raw_url": url,
            "raw_sha256": sha256_bytes(payload),
            "raw_bytes": len(payload),
            "gzip_magic_verified": True,
            "decompressed": False,
            "records_or_labels_parsed": False,
        })
        selected.append(locked)
    return selected, rejections


def lock(output_path: Path) -> dict[str, object]:
    preregistration, registry, source = load_inputs()
    summary_payload = download(SUMMARY_URL)
    summary_rows = parse_summary(summary_payload)
    eligible_rows = [
        row for row in summary_rows if eligible_summary(row, preregistration)
    ]
    excluded_names = {
        str(value) for value in registry["excluded_dataset_names"]
    }
    candidates: list[dict[str, object]] = []
    metadata_rejections: list[dict[str, object]] = []
    metadata_failures: list[dict[str, object]] = []

    for row in eligible_rows:
        name = str(row["name"])
        try:
            payload = download(metadata_url(name))
        except Exception as error:
            metadata_failures.append({
                "name": name,
                "error": str(error)[-500:],
            })
            continue
        candidate, rejection = metadata_candidate(
            row,
            payload,
            excluded_names,
            str(preregistration["selection_seed"]),
        )
        if candidate is not None:
            candidates.append(candidate)
        elif rejection is not None:
            metadata_rejections.append(rejection)

    selected, byte_rejections = selected_byte_lock(
        candidates,
        int(preregistration["dataset_count"]),
    )
    if len(selected) != int(preregistration["dataset_count"]):
        raise RuntimeError(
            f"insufficient PMLB byte-lock candidates: {len(selected)}"
        )
    selected_names = [str(row["normalized_name"]) for row in selected]
    overlap = sorted(set(selected_names) & excluded_names)
    if overlap:
        raise RuntimeError(f"selected PMLB names overlap registry: {overlap}")

    protocol = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "summary_path": SUMMARY_PATH,
        "selection_seed": preregistration["selection_seed"],
        "metadata_only_eligibility": preregistration[
            "metadata_only_eligibility"
        ],
        "exclusions": preregistration["exclusions"],
        "ranking": preregistration["ranking"],
        "byte_lock_protocol": preregistration["byte_lock_protocol"],
        "records_or_labels_accessed": False,
        "solver_executed": False,
    }
    result = {
        "status": "pmlb_hash_lock_v81_complete",
        "protocol": protocol,
        "parent_v80_commit": PARENT_V80_COMMIT,
        "parent_v80_registry_digest": V80_REGISTRY_DIGEST,
        "source_record": source,
        "summary_url": SUMMARY_URL,
        "summary_sha256": sha256_bytes(summary_payload),
        "summary_bytes": len(summary_payload),
        "summary_dataset_count": len(summary_rows),
        "summary_eligible_count": len(eligible_rows),
        "metadata_candidate_count": len(candidates),
        "metadata_failure_count": len(metadata_failures),
        "metadata_failures": metadata_failures,
        "metadata_rejection_count": len(metadata_rejections),
        "metadata_rejections": metadata_rejections,
        "byte_rejection_count": len(byte_rejections),
        "byte_rejections": byte_rejections,
        "dataset_count": len(selected),
        "selected_name_overlap": overlap,
        "datasets": selected,
        "records_or_labels_accessed": False,
        "solver_executed": False,
        "claim_scope": preregistration["claim_boundary"],
    }
    result["lock_digest"] = canonical_digest({
        "protocol": protocol,
        "parent_v80_registry_digest": V80_REGISTRY_DIGEST,
        "summary_sha256": result["summary_sha256"],
        "datasets": [
            {
                "name": row["name"],
                "normalized_name": row["normalized_name"],
                "instances": row["instances"],
                "features": row["features"],
                "classes": row["classes"],
                "canonical_metadata_digest": row[
                    "canonical_metadata_digest"
                ],
                "raw_sha256": row["raw_sha256"],
                "raw_bytes": row["raw_bytes"],
            }
            for row in selected
        ],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = lock(args.output)
    print(json.dumps({
        "status": result["status"],
        "source_commit": result["protocol"]["source_commit"],
        "summary_datasets": result["summary_dataset_count"],
        "eligible": result["summary_eligible_count"],
        "metadata_candidates": result["metadata_candidate_count"],
        "metadata_rejections": result["metadata_rejection_count"],
        "metadata_failures": result["metadata_failure_count"],
        "byte_rejections": result["byte_rejection_count"],
        "selected": result["dataset_count"],
        "lock_digest": result["lock_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
