from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from . import repository_dataset_audit_v63 as base


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v80-cross-source-preblind-audit.json"
)
V79_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v79-small-query-coverage-pass.json"
)
FROZEN_V79_COMMIT = "555c3146111a7726702bb98e0a72f3b214d07190"
V79_EVIDENCE_DIGEST = "5c28f39546d8e7d988fbc78ce254fa35d2b1d85d49a4b270b50edde1235d3001"
REQUIRED_V79_OPENML_IDS = (1068, 188, 40499, 1067, 1063, 469, 1049)

OPENML_ID_PATTERNS = (
    re.compile(r'["\']openml_(?:dataset_)?id["\']\s*[:=]\s*(\d+)', re.IGNORECASE),
    re.compile(r'\bopenml\.org/(?:d|data)/(\d+)\b', re.IGNORECASE),
    re.compile(r'\bwww\.openml\.org/(?:d|data)/(\d+)\b', re.IGNORECASE),
)
OPENML_NAME_ID_PATTERN = re.compile(
    r'["\']name["\']\s*:\s*["\']([^"\']+)["\'][\s\S]{0,350}?'
    r'["\']openml_(?:dataset_)?id["\']\s*:\s*(\d+)',
    re.IGNORECASE,
)
OPENML_ID_NAME_PATTERN = re.compile(
    r'["\']openml_(?:dataset_)?id["\']\s*:\s*(\d+)[\s\S]{0,350}?'
    r'["\']name["\']\s*:\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_openml_ids(text: str) -> set[int]:
    result: set[int] = set()
    for pattern in OPENML_ID_PATTERNS:
        result.update(int(value) for value in pattern.findall(text))
    return result


def collect_json_dataset_names(value: object, output: set[str]) -> None:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        id_keys = {
            "uci_id",
            "openml_id",
            "openml_dataset_id",
            "dataset_id",
        }
        if keys & id_keys:
            for name_key in ("name", "task", "dataset_name"):
                candidate = value.get(name_key)
                if isinstance(candidate, str) and candidate.strip():
                    output.add(normalize_name(candidate))
        for child in value.values():
            collect_json_dataset_names(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_json_dataset_names(child, output)


def extract_dataset_names(text: str, suffix: str) -> set[str]:
    names: set[str] = set()
    for name, _ in base.NAME_ID_PATTERN.findall(text):
        names.add(normalize_name(name))
    for _, name in base.ID_NAME_PATTERN.findall(text):
        names.add(normalize_name(name))
    for name, _ in OPENML_NAME_ID_PATTERN.findall(text):
        names.add(normalize_name(name))
    for _, name in OPENML_ID_NAME_PATTERN.findall(text):
        names.add(normalize_name(name))
    if suffix == ".json":
        try:
            collect_json_dataset_names(json.loads(text), names)
        except json.JSONDecodeError:
            pass
    return {name for name in names if name}


def audit() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_new_source_metadata_access":
        raise RuntimeError("v0.80 preregistration status changed")
    if preregistration["frozen_v79_commit"] != FROZEN_V79_COMMIT:
        raise RuntimeError("frozen v0.79 commit changed")
    if preregistration["parent_v79_evidence_digest"] != V79_EVIDENCE_DIGEST:
        raise RuntimeError("v0.79 evidence commitment changed")
    for key in (
        "new_source_candidate_ids_names_or_urls_committed_before_audit",
        "new_source_api_access_before_audit",
        "new_source_dataset_bytes_access_before_audit",
        "solver_execution_before_audit",
    ):
        if preregistration[key] is not False:
            raise RuntimeError(f"preblind audit boundary violated: {key}")

    parent = json.loads(V79_EVIDENCE.read_text(encoding="utf-8"))
    if not parent["development_gate"]:
        raise RuntimeError("v0.79 parent must remain a pass")
    if parent["evidence_digest"] != V79_EVIDENCE_DIGEST:
        raise RuntimeError("unexpected v0.79 evidence digest")

    base_report = base.audit()
    refs = base.remote_refs()
    openml_occurrences: dict[str, list[dict[str, str]]] = defaultdict(list)
    names: set[str] = set()
    files_scanned = 0
    bytes_scanned = 0
    failures: list[dict[str, str]] = []

    for ref in refs:
        for path, size in base.tree_files(ref):
            try:
                text = base.show_text(ref, path)
            except Exception as error:
                failures.append({"ref": ref, "path": path, "error": str(error)[-500:]})
                continue
            files_scanned += 1
            bytes_scanned += size
            occurrence = {"ref": ref, "path": path}
            for openml_id in sorted(extract_openml_ids(text)):
                key = str(openml_id)
                if occurrence not in openml_occurrences[key]:
                    openml_occurrences[key].append(occurrence)
            names.update(extract_dataset_names(text, Path(path).suffix.lower()))

    discovered_uci_ids = {
        int(row["uci_id"]) for row in base_report["uci_datasets"]
    }
    required_openml_checks = {
        str(value): str(value) in openml_occurrences for value in REQUIRED_V79_OPENML_IDS
    }
    required_names = {"pc1", "eucalyptus", "texture", "kc1", "kc2", "analcatdata-dmft", "pc4"}
    required_name_checks = {
        value: value in names for value in sorted(required_names)
    }
    openml_rows = [
        {
            "openml_dataset_id": int(openml_id),
            "occurrence_count_capped": len(occurrences),
            "occurrences": occurrences[:40],
        }
        for openml_id, occurrences in sorted(
            openml_occurrences.items(), key=lambda item: int(item[0])
        )
    ]
    protocol = {
        "frozen_v79_commit": FROZEN_V79_COMMIT,
        "parent_v79_evidence_digest": V79_EVIDENCE_DIGEST,
        "refs": refs,
        "scanned_roots": [
            "mini-origin/",
            "research-evidence/",
            ".github/workflows/",
        ],
        "openml_id_patterns": [pattern.pattern for pattern in OPENML_ID_PATTERNS],
        "dataset_name_normalization": "lowercase alphanumeric tokens joined by hyphens",
        "candidate_metadata_access": False,
    }
    report = {
        "status": "cross_source_preblind_registry_v80_complete",
        "protocol": protocol,
        "ref_count": len(refs),
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "failures": failures,
        "parent_v79_evidence_digest": V79_EVIDENCE_DIGEST,
        "frozen_v79_commit": FROZEN_V79_COMMIT,
        "excluded_uci_ids": sorted(discovered_uci_ids),
        "excluded_uci_id_count": len(discovered_uci_ids),
        "excluded_openml_dataset_ids": [
            row["openml_dataset_id"] for row in openml_rows
        ],
        "excluded_openml_dataset_id_count": len(openml_rows),
        "excluded_dataset_names": sorted(names),
        "excluded_dataset_name_count": len(names),
        "openml_occurrences": openml_rows,
        "required_v79_openml_checks": required_openml_checks,
        "required_v79_name_checks": required_name_checks,
        "claim_scope": preregistration["claim_boundary"],
    }
    complete = (
        base_report["status"] == "repository_dataset_registry_complete"
        and not failures
        and all(required_openml_checks.values())
        and all(required_name_checks.values())
    )
    if not complete:
        report["status"] = "cross_source_preblind_registry_v80_incomplete"
    report["registry_digest"] = canonical_digest({
        "protocol": protocol,
        "excluded_uci_ids": report["excluded_uci_ids"],
        "excluded_openml_dataset_ids": report["excluded_openml_dataset_ids"],
        "excluded_dataset_names": report["excluded_dataset_names"],
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "refs": report["ref_count"],
        "files": report["files_scanned"],
        "bytes": report["bytes_scanned"],
        "excluded_uci_ids": report["excluded_uci_id_count"],
        "excluded_openml_dataset_ids": report["excluded_openml_dataset_id_count"],
        "excluded_dataset_names": report["excluded_dataset_name_count"],
        "required_v79_openml_checks": all(report["required_v79_openml_checks"].values()),
        "required_v79_name_checks": all(report["required_v79_name_checks"].values()),
        "registry_digest": report["registry_digest"],
    }, indent=2))
    if report["status"] != "cross_source_preblind_registry_v80_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
