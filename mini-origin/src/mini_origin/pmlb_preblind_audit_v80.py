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
    / "v80-pmlb-preblind-audit.json"
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
PMLB_PATTERNS = (
    re.compile(r"pmlb\.fetch_data\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"(?<![.\w])fetch_data\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(
        r"EpistasisLab/pmlb/(?:(?:raw|tree)/[^/]+/)?datasets/([^/'\"\s]+)",
        re.IGNORECASE,
    ),
    re.compile(r"datasets/([^/'\"\s]+)/[^/'\"\s]+\.tsv\.gz", re.IGNORECASE),
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


def extract_pmlb_names(text: str) -> set[str]:
    names: set[str] = set()
    for pattern in PMLB_PATTERNS:
        names.update(normalize_name(value) for value in pattern.findall(text))
    return {name for name in names if name}


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


def load_inputs() -> tuple[dict[str, object], dict[str, object]]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_pmlb_catalogue_access":
        raise RuntimeError("v0.80 preregistration status changed")
    if preregistration["parent_v79_commit"] != FROZEN_V79_COMMIT:
        raise RuntimeError("frozen v0.79 commit changed")
    if preregistration["parent_v79_evidence_digest"] != V79_EVIDENCE_DIGEST:
        raise RuntimeError("v0.79 evidence commitment changed")
    for key in (
        "pmlb_catalogue_or_repository_tree_access_before_audit",
        "pmlb_candidate_names_ids_or_urls_committed_before_audit",
        "pmlb_dataset_bytes_access_before_audit",
        "solver_execution_before_audit",
    ):
        if preregistration[key] is not False:
            raise RuntimeError(f"preblind boundary violated: {key}")

    parent = json.loads(V79_EVIDENCE.read_text(encoding="utf-8"))
    if parent["status"] != "small_query_coverage_development_pass_v79":
        raise RuntimeError("v0.79 parent must remain a pass")
    if parent["evidence_digest"] != V79_EVIDENCE_DIGEST:
        raise RuntimeError("unexpected v0.79 evidence digest")
    if parent["development_gate"] is not True:
        raise RuntimeError("v0.79 gate changed")
    if int(parent["rust_mismatch_count"]) != 0:
        raise RuntimeError("v0.79 Rust mismatches changed")
    if int(parent["normal_state_digest_check"]["mismatch_count"]) != 0:
        raise RuntimeError("v0.79 normal-state mismatch changed")
    if int(parent["label_independence_certificate"]["mismatch_count"]) != 0:
        raise RuntimeError("v0.79 label-independence mismatch changed")
    selected_ids = tuple(
        int(row["uci_id"]) for row in parent["dataset_summaries"]
    )
    if selected_ids != REQUIRED_V79_OPENML_IDS:
        raise RuntimeError("v0.79 OpenML dataset order changed")
    return preregistration, parent


def occurrence_rows(
    mapping: dict[str, list[dict[str, object]]],
    key_name: str,
    numeric: bool = False,
) -> list[dict[str, object]]:
    order = (
        (lambda item: int(item[0])) if numeric else (lambda item: item[0])
    )
    return [
        {
            key_name: int(key) if numeric else key,
            "occurrence_count_capped": len(occurrences),
            "occurrences": occurrences,
        }
        for key, occurrences in sorted(mapping.items(), key=order)
    ]


def audit() -> dict[str, object]:
    preregistration, parent = load_inputs()
    refs = base.remote_refs()
    uci_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    openml_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    pystreed_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    pmlb_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    names: set[str] = set()
    files_scanned = 0
    bytes_scanned = 0
    failures: list[dict[str, str]] = []

    for ref in refs:
        for path, size in base.tree_files(ref):
            try:
                text = base.show_text(ref, path)
            except Exception as error:
                failures.append({
                    "ref": ref,
                    "path": path,
                    "error": str(error)[-500:],
                })
                continue
            files_scanned += 1
            bytes_scanned += size
            occurrence = {"ref": ref, "path": path}

            uci_ids: set[str] = set()
            for pattern in base.UCI_PATTERNS:
                uci_ids.update(pattern.findall(text))
            for uci_id in sorted(uci_ids, key=int):
                base.append_occurrence(uci_occurrences, uci_id, occurrence)

            for name, uci_id in base.NAME_ID_PATTERN.findall(text):
                names.add(normalize_name(name))
                base.append_occurrence(uci_occurrences, uci_id, occurrence)
            for uci_id, name in base.ID_NAME_PATTERN.findall(text):
                names.add(normalize_name(name))
                base.append_occurrence(uci_occurrences, uci_id, occurrence)

            for token in base.PYSTREED_PATTERN.findall(text):
                canonical = normalize_name(token)
                base.append_occurrence(pystreed_occurrences, canonical, occurrence)

            for openml_id in sorted(extract_openml_ids(text)):
                base.append_occurrence(
                    openml_occurrences, str(openml_id), occurrence
                )
            for name, openml_id in OPENML_NAME_ID_PATTERN.findall(text):
                names.add(normalize_name(name))
                base.append_occurrence(openml_occurrences, openml_id, occurrence)
            for openml_id, name in OPENML_ID_NAME_PATTERN.findall(text):
                names.add(normalize_name(name))
                base.append_occurrence(openml_occurrences, openml_id, occurrence)

            names.update(extract_dataset_names(text, Path(path).suffix.lower()))
            for pmlb_name in sorted(extract_pmlb_names(text)):
                names.add(pmlb_name)
                base.append_occurrence(
                    pmlb_occurrences, pmlb_name, occurrence
                )

    required_names = {
        normalize_name(str(row["task"]))
        for row in parent["dataset_summaries"]
    }
    required_openml_checks = {
        str(value): str(value) in openml_occurrences
        for value in REQUIRED_V79_OPENML_IDS
    }
    required_name_checks = {
        name: name in names for name in sorted(required_names)
    }
    known_checks = {
        "uci_3_annealing": "3" in uci_occurrences,
        "uci_83_primary_tumor": "83" in uci_occurrences,
        "uci_90_soybean_large": "90" in uci_occurrences,
        "pystreed_annealing": "annealing" in pystreed_occurrences,
    }

    uci_rows = occurrence_rows(uci_occurrences, "uci_id", numeric=True)
    openml_rows = occurrence_rows(
        openml_occurrences, "openml_dataset_id", numeric=True
    )
    pystreed_rows = occurrence_rows(pystreed_occurrences, "token")
    pmlb_rows = occurrence_rows(pmlb_occurrences, "normalized_name")
    protocol = {
        "frozen_v79_commit": FROZEN_V79_COMMIT,
        "parent_v79_evidence_digest": V79_EVIDENCE_DIGEST,
        "refs": refs,
        "scan_roots": preregistration["audit_protocol"]["scan_roots"],
        "text_suffixes": sorted(base.TEXT_SUFFIXES),
        "max_file_bytes": base.MAX_FILE_BYTES,
        "uci_patterns": [pattern.pattern for pattern in base.UCI_PATTERNS],
        "openml_id_patterns": [
            pattern.pattern for pattern in OPENML_ID_PATTERNS
        ],
        "pystreed_pattern": base.PYSTREED_PATTERN.pattern,
        "pmlb_patterns": [pattern.pattern for pattern in PMLB_PATTERNS],
        "dataset_name_normalization": preregistration["audit_protocol"][
            "dataset_name_normalization"
        ],
        "pmlb_catalogue_access": False,
        "pmlb_repository_tree_access": False,
        "pmlb_dataset_byte_access": False,
    }
    complete = (
        not failures
        and all(known_checks.values())
        and all(required_openml_checks.values())
        and all(required_name_checks.values())
    )
    report = {
        "status": (
            "pmlb_preblind_registry_v80_complete"
            if complete else "pmlb_preblind_registry_v80_incomplete"
        ),
        "protocol": protocol,
        "ref_count": len(refs),
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "failures": failures,
        "known_contamination_checks": known_checks,
        "parent_v79_evidence_digest": V79_EVIDENCE_DIGEST,
        "frozen_v79_commit": FROZEN_V79_COMMIT,
        "excluded_uci_ids": [row["uci_id"] for row in uci_rows],
        "excluded_uci_id_count": len(uci_rows),
        "excluded_openml_dataset_ids": [
            row["openml_dataset_id"] for row in openml_rows
        ],
        "excluded_openml_dataset_id_count": len(openml_rows),
        "excluded_dataset_names": sorted(names),
        "excluded_dataset_name_count": len(names),
        "pystreed_dataset_tokens": pystreed_rows,
        "explicit_pmlb_name_occurrences": pmlb_rows,
        "explicit_pmlb_name_count": len(pmlb_rows),
        "uci_occurrences": uci_rows,
        "openml_occurrences": openml_rows,
        "required_v79_openml_checks": required_openml_checks,
        "required_v79_name_checks": required_name_checks,
        "claim_scope": preregistration["claim_boundary"],
    }
    report["registry_digest"] = canonical_digest({
        "protocol": protocol,
        "known_contamination_checks": known_checks,
        "excluded_uci_ids": report["excluded_uci_ids"],
        "excluded_openml_dataset_ids": report["excluded_openml_dataset_ids"],
        "excluded_dataset_names": report["excluded_dataset_names"],
        "pystreed_tokens": [row["token"] for row in pystreed_rows],
        "explicit_pmlb_names": [row["normalized_name"] for row in pmlb_rows],
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
        "failures": len(report["failures"]),
        "excluded_uci_ids": report["excluded_uci_id_count"],
        "excluded_openml_ids": report["excluded_openml_dataset_id_count"],
        "excluded_names": report["excluded_dataset_name_count"],
        "pmlb_names": report["explicit_pmlb_name_count"],
        "known_checks": all(report["known_contamination_checks"].values()),
        "required_openml": all(report["required_v79_openml_checks"].values()),
        "required_names": all(report["required_v79_name_checks"].values()),
        "registry_digest": report["registry_digest"],
    }, indent=2))
    if report["status"] != "pmlb_preblind_registry_v80_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
