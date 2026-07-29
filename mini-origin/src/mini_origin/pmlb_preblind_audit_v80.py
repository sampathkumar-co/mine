from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from . import openml_preblind_audit_v76 as previous


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

PMLB_PATTERNS = (
    re.compile(r"pmlb\.fetch_data\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"fetch_data\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
    re.compile(r"EpistasisLab/pmlb/(?:(?:raw|tree)/[^/]+/)?datasets/([^/'\"\s]+)", re.IGNORECASE),
    re.compile(r"datasets/([^/'\"\s]+)/[^/'\"\s]+\.tsv\.gz", re.IGNORECASE),
)


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_pmlb_names(text: str) -> set[str]:
    names: set[str] = set()
    for pattern in PMLB_PATTERNS:
        names.update(previous.normalize_name(value) for value in pattern.findall(text))
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
    return preregistration, parent


def audit() -> dict[str, object]:
    preregistration, parent = load_inputs()
    inherited = previous.audit()
    if inherited["status"] != "openml_preblind_registry_v76_complete":
        raise RuntimeError("inherited repository registry is incomplete")

    refs = previous.base.remote_refs()
    pmlb_occurrences: dict[str, list[dict[str, str]]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    files_scanned = 0
    bytes_scanned = 0

    for ref in refs:
        for path, size in previous.base.tree_files(ref):
            try:
                text = previous.base.show_text(ref, path)
            except Exception as error:
                failures.append({"ref": ref, "path": path, "error": str(error)[-500:]})
                continue
            files_scanned += 1
            bytes_scanned += size
            occurrence = {"ref": ref, "path": path}
            for name in sorted(extract_pmlb_names(text)):
                if occurrence not in pmlb_occurrences[name]:
                    pmlb_occurrences[name].append(occurrence)

    excluded_names = set(inherited["excluded_dataset_names"])
    excluded_names.update(pmlb_occurrences)
    required_names = {
        previous.normalize_name(str(row["task"]))
        for row in parent["dataset_summaries"]
    }
    required_checks = {
        name: name in excluded_names for name in sorted(required_names)
    }
    pmlb_rows = [
        {
            "normalized_name": name,
            "occurrence_count_capped": len(occurrences),
            "occurrences": occurrences[:40],
        }
        for name, occurrences in sorted(pmlb_occurrences.items())
    ]

    protocol = {
        "frozen_v79_commit": FROZEN_V79_COMMIT,
        "parent_v79_evidence_digest": V79_EVIDENCE_DIGEST,
        "refs": refs,
        "scan_roots": preregistration["audit_protocol"]["scan_roots"],
        "pmlb_patterns": [pattern.pattern for pattern in PMLB_PATTERNS],
        "candidate_catalogue_access": False,
        "dataset_name_normalization": preregistration["audit_protocol"][
            "dataset_name_normalization"
        ],
    }
    report = {
        "status": "pmlb_preblind_registry_v80_complete",
        "protocol": protocol,
        "ref_count": len(refs),
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "failures": failures,
        "parent_v79_evidence_digest": V79_EVIDENCE_DIGEST,
        "frozen_v79_commit": FROZEN_V79_COMMIT,
        "excluded_uci_ids": inherited["excluded_uci_ids"],
        "excluded_openml_dataset_ids": inherited["excluded_openml_dataset_ids"],
        "excluded_dataset_names": sorted(excluded_names),
        "excluded_dataset_name_count": len(excluded_names),
        "explicit_pmlb_name_occurrences": pmlb_rows,
        "explicit_pmlb_name_count": len(pmlb_rows),
        "required_v79_name_checks": required_checks,
        "claim_scope": preregistration["claim_boundary"],
    }
    complete = (
        not failures
        and all(required_checks.values())
        and inherited["parent_v75_evidence_digest"]
        == "db379850b2a517e16d5ea442047ac4933ad06fdcf4d6838d91fc36d72e75bc47"
    )
    if not complete:
        report["status"] = "pmlb_preblind_registry_v80_incomplete"
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
        "excluded_names": report["excluded_dataset_name_count"],
        "explicit_pmlb_names": report["explicit_pmlb_name_count"],
        "required_v79_names": all(report["required_v79_name_checks"].values()),
        "registry_digest": report["registry_digest"],
    }, indent=2))
    if report["status"] != "pmlb_preblind_registry_v80_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
