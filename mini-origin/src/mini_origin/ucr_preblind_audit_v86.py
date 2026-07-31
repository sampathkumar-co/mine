from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from . import pmlb_preblind_audit_v80 as prior
from . import repository_dataset_audit_v63 as base


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v86-ucr-preblind-audit.json"
)
PROTOCOL_AMENDMENT = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v86-ucr-preblind-protocol-amendment.json"
)
V85_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v85-authoritative-opened-data-development-pass.json"
)
V85_REPRODUCIBILITY = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v85-exact-rerun-reproducibility.json"
)
V80_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v80-pmlb-preblind-registry.json"
)
V82_PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v82-pmlb-blind.json"
)
FROZEN_V85_COMMIT = "912c3ebd933ae39eb05e10467f1ecad56e326b03"
V85_EVIDENCE_DIGEST = "ca99dd822bc57fca55ffaf6de3614c7403cdbc0c85f3db81e0652dfe0acc0c20"
V85_STATE_INPUT_SHA256 = "3e784b85ff4cf38ec4908fa1da4c57b164ab62b6f3885e6e9c57881436f0c7ac"
V80_REGISTRY_DIGEST = "aa6bab47d8d2453b669eee2f7a36720e0eb798a79355b0d2c9a509d81959038c"
V80_REGISTRY_SHA256 = "242a3b4173786ab52435658c959ce6195c3f831c25cf3dff4bba2d784796a3f9"

UCR_NAME_PATTERNS = (
    re.compile(
        r"timeseriesclassification\.com/description\.php\?Dataset=([^&#\"'\s]+)",
        re.IGNORECASE,
    ),
    re.compile(r"UCRArchive_20\d{2}/([^/\"'\s]+)", re.IGNORECASE),
    re.compile(
        r"load_UCR_UEA_dataset\([^)]*?name\s*=\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    re.compile(
        r"UCR_UEA_datasets\(\)\.load_dataset\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:aeon\.datasets\.)?load_classification\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
)
UCR_SOURCE_PATTERN = re.compile(
    r"(?:timeseriesclassification\.com|UCRArchive_20\d{2}|"
    r"UCR[_ /-]UEA|UCR Time Series Classification Archive)",
    re.IGNORECASE,
)
NAMED_DATASET_PATTERNS = (
    re.compile(
        r"['\"](?:dataset_name|dataset|task_name|benchmark_dataset)['\"]"
        r"\s*[:=]\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:dataset_name|dataset|task_name|benchmark_dataset)\s*:\s*"
        r"['\"]?([^#\r\n'\"]+?)['\"]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
)

normalize_name = prior.normalize_name
canonical_digest = prior.canonical_digest
occurrence_rows = prior.occurrence_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_ucr_names(text: str) -> set[str]:
    names: set[str] = set()
    for pattern in UCR_NAME_PATTERNS:
        names.update(normalize_name(value) for value in pattern.findall(text))
    return {name for name in names if name}


def collect_json_named_datasets(value: object, output: set[str]) -> None:
    if isinstance(value, dict):
        lowered = {str(key).lower(): child for key, child in value.items()}
        for key in ("dataset_name", "dataset", "task_name", "benchmark_dataset"):
            candidate = lowered.get(key)
            if isinstance(candidate, str) and candidate.strip():
                output.add(normalize_name(candidate))
        context_keys = {
            "uci_id",
            "openml_id",
            "openml_dataset_id",
            "dataset_id",
            "raw_sha256",
            "raw_md5",
            "raw_path",
            "source_url",
            "instances",
            "features",
            "classes",
            "record_count",
            "target",
        }
        candidate = lowered.get("name")
        if context_keys & set(lowered) and isinstance(candidate, str) and candidate.strip():
            output.add(normalize_name(candidate))
        for child in value.values():
            collect_json_named_datasets(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_json_named_datasets(child, output)


def extract_named_dataset_values(text: str, suffix: str) -> set[str]:
    names: set[str] = set()
    for pattern in NAMED_DATASET_PATTERNS:
        names.update(normalize_name(value.strip()) for value in pattern.findall(text))
    if suffix == ".json":
        try:
            collect_json_named_datasets(json.loads(text), names)
        except json.JSONDecodeError:
            pass
    return {name for name in names if name}


def load_inputs() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_ucr_catalogue_access":
        raise RuntimeError("v0.86 preregistration status changed")
    amendment = json.loads(PROTOCOL_AMENDMENT.read_text(encoding="utf-8"))
    if preregistration["protocol_amendment"] != PROTOCOL_AMENDMENT.name:
        raise RuntimeError("v0.86 amendment reference changed")
    if amendment["status"] != "protocol_amendment_before_completed_audit_or_ucr_catalogue_access":
        raise RuntimeError("v0.86 protocol amendment status changed")
    for key in (
        "catalogue_access_before_amendment",
        "candidate_dataset_metadata_access_before_amendment",
        "candidate_dataset_names_accessed_before_amendment",
        "candidate_dataset_bytes_accessed_before_amendment",
        "records_or_labels_accessed_before_amendment",
        "solver_execution_before_amendment",
    ):
        if amendment[key] is not False:
            raise RuntimeError(f"amendment preaccess boundary violated: {key}")
    if preregistration["parent_v85_commit"] != FROZEN_V85_COMMIT:
        raise RuntimeError("frozen v0.85 commit changed")
    if preregistration["parent_v85_authoritative_evidence_digest"] != V85_EVIDENCE_DIGEST:
        raise RuntimeError("v0.85 evidence commitment changed")
    if preregistration["parent_v85_state_input_sha256"] != V85_STATE_INPUT_SHA256:
        raise RuntimeError("v0.85 state commitment changed")
    if preregistration["parent_v80_registry_digest"] != V80_REGISTRY_DIGEST:
        raise RuntimeError("v0.80 registry commitment changed")
    if preregistration["parent_v80_registry_sha256"] != V80_REGISTRY_SHA256:
        raise RuntimeError("v0.80 registry file commitment changed")

    source = preregistration["future_source"]
    for key in (
        "catalogue_access_before_audit",
        "candidate_dataset_names_ids_or_file_urls_committed_before_audit",
        "dataset_bytes_accessed_before_audit",
        "records_or_labels_accessed_before_audit",
    ):
        if source[key] is not False:
            raise RuntimeError(f"preblind boundary violated: {key}")
    if preregistration["solver_execution_before_audit"] is not False:
        raise RuntimeError("solver execution occurred before audit")
    if preregistration["audit_protocol"]["network_access"] is not False:
        raise RuntimeError("audit network boundary changed")
    blind_gate = preregistration["future_blind_gate"]
    if blind_gate["gate_source"] != "exact v0.82 locked gate with future UCR datasets treated as the seven fresh datasets":
        raise RuntimeError("future blind gate source changed")
    v82 = json.loads(V82_PREREGISTRATION.read_text(encoding="utf-8"))
    if blind_gate["exact_budget"] != v82["exact_budget"]:
        raise RuntimeError("future exact budget differs from v0.82")
    if blind_gate["budget_ladder"] != v82["budget_ladder"]:
        raise RuntimeError("future budget ladder differs from v0.82")
    locked = v82["locked_gate"]
    for key, expected in locked.items():
        if key in {"previously_zero_datasets", "minimum_states_from_each_previously_zero_dataset"}:
            continue
        if blind_gate.get(key) != expected:
            raise RuntimeError(f"future blind gate differs from v0.82: {key}")
    if (
        blind_gate["minimum_states_from_each_fresh_holdout_dataset"]
        != locked["minimum_states_from_each_previously_zero_dataset"]
    ):
        raise RuntimeError("future fresh-dataset minimum differs from v0.82")
    for key in (
        "algorithm_revisions",
        "compiler_revisions",
        "selector_revisions",
        "solver_revisions",
        "scientific_threshold_revisions",
        "acceptance_gate_revisions",
    ):
        if int(preregistration[key]) != 0:
            raise RuntimeError(f"unexpected scientific revision: {key}")

    evidence = json.loads(V85_EVIDENCE.read_text(encoding="utf-8"))
    if evidence["verdict"] != "development_pass":
        raise RuntimeError("v0.85 parent is not the frozen development pass")
    if evidence["evidence_digest"] != V85_EVIDENCE_DIGEST:
        raise RuntimeError("unexpected v0.85 evidence digest")
    if evidence["state_input_sha256"] != V85_STATE_INPUT_SHA256:
        raise RuntimeError("unexpected v0.85 state digest")
    gates = evidence["gate_results"]
    required = preregistration["required_parent_checks"]
    if gates["development_gate"] is not required["v85_development_gate"]:
        raise RuntimeError("v0.85 development gate changed")
    if int(gates["contributing_dataset_count"]) != int(required["v85_contributing_dataset_count"]):
        raise RuntimeError("v0.85 contributing dataset count changed")
    if int(gates["profiled_state_count"]) != int(required["v85_profiled_state_count"]):
        raise RuntimeError("v0.85 profiled state count changed")
    if int(evidence["rust_crosscheck"]["mismatch_count"]) != int(required["v85_rust_mismatch_count"]):
        raise RuntimeError("v0.85 Rust mismatch count changed")
    if gates["fresh_external_evidence"] is not required["v85_fresh_external_evidence"]:
        raise RuntimeError("v0.85 claim boundary changed")

    reproducibility = json.loads(V85_REPRODUCIBILITY.read_text(encoding="utf-8"))
    if reproducibility["verdict"] != required["v85_exact_rerun_verdict"]:
        raise RuntimeError("v0.85 exact rerun verdict changed")
    comparison = reproducibility["comparison"]
    if comparison["states_byte_identical"] is not True:
        raise RuntimeError("v0.85 rerun state bytes changed")
    if comparison["python_reference_byte_identical"] is not True:
        raise RuntimeError("v0.85 rerun Python reference changed")
    if comparison["scientific_gate_fields_identical"] is not True:
        raise RuntimeError("v0.85 rerun scientific fields changed")

    prior_registry = json.loads(V80_REGISTRY.read_text(encoding="utf-8"))
    if sha256(V80_REGISTRY) != V80_REGISTRY_SHA256:
        raise RuntimeError("v0.80 registry file bytes changed")
    if prior_registry["status"] != "pmlb_preblind_registry_v80_complete":
        raise RuntimeError("v0.80 registry is not complete")
    if prior_registry["registry_digest"] != V80_REGISTRY_DIGEST:
        raise RuntimeError("v0.80 registry digest changed")
    return preregistration, evidence, reproducibility, prior_registry


def audit() -> dict[str, object]:
    preregistration, evidence, reproducibility, prior_registry = load_inputs()
    refs = base.remote_refs()
    uci_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    openml_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    pmlb_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    ucr_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    source_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
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
            suffix = Path(path).suffix.lower()

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

            for openml_id in sorted(prior.extract_openml_ids(text, suffix)):
                base.append_occurrence(openml_occurrences, str(openml_id), occurrence)
            names.update(prior.extract_dataset_names(text, suffix))
            names.update(extract_named_dataset_values(text, suffix))

            for pmlb_name in sorted(prior.extract_pmlb_names(text)):
                names.add(pmlb_name)
                base.append_occurrence(pmlb_occurrences, pmlb_name, occurrence)

            ucr_names = extract_ucr_names(text)
            for ucr_name in sorted(ucr_names):
                names.add(ucr_name)
                base.append_occurrence(ucr_occurrences, ucr_name, occurrence)
            if UCR_SOURCE_PATTERN.search(text):
                base.append_occurrence(source_occurrences, "ucr-archive", occurrence)

    uci_rows = occurrence_rows(uci_occurrences, "uci_id", numeric=True)
    openml_rows = occurrence_rows(openml_occurrences, "openml_dataset_id", numeric=True)
    pmlb_rows = occurrence_rows(pmlb_occurrences, "normalized_name")
    ucr_rows = occurrence_rows(ucr_occurrences, "normalized_name")

    baseline_uci = {int(value) for value in prior_registry["excluded_uci_ids"]}
    baseline_openml = {int(value) for value in prior_registry["excluded_openml_dataset_ids"]}
    baseline_names = {normalize_name(value) for value in prior_registry["excluded_dataset_names"]}
    scanned_uci = {int(row["uci_id"]) for row in uci_rows}
    scanned_openml = {int(row["openml_dataset_id"]) for row in openml_rows}
    v85_names = {
        normalize_name(value)
        for value in evidence["selected_dataset_state_set_digests"]
    }
    excluded_names = sorted(names | baseline_names | v85_names)

    baseline_checks = {
        "v80_uci_ids_missing": sorted(baseline_uci - scanned_uci),
        "v80_openml_ids_missing": sorted(baseline_openml - scanned_openml),
        "v80_names_missing": sorted(baseline_names - names),
        "v85_selected_names_missing": sorted(v85_names - names),
    }
    known_checks = {
        "uci_3_annealing": 3 in scanned_uci,
        "openml_1068": 1068 in scanned_openml,
        "pmlb_vowel": "vowel" in {row["normalized_name"] for row in pmlb_rows},
        "v85_exact_rerun": reproducibility["verdict"] == "reproduced",
    }
    complete = (
        not failures
        and all(known_checks.values())
        and all(not values for values in baseline_checks.values())
    )
    protocol = {
        "parent_v85_commit": FROZEN_V85_COMMIT,
        "parent_v85_evidence_digest": V85_EVIDENCE_DIGEST,
        "parent_v85_state_input_sha256": V85_STATE_INPUT_SHA256,
        "parent_v80_registry_digest": V80_REGISTRY_DIGEST,
        "refs": refs,
        "scan_roots": preregistration["audit_protocol"]["scan_roots"],
        "text_suffixes": sorted(base.TEXT_SUFFIXES),
        "max_file_bytes": base.MAX_FILE_BYTES,
        "dataset_name_normalization": preregistration["audit_protocol"]["dataset_name_normalization"],
        "ucr_name_patterns": [pattern.pattern for pattern in UCR_NAME_PATTERNS],
        "ucr_source_pattern": UCR_SOURCE_PATTERN.pattern,
        "network_access": False,
        "ucr_catalogue_access": False,
        "ucr_dataset_byte_access": False,
        "record_or_label_access": False,
        "solver_execution": False,
    }
    report = {
        "status": (
            "ucr_preblind_registry_v86_complete"
            if complete else "ucr_preblind_registry_v86_incomplete"
        ),
        "protocol": protocol,
        "ref_count": len(refs),
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "failures": failures,
        "known_contamination_checks": known_checks,
        "baseline_preservation_checks": baseline_checks,
        "excluded_uci_ids": [row["uci_id"] for row in uci_rows],
        "excluded_uci_id_count": len(uci_rows),
        "excluded_openml_dataset_ids": [row["openml_dataset_id"] for row in openml_rows],
        "excluded_openml_dataset_id_count": len(openml_rows),
        "excluded_dataset_names": excluded_names,
        "excluded_dataset_name_count": len(excluded_names),
        "explicit_pmlb_name_occurrences": pmlb_rows,
        "explicit_pmlb_name_count": len(pmlb_rows),
        "explicit_ucr_name_occurrences": ucr_rows,
        "explicit_ucr_name_count": len(ucr_rows),
        "ucr_source_occurrences": source_occurrences.get("ucr-archive", []),
        "ucr_source_occurrence_count_capped": len(source_occurrences.get("ucr-archive", [])),
        "uci_occurrences": uci_rows,
        "openml_occurrences": openml_rows,
        "frozen_future_source": preregistration["future_source"],
        "frozen_future_selection": preregistration["future_metadata_only_selection"],
        "frozen_future_blind_gate": preregistration["future_blind_gate"],
        "claim_scope": preregistration["claim_boundary"],
    }
    report["registry_digest"] = canonical_digest({
        "protocol": protocol,
        "known_contamination_checks": known_checks,
        "excluded_uci_ids": report["excluded_uci_ids"],
        "excluded_openml_dataset_ids": report["excluded_openml_dataset_ids"],
        "excluded_dataset_names": report["excluded_dataset_names"],
        "explicit_pmlb_names": [row["normalized_name"] for row in pmlb_rows],
        "explicit_ucr_names": [row["normalized_name"] for row in ucr_rows],
        "frozen_future_source": report["frozen_future_source"],
        "frozen_future_selection": report["frozen_future_selection"],
        "frozen_future_blind_gate": report["frozen_future_blind_gate"],
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
        "ucr_names": report["explicit_ucr_name_count"],
        "baseline_preserved": all(
            not values for values in report["baseline_preservation_checks"].values()
        ),
        "registry_digest": report["registry_digest"],
    }, indent=2))
    if report["status"] != "ucr_preblind_registry_v86_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
