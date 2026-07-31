from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

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
REVIEW_AMENDMENT = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v86-ucr-preblind-review-amendment.json"
)
SECOND_REVIEW_AMENDMENT = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v86-ucr-preblind-second-review-amendment.json"
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
AUDIT_ROOTS = (
    "mini-origin/",
    "research-evidence/",
    ".github/workflows/",
)
AUDIT_MAX_FILE_BYTES = 2_000_000
ASCII_EDGE_WHITESPACE = "\t\n\v\f\r "
CANONICAL_METADATA_FIELDS = (
    "normalized_dataset_name",
    "total_instances",
    "series_length",
    "class_count",
    "classification",
    "univariate",
    "train_url",
    "test_url",
)

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


def normalize_protocol_string(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("protocol string must be str")
    return unicodedata.normalize("NFC", value.strip(ASCII_EDGE_WHITESPACE))


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_metadata_bytes(metadata: dict[str, object]) -> bytes:
    if set(metadata) != set(CANONICAL_METADATA_FIELDS):
        raise ValueError("canonical metadata fields differ from frozen schema")
    for key in ("normalized_dataset_name", "train_url", "test_url"):
        if not isinstance(metadata[key], str):
            raise TypeError(f"{key} must be a string")
    normalized_name = normalize_name(
        normalize_protocol_string(metadata["normalized_dataset_name"])
    )
    if normalized_name != metadata["normalized_dataset_name"]:
        raise ValueError("dataset name is not already in frozen normalized form")
    integers = {}
    for key in ("total_instances", "series_length", "class_count"):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be a JSON integer")
        integers[key] = value
    for key in ("classification", "univariate"):
        if metadata[key] is not True:
            raise ValueError(f"{key} must be true")
    canonical = {
        "normalized_dataset_name": normalized_name,
        "total_instances": integers["total_instances"],
        "series_length": integers["series_length"],
        "class_count": integers["class_count"],
        "classification": True,
        "univariate": True,
        "train_url": normalize_protocol_string(metadata["train_url"]),
        "test_url": normalize_protocol_string(metadata["test_url"]),
    }
    return canonical_json_bytes(canonical)


def canonical_metadata_digest(metadata: dict[str, object]) -> str:
    return hashlib.sha256(canonical_metadata_bytes(metadata)).hexdigest()


def ranking_bytes(
    selection_seed: str,
    frozen_source_release: str,
    normalized_dataset_name: str,
    metadata_digest: str,
) -> bytes:
    normalized_name = normalize_name(normalize_protocol_string(normalized_dataset_name))
    if normalized_name != normalized_dataset_name:
        raise ValueError("ranking dataset name is not normalized")
    if not isinstance(normalized_dataset_name, str):
        raise TypeError("ranking dataset name must be str")
    if not isinstance(metadata_digest, str):
        raise TypeError("canonical metadata digest must be str")
    digest = metadata_digest
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("canonical metadata digest must be lowercase SHA-256 hex")
    return canonical_json_bytes([
        normalize_protocol_string(selection_seed),
        normalize_protocol_string(frozen_source_release),
        normalized_name,
        digest,
    ])


def ranking_digest(
    selection_seed: str,
    frozen_source_release: str,
    normalized_dataset_name: str,
    metadata_digest: str,
) -> str:
    return hashlib.sha256(ranking_bytes(
        selection_seed,
        frozen_source_release,
        normalized_dataset_name,
        metadata_digest,
    )).hexdigest()


def resolve_ref(ref: str) -> str:
    sha = base.git("rev-parse", f"{ref}^{{commit}}").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError(f"invalid commit SHA for {ref}: {sha}")
    return sha


def audit_ref_snapshots() -> tuple[dict[str, str], ...]:
    rows = [{"kind": "checked_out_head", "ref": "HEAD", "sha": resolve_ref("HEAD")}]
    rows.extend(
        {"kind": "origin_branch", "ref": ref, "sha": resolve_ref(ref)}
        for ref in base.remote_refs()
    )
    return tuple(sorted(rows, key=lambda row: (row["kind"], row["ref"], row["sha"])))


def git_output_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def all_tree_blobs(commit_sha: str) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    raw = git_output_bytes(
        "-c",
        "core.quotepath=false",
        "ls-tree",
        "-r",
        "--long",
        "-z",
        commit_sha,
    )
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            metadata_bytes, path_bytes = entry.split(b"\t", 1)
        except ValueError as error:
            raise RuntimeError("malformed NUL-delimited git tree entry") from error
        parts = metadata_bytes.decode("ascii", errors="strict").split()
        if len(parts) < 4 or parts[1] != "blob":
            continue
        path = path_bytes.decode("utf-8", errors="surrogateescape")
        if not path.startswith(AUDIT_ROOTS):
            continue
        rows.append({
            "path": path,
            "size": int(parts[3]),
            "blob_sha": parts[2],
        })
    return tuple(sorted(rows, key=lambda row: (row["path"], row["blob_sha"])))


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
    review_amendment = json.loads(REVIEW_AMENDMENT.read_text(encoding="utf-8"))
    if preregistration["review_amendment"] != REVIEW_AMENDMENT.name:
        raise RuntimeError("v0.86 review amendment reference changed")
    if review_amendment["status"] != "review_amendment_before_ucr_catalogue_access":
        raise RuntimeError("v0.86 review amendment status changed")
    for key in (
        "catalogue_access_before_amendment",
        "candidate_dataset_metadata_access_before_amendment",
        "candidate_dataset_names_accessed_before_amendment",
        "candidate_dataset_file_urls_accessed_before_amendment",
        "candidate_dataset_bytes_accessed_before_amendment",
        "records_or_labels_accessed_before_amendment",
        "solver_execution_before_amendment",
        "external_dataset_network_access_before_amendment",
    ):
        if review_amendment[key] is not False:
            raise RuntimeError(f"review-amendment boundary violated: {key}")
    second_review = json.loads(SECOND_REVIEW_AMENDMENT.read_text(encoding="utf-8"))
    if preregistration["second_review_amendment"] != SECOND_REVIEW_AMENDMENT.name:
        raise RuntimeError("v0.86 second review amendment reference changed")
    if second_review["status"] != "second_review_amendment_before_ucr_catalogue_access":
        raise RuntimeError("v0.86 second review amendment status changed")
    for key in (
        "catalogue_access_before_amendment",
        "candidate_dataset_metadata_access_before_amendment",
        "candidate_dataset_names_accessed_before_amendment",
        "candidate_dataset_file_urls_accessed_before_amendment",
        "candidate_dataset_bytes_accessed_before_amendment",
        "records_or_labels_accessed_before_amendment",
        "solver_execution_before_amendment",
        "external_dataset_network_access_before_amendment",
    ):
        if second_review[key] is not False:
            raise RuntimeError(f"second-review boundary violated: {key}")
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
    audit_protocol = preregistration["audit_protocol"]
    if audit_protocol["repository_git_fetch_allowed"] is not True:
        raise RuntimeError("repository fetch provenance changed")
    if audit_protocol["repository_git_fetch_refspec"] != "+refs/heads/*:refs/remotes/origin/*":
        raise RuntimeError("repository fetch refspec changed")
    if audit_protocol["external_dataset_network_access"] is not False:
        raise RuntimeError("external dataset network boundary changed")
    if audit_protocol["scan_checked_out_head"] is not True:
        raise RuntimeError("checked-out HEAD scan was disabled")
    if audit_protocol["record_ref_commit_sha_pairs"] is not True:
        raise RuntimeError("immutable ref snapshot was disabled")
    if audit_protocol["scan_all_blob_suffixes"] is not True:
        raise RuntimeError("all-suffix blob scan was disabled")
    if int(audit_protocol["maximum_blob_bytes"]) != AUDIT_MAX_FILE_BYTES:
        raise RuntimeError("audit blob size limit changed")
    if audit_protocol["oversized_blob_policy"] != "fail the audit and record ref, commit SHA, blob SHA, path and byte size":
        raise RuntimeError("oversized blob policy changed")
    if audit_protocol["git_tree_entry_encoding"] != "git ls-tree -r --long -z raw bytes; split entries on NUL and metadata/path on the first TAB; decode path bytes with UTF-8 surrogateescape; Git path quoting is forbidden":
        raise RuntimeError("Git tree entry encoding changed")
    if source["active_endpoint"] != "https://www.timeseriesclassification.com/":
        raise RuntimeError("active UCR endpoint changed")
    if source["fallback_permitted_in_v87"] is not False:
        raise RuntimeError("v0.87 endpoint fallback was enabled")
    if source["endpoint_attempts"] != 3:
        raise RuntimeError("endpoint attempt count changed")
    if source["endpoint_retry_delays_seconds"] != [0, 5, 20]:
        raise RuntimeError("endpoint retry schedule changed")
    if source["connect_timeout_seconds"] != 15 or source["read_timeout_seconds"] != 60:
        raise RuntimeError("endpoint timeout policy changed")
    if source["maximum_redirects"] != 5:
        raise RuntimeError("endpoint redirect limit changed")
    if source["tls_certificate_validation"] is not True:
        raise RuntimeError("endpoint TLS validation was disabled")
    if source["all_attempts_failed"] != "fail v0.87 without accessing the inactive fallback endpoint, changing criteria, or selecting datasets":
        raise RuntimeError("endpoint all-attempts-failed policy changed")
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
    selection = preregistration["future_metadata_only_selection"]
    ranking = selection["ranking"]
    if ranking["canonical_metadata_fields"] != list(CANONICAL_METADATA_FIELDS):
        raise RuntimeError("canonical metadata field order changed")
    if ranking["canonical_metadata_digest"] != "lowercase hexadecimal SHA-256 of canonical metadata serialization bytes":
        raise RuntimeError("canonical metadata digest rule changed")
    if ranking["rank"] != "lowercase hexadecimal SHA-256 of ranking serialization bytes":
        raise RuntimeError("ranking hash rule changed")
    lock = selection["byte_lock_only"]
    if lock["selection_before_download"] != "deterministically select the top seven metadata-ranked candidates before downloading any candidate bytes":
        raise RuntimeError("selection-before-download rule changed")
    if lock["selected_file_download_attempts"] != 3:
        raise RuntimeError("selected file attempt count changed")
    if lock["selected_file_retry_delays_seconds"] != [0, 5, 20]:
        raise RuntimeError("selected file retry schedule changed")
    if lock["selected_file_unavailability"] != "fail the entire lock; never substitute a lower-ranked candidate":
        raise RuntimeError("selected file failure policy changed")
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
    ref_snapshots = audit_ref_snapshots()
    refs_by_commit: dict[str, set[str]] = defaultdict(set)
    for row in ref_snapshots:
        refs_by_commit[row["sha"]].add(row["ref"])

    file_occurrences: dict[tuple[str, str], dict[str, object]] = {}
    tree_entry_count = 0
    oversized_files: list[dict[str, object]] = []
    for commit_sha, refs in sorted(refs_by_commit.items()):
        for row in all_tree_blobs(commit_sha):
            tree_entry_count += 1
            occurrence = {
                "refs": sorted(refs),
                "commit_sha": commit_sha,
                "commit_shas": [commit_sha],
                "path": row["path"],
                "blob_sha": row["blob_sha"],
                "size": row["size"],
            }
            if int(row["size"]) > AUDIT_MAX_FILE_BYTES:
                oversized_files.append(occurrence)
                continue
            key = (str(row["blob_sha"]), str(row["path"]))
            existing = file_occurrences.get(key)
            if existing is None:
                file_occurrences[key] = occurrence
            else:
                existing["refs"] = sorted(set(existing["refs"]) | set(refs))
                existing["commit_shas"] = sorted(
                    set(existing["commit_shas"]) | {commit_sha}
                )

    uci_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    openml_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    pmlb_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    ucr_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    source_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    names: set[str] = set()
    files_scanned = 0
    bytes_scanned = 0
    failures: list[dict[str, str]] = []
    scanned_suffixes: set[str] = set()

    for (_, path), occurrence in sorted(file_occurrences.items(), key=lambda item: item[0]):
        try:
            text = base.show_text(str(occurrence["commit_sha"]), path)
        except Exception as error:
            failures.append({
                "refs": occurrence["refs"],
                "commit_sha": str(occurrence["commit_sha"]),
                "path": path,
                "blob_sha": str(occurrence["blob_sha"]),
                "error": str(error)[-500:],
            })
            continue
        files_scanned += 1
        bytes_scanned += int(occurrence["size"])
        suffix = Path(path).suffix.lower() or "<none>"
        scanned_suffixes.add(suffix)
        compact_occurrence = {
            "refs": occurrence["refs"],
            "commit_shas": occurrence["commit_shas"],
            "path": path,
            "blob_sha": occurrence["blob_sha"],
        }

        uci_ids: set[str] = set()
        for pattern in base.UCI_PATTERNS:
            uci_ids.update(pattern.findall(text))
        for uci_id in sorted(uci_ids, key=int):
            base.append_occurrence(uci_occurrences, uci_id, compact_occurrence)
        for name, uci_id in base.NAME_ID_PATTERN.findall(text):
            names.add(normalize_name(name))
            base.append_occurrence(uci_occurrences, uci_id, compact_occurrence)
        for uci_id, name in base.ID_NAME_PATTERN.findall(text):
            names.add(normalize_name(name))
            base.append_occurrence(uci_occurrences, uci_id, compact_occurrence)

        for openml_id in sorted(prior.extract_openml_ids(text, suffix)):
            base.append_occurrence(openml_occurrences, str(openml_id), compact_occurrence)
        names.update(prior.extract_dataset_names(text, suffix))
        names.update(extract_named_dataset_values(text, suffix))

        for pmlb_name in sorted(prior.extract_pmlb_names(text)):
            names.add(pmlb_name)
            base.append_occurrence(pmlb_occurrences, pmlb_name, compact_occurrence)

        ucr_names = extract_ucr_names(text + "\n" + path.replace("\\", "/"))
        for ucr_name in sorted(ucr_names):
            names.add(ucr_name)
            base.append_occurrence(ucr_occurrences, ucr_name, compact_occurrence)
        if UCR_SOURCE_PATTERN.search(text) or UCR_SOURCE_PATTERN.search(path):
            base.append_occurrence(source_occurrences, "ucr-archive", compact_occurrence)

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
        "checked_out_head_recorded": any(
            row["kind"] == "checked_out_head" and row["ref"] == "HEAD"
            for row in ref_snapshots
        ),
        "shell_files_scanned": ".sh" in scanned_suffixes,
        "rust_files_scanned": ".rs" in scanned_suffixes,
        "javascript_module_files_scanned": ".mjs" in scanned_suffixes,
        "cpp_files_scanned": ".cpp" in scanned_suffixes,
    }
    complete = (
        not failures
        and not oversized_files
        and all(known_checks.values())
        and all(not values for values in baseline_checks.values())
    )
    protocol = {
        "parent_v85_commit": FROZEN_V85_COMMIT,
        "parent_v85_evidence_digest": V85_EVIDENCE_DIGEST,
        "parent_v85_state_input_sha256": V85_STATE_INPUT_SHA256,
        "parent_v80_registry_digest": V80_REGISTRY_DIGEST,
        "repository_git_fetch_allowed": True,
        "repository_git_fetch_refspec": "+refs/heads/*:refs/remotes/origin/*",
        "external_dataset_network_access": False,
        "ref_snapshots": list(ref_snapshots),
        "checked_out_head_sha": next(
            row["sha"] for row in ref_snapshots if row["kind"] == "checked_out_head"
        ),
        "scan_roots": list(AUDIT_ROOTS),
        "scan_all_blob_suffixes": True,
        "scanned_suffixes": sorted(scanned_suffixes),
        "maximum_blob_bytes": AUDIT_MAX_FILE_BYTES,
        "oversized_blob_policy": "fail and record immutable provenance",
        "blob_decoding": "UTF-8 with replacement for invalid blob content byte sequences",
        "git_tree_entry_encoding": preregistration["audit_protocol"]["git_tree_entry_encoding"],
        "dataset_name_normalization": preregistration["audit_protocol"]["dataset_name_normalization"],
        "ucr_name_patterns": [pattern.pattern for pattern in UCR_NAME_PATTERNS],
        "ucr_source_pattern": UCR_SOURCE_PATTERN.pattern,
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
        "ref_count": len(ref_snapshots),
        "unique_commit_count": len(refs_by_commit),
        "tree_entry_count": tree_entry_count,
        "unique_blob_path_count": len(file_occurrences),
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "oversized_files": sorted(
            oversized_files,
            key=lambda row: (row["commit_sha"], row["path"], row["blob_sha"]),
        ),
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
        "baseline_preservation_checks": baseline_checks,
        "oversized_files": report["oversized_files"],
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
        "oversized_files": len(report["oversized_files"]),
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
