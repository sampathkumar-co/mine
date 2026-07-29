from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from . import clean_lower_bound_conditioned_v68 as opened
from . import numeric_threshold_frontier_v70 as core
from . import small_query_coverage_v79 as selector


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v82-pmlb-blind.json"
)
MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v81-pmlb-hash-lock.json"
)
V79_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v79-small-query-coverage-pass.json"
)
V81_LOCK_DIGEST = "424c98761001c268b1f0574c4e82282e2a4d9321073bc27c87589a92e0f61d8e"
V81_MANIFEST_SHA256 = "00b9bc1830659acab2b8010136706ec065c1ebf2a416db3a223909e11d099f2d"
V80_REGISTRY_DIGEST = "aa6bab47d8d2453b669eee2f7a36720e0eb798a79355b0d2c9a509d81959038c"
V79_EVIDENCE_DIGEST = "5c28f39546d8e7d988fbc78ce254fa35d2b1d85d49a4b270b50edde1235d3001"
FROZEN_V79_COMMIT = "555c3146111a7726702bb98e0a72f3b214d07190"
FROZEN_V81_COMMIT = "d04379797d5aa3da5328bc8c1f51bfc6d4204f4f"
LEGACY_PARENT_V68_EVIDENCE_DIGEST = "b1fc70852a2ad35d91972889eb853856cde18bca0ed02db37cd37ac333639090"
USER_AGENT = "Mini-ORIGIN-v0.82-pmlb-blind/1"
RETRY_ATTEMPTS = 6
RETRY_BASE_SECONDS = 2.0
TARGET_COLUMN = "target"

_DATASETS_BY_NAME: dict[str, dict[str, object]] = {}
_PARSER_SUMMARIES: dict[str, dict[str, object]] = {}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def load_frozen_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_record_access":
        raise RuntimeError("v0.82 preregistration status changed")
    if preregistration["parent_v81_commit"] != FROZEN_V81_COMMIT:
        raise RuntimeError("frozen v0.81 commit changed")
    if preregistration["parent_v81_lock_digest"] != V81_LOCK_DIGEST:
        raise RuntimeError("v0.81 lock digest changed")
    if preregistration["parent_v81_manifest_sha256"] != V81_MANIFEST_SHA256:
        raise RuntimeError("v0.81 manifest hash changed")
    if preregistration["parent_v80_registry_digest"] != V80_REGISTRY_DIGEST:
        raise RuntimeError("v0.80 registry digest changed")
    if preregistration["frozen_v79_commit"] != FROZEN_V79_COMMIT:
        raise RuntimeError("frozen v0.79 commit changed")
    if preregistration["parent_v79_evidence_digest"] != V79_EVIDENCE_DIGEST:
        raise RuntimeError("v0.79 evidence digest changed")
    for key in (
        "record_or_label_access_before_preregistration",
        "solver_execution_before_preregistration",
    ):
        if preregistration[key] is not False:
            raise RuntimeError(f"v0.82 boundary violated: {key}")
    for key in (
        "adapter_revisions_after_record_access",
        "algorithm_revisions_after_record_access",
        "compiler_revisions_after_record_access",
        "selector_revisions_after_record_access",
        "scientific_threshold_revisions_after_record_access",
    ):
        if int(preregistration[key]) != 0:
            raise RuntimeError(f"v0.82 revision budget changed: {key}")

    manifest_bytes = MANIFEST.read_bytes()
    if sha256_bytes(manifest_bytes) != V81_MANIFEST_SHA256:
        raise RuntimeError("v0.81 manifest bytes changed")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest["status"] != "pmlb_hash_lock_v81_complete":
        raise RuntimeError("v0.81 manifest is not complete")
    if manifest["lock_digest"] != V81_LOCK_DIGEST:
        raise RuntimeError("unexpected v0.81 lock digest")
    if int(manifest["dataset_count"]) != 7:
        raise RuntimeError("v0.81 dataset count changed")
    if manifest["selected_name_overlap"]:
        raise RuntimeError("v0.81 selected names overlap the registry")
    if manifest["byte_rejections"]:
        raise RuntimeError("v0.81 contains byte rejections")
    if manifest["records_or_labels_accessed"] is not False:
        raise RuntimeError("records were accessed during v0.81")
    if manifest["solver_executed"] is not False:
        raise RuntimeError("solver ran during v0.81")
    selected_names = [str(row["name"]) for row in manifest["datasets"]]
    if selected_names != list(preregistration["selected_datasets"]):
        raise RuntimeError("v0.81 selected dataset order changed")

    parent = json.loads(V79_EVIDENCE.read_text(encoding="utf-8"))
    if parent["status"] != "small_query_coverage_development_pass_v79":
        raise RuntimeError("v0.79 parent must remain a pass")
    if parent["evidence_digest"] != V79_EVIDENCE_DIGEST:
        raise RuntimeError("unexpected v0.79 evidence digest")
    if parent["development_gate"] is not True:
        raise RuntimeError("v0.79 development gate changed")
    if int(parent["rust_mismatch_count"]) != 0:
        raise RuntimeError("v0.79 Rust mismatches changed")
    if int(parent["label_independence_certificate"]["mismatch_count"]) != 0:
        raise RuntimeError("v0.79 label-independence certificate changed")
    return preregistration, manifest, parent


def token(value: str) -> str:
    return value.strip()


def parse_pmlb_table(
    dataset: dict[str, object],
    payload: bytes,
) -> tuple[list[tuple[tuple[str, ...], str]], dict[str, object]]:
    name = str(dataset["name"])
    if sha256_bytes(payload) != str(dataset["raw_sha256"]):
        raise RuntimeError(f"SHA-256 mismatch for PMLB {name}")
    if len(payload) != int(dataset["raw_bytes"]):
        raise RuntimeError(f"byte length mismatch for PMLB {name}")
    if not payload.startswith(b"\x1f\x8b"):
        raise RuntimeError(f"gzip magic mismatch for PMLB {name}")
    try:
        decompressed = gzip.decompress(payload)
    except Exception as error:
        raise RuntimeError(f"gzip decompression failed for PMLB {name}") from error
    try:
        text = decompressed.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"UTF-8 decoding failed for PMLB {name}") from error

    reader = csv.reader(io.StringIO(text), delimiter="\t", strict=True)
    try:
        header = next(reader)
    except StopIteration as error:
        raise RuntimeError(f"missing TSV header for PMLB {name}") from error
    header = [token(value) for value in header]
    if not header or any(not value for value in header):
        raise RuntimeError(f"empty TSV header value for PMLB {name}")
    if len(header) != len(set(header)):
        raise RuntimeError(f"duplicate TSV header for PMLB {name}")
    if header.count(TARGET_COLUMN) != 1:
        raise RuntimeError(f"exact target column missing for PMLB {name}")
    target_index = header.index(TARGET_COLUMN)
    feature_indexes = [index for index in range(len(header)) if index != target_index]
    feature_names = [header[index] for index in feature_indexes]
    if len(feature_names) != int(dataset["features"]):
        raise RuntimeError(f"feature count mismatch for PMLB {name}")

    records: list[tuple[tuple[str, ...], str]] = []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise RuntimeError(
                f"malformed PMLB row {row_number} for {name}: "
                f"expected {len(header)}, got {len(row)}"
            )
        features = tuple(token(row[index]) for index in feature_indexes)
        label = token(row[target_index])
        if not label:
            raise RuntimeError(f"empty target in PMLB {name} row {row_number}")
        records.append((features, label))
    if len(records) != int(dataset["instances"]):
        raise RuntimeError(
            f"record count mismatch for PMLB {name}: "
            f"expected {dataset['instances']}, got {len(records)}"
        )
    return records, {
        "compressed_bytes": len(payload),
        "decompressed_bytes": len(decompressed),
        "header_columns": header,
        "feature_columns": feature_names,
        "target_column": TARGET_COLUMN,
        "record_count": len(records),
        "missing_feature_cells": sum(
            1 for features, _ in records for value in features if value == ""
        ),
        "raw_sha256_verified": True,
        "raw_bytes_verified": True,
    }


def parse_records(name: str, payload: bytes):
    records, summary = parse_pmlb_table(_DATASETS_BY_NAME[name], payload)
    _PARSER_SUMMARIES[name] = summary
    return records


def compile_task(name: str, records):
    task, summary = core.compile_task(name, records)
    summary.update(_PARSER_SUMMARIES[name])
    return task, summary


def protocol(preregistration: dict[str, object]) -> dict[str, object]:
    result = selector.protocol()
    result["pmlb_cross_source_adapter"] = preregistration["adapter_protocol"]
    result["v81_lock_digest"] = V81_LOCK_DIGEST
    result["v81_manifest_sha256"] = V81_MANIFEST_SHA256
    result["frozen_v79_commit"] = FROZEN_V79_COMMIT
    return result


def prepare_opened(
    preregistration: dict[str, object],
    manifest: dict[str, object],
    adapted_path: Path,
) -> None:
    selector.configure_module()
    _DATASETS_BY_NAME.clear()
    _PARSER_SUMMARIES.clear()
    adapted = {
        "lock_digest": V81_LOCK_DIGEST,
        "repository_registry_digest": V80_REGISTRY_DIGEST,
        "parent_v66_evidence_digest": core.V66_DIGEST,
        "datasets": [],
    }
    for ordinal, dataset in enumerate(manifest["datasets"], start=1):
        name = str(dataset["name"])
        _DATASETS_BY_NAME[name] = dataset
        adapted["datasets"].append({
            "name": name,
            "uci_id": ordinal,
            "pmlb_ordinal": ordinal,
            "url": dataset["raw_url"],
            "sha256": dataset["raw_sha256"],
            "bytes": dataset["raw_bytes"],
        })
    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_path.write_text(json.dumps(adapted, indent=2), encoding="utf-8")

    opened.MANIFEST = adapted_path
    opened.LOCK_DIGEST = V81_LOCK_DIGEST
    opened.REGISTRY_DIGEST = V80_REGISTRY_DIGEST
    opened.V66_DIGEST = core.V66_DIGEST
    opened.download = download
    opened.parse_records = parse_records
    opened.external.task_from_records = compile_task
    opened.compact_state = selector.compact_state
    opened.protocol = lambda: protocol(preregistration)


def run_reference(states_path: Path, reference_path: Path):
    preregistration, manifest, parent_evidence = load_frozen_inputs()
    prepare_opened(
        preregistration,
        manifest,
        reference_path.parent / "adapted-manifest.json",
    )
    result = opened.run(states_path, reference_path)
    result["status"] = "pmlb_blind_python_reference_v82"
    result["v81_lock_digest"] = V81_LOCK_DIGEST
    result["v81_manifest_sha256"] = V81_MANIFEST_SHA256
    result["parent_v79_evidence_digest"] = V79_EVIDENCE_DIGEST
    result["parent_v68_evidence_digest"] = LEGACY_PARENT_V68_EVIDENCE_DIGEST
    result["compiler_protocol"] = selector.frontier.compiler_protocol()
    result["selector_protocol"] = selector.protocol()["state_selector"]
    result["selected_pmlb_datasets"] = [
        str(row["name"]) for row in manifest["datasets"]
    ]
    result["frozen_v79_label_independence"] = {
        "mismatch_count": int(
            parent_evidence["label_independence_certificate"]["mismatch_count"]
        ),
        "evidence_digest": V79_EVIDENCE_DIGEST,
    }
    result["frozen_external_digest"] = canonical_digest({
        "v81_lock_digest": V81_LOCK_DIGEST,
        "parent_v79_evidence_digest": V79_EVIDENCE_DIGEST,
        "protocol": result["protocol"],
        "dataset_summaries": result["dataset_summaries"],
        "state_input_sha256": result["state_input_sha256"],
        "state_digests": [row["state_digest"] for row in result["rows"]],
    })
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(reference_path: Path, rust_path: Path, output_path: Path):
    preregistration, manifest, parent_evidence = load_frozen_inputs()
    core.PREREGISTRATION = PREREGISTRATION
    result = core.validate(reference_path, rust_path, output_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    parent_label_mismatches = int(
        parent_evidence["label_independence_certificate"]["mismatch_count"]
    )
    adapter_verified = (
        len(reference["dataset_summaries"]) == len(manifest["datasets"])
        and all(
            row.get("raw_sha256_verified") is True
            and row.get("raw_bytes_verified") is True
            and int(row.get("record_count", -1)) > 0
            for row in reference["dataset_summaries"]
        )
    )
    base_gate = bool(result["development_gate"])
    gate = bool(
        base_gate
        and adapter_verified
        and parent_label_mismatches
        == int(preregistration["locked_gate"]["label_independence_mismatches"])
    )
    result["status"] = (
        "pmlb_cross_source_blind_pass_v82"
        if gate else "pmlb_cross_source_blind_rejected_v82"
    )
    result["external_gate"] = gate
    result["base_validator_gate"] = base_gate
    result.pop("development_gate", None)
    result["claim_scope"] = preregistration["claim_boundary"]
    result["fresh_dataset_minimum_states_passed"] = result.pop(
        "previously_zero_datasets_passed"
    )
    result["adapter_verification_passed"] = adapter_verified
    result["v81_lock_digest"] = V81_LOCK_DIGEST
    result["v81_manifest_sha256"] = V81_MANIFEST_SHA256
    result["frozen_v79_commit"] = FROZEN_V79_COMMIT
    result["parent_v79_evidence_digest"] = V79_EVIDENCE_DIGEST
    result["frozen_v79_label_independence_mismatches"] = parent_label_mismatches
    result["selected_pmlb_datasets"] = [
        str(row["name"]) for row in manifest["datasets"]
    ]
    result["frozen_external_digest"] = reference["frozen_external_digest"]
    result.pop("parent_v68_evidence_digest", None)
    result["evidence_digest"] = canonical_digest(result)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    reference_parser = commands.add_parser("reference")
    reference_parser.add_argument("--states", type=Path, required=True)
    reference_parser.add_argument("--reference", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--reference", type=Path, required=True)
    validate_parser.add_argument("--rust", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reference":
        result = run_reference(args.states, args.reference)
        print(json.dumps({
            "status": result["status"],
            "datasets": result["contributing_dataset_count"],
            "base_states": result["base_state_count"],
            "profiled_states": result["profiled_state_count"],
            "bounded_solved": result["bounded_solved_count"],
            "plain_solved": result["both_plain_bounded_count"],
            "bounded_only": result["bounded_only_count"],
        }, indent=2))
        return
    result = validate(args.reference, args.rust, args.output)
    print(json.dumps({
        "status": result["status"],
        "gate": result["external_gate"],
        "datasets": result["contributing_dataset_count"],
        "base_states": result["base_state_count"],
        "profiled_states": result["profiled_state_count"],
        "bounded_solved": result["bounded_solved_count"],
        "plain_solved": result["both_plain_bounded_count"],
        "bounded_only": result["bounded_only_count"],
        "median": result["expansion_ratio_median"],
        "rust_mismatches": result["rust_mismatch_count"],
        "adapter_verified": result["adapter_verification_passed"],
    }, indent=2))
    if not result["external_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
