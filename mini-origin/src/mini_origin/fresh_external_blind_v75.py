from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from urllib.request import Request, urlopen

from . import clean_lower_bound_conditioned_v68 as opened
from . import label_free_frontier_v72 as frontier
from . import numeric_threshold_frontier_v70 as core


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v75-fresh-external-blind.json"
)
MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "external-data"
    / "uci-v74"
    / "manifest.json"
)
V74_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v74-fresh-external-hash-lock.json"
)
V74_LOCK_DIGEST = "6730d5029294a753236f77cef1be1334885a5e8cc8d114a285637987eae5fbaf"
V74_MANIFEST_SHA256 = "d0d6b849f4cb07a6ee2ae42b6dfc92319579368c91480dd074522dda7d475019"
V73_REGISTRY_DIGEST = "3fde8f0138548928651937201cef66aa71ee41bbb792ade82b1a02337ae8b392"
V72_EVIDENCE_DIGEST = "b1fc70852a2ad35d91972889eb853856cde18bca0ed02db37cd37ac333639090"
FROZEN_V72_COMMIT = "dae02829efc4819935a4ec87c31ea5eee3305d83"
METADATA_URL = "https://archive.ics.uci.edu/api/dataset?id={uci_id}"
USER_AGENT = "Mini-ORIGIN-v0.75-fresh-external-blind/1"

_DATASETS_BY_NAME: dict[str, dict[str, object]] = {}
_METADATA_BY_NAME: dict[str, dict[str, object]] = {}
_PARSER_SUMMARIES: dict[str, dict[str, object]] = {}


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as handle:
        return handle.read()


def load_frozen_inputs() -> tuple[dict[str, object], dict[str, object]]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_record_access":
        raise RuntimeError("v0.75 preregistration status changed")
    if preregistration["parent_v74_lock_digest"] != V74_LOCK_DIGEST:
        raise RuntimeError("v0.74 lock commitment changed")
    if preregistration["parent_v74_manifest_sha256"] != V74_MANIFEST_SHA256:
        raise RuntimeError("v0.74 manifest commitment changed")
    if preregistration["frozen_v72_commit"] != FROZEN_V72_COMMIT:
        raise RuntimeError("frozen v0.72 commit changed")
    if preregistration["parent_v72_evidence_digest"] != V72_EVIDENCE_DIGEST:
        raise RuntimeError("v0.72 evidence commitment changed")
    if preregistration["record_or_label_access_before_preregistration"] is not False:
        raise RuntimeError("records were accessed before preregistration")
    if preregistration["solver_execution_before_preregistration"] is not False:
        raise RuntimeError("solver ran before preregistration")
    if int(preregistration["algorithm_revisions_after_record_access"]) != 0:
        raise RuntimeError("algorithm revision budget must remain zero")
    if int(preregistration["scientific_threshold_revisions_after_record_access"]) != 0:
        raise RuntimeError("threshold revision budget must remain zero")

    manifest_bytes = MANIFEST.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != V74_MANIFEST_SHA256:
        raise RuntimeError("v0.74 manifest bytes changed")
    manifest = json.loads(manifest_bytes)
    if manifest["status"] != "fresh_external_hash_lock_v74_complete":
        raise RuntimeError("v0.74 hash lock must remain complete")
    if manifest["lock_digest"] != V74_LOCK_DIGEST:
        raise RuntimeError("unexpected v0.74 lock digest")
    if manifest["selected_overlap"] or int(manifest["dataset_count"]) != 7:
        raise RuntimeError("v0.74 selected suite changed")
    evidence = json.loads(V74_EVIDENCE.read_text(encoding="utf-8"))
    if evidence["status"] != "fresh_external_hash_lock_v74_complete":
        raise RuntimeError("v0.74 evidence must remain complete")
    if evidence["lock_digest"] != V74_LOCK_DIGEST:
        raise RuntimeError("v0.74 evidence lock changed")
    return preregistration, manifest


def parse_standardized_csv(
    dataset: dict[str, object],
    payload: bytes,
    metadata_payload: dict[str, object],
) -> tuple[list[tuple[tuple[str, ...], str]], dict[str, object]]:
    if int(metadata_payload.get("status", 0)) != 200:
        raise RuntimeError(f"metadata unavailable for UCI {dataset['uci_id']}")
    metadata = metadata_payload["data"]
    variables = metadata.get("variables") or []
    feature_names = [
        str(row["name"])
        for row in variables
        if isinstance(row, dict) and str(row.get("role", "")).lower() == "feature"
    ]
    target_names = [
        str(row["name"])
        for row in variables
        if isinstance(row, dict) and str(row.get("role", "")).lower() == "target"
    ]
    expected_targets = [str(value) for value in dataset["target_columns"]]
    if int(metadata["uci_id"]) != int(dataset["uci_id"]):
        raise RuntimeError(f"metadata ID mismatch for UCI {dataset['uci_id']}")
    if target_names != expected_targets or len(target_names) != 1:
        raise RuntimeError(f"target role mismatch for UCI {dataset['uci_id']}")
    if len(feature_names) != int(dataset["num_features"]):
        raise RuntimeError(f"feature role mismatch for UCI {dataset['uci_id']}")

    reader = csv.DictReader(StringIO(payload.decode("utf-8-sig", errors="strict")))
    header = list(reader.fieldnames or [])
    if not header or len(header) != len(set(header)):
        raise RuntimeError(f"missing or duplicate header for UCI {dataset['uci_id']}")
    required = feature_names + target_names
    missing = [name for name in required if name not in header]
    if missing:
        raise RuntimeError(f"missing CSV columns for UCI {dataset['uci_id']}: {missing}")

    records = []
    target = target_names[0]
    for row_number, row in enumerate(reader, start=2):
        features = tuple(
            "" if row.get(name) is None else str(row[name]).strip()
            for name in feature_names
        )
        label_value = row.get(target)
        label = "" if label_value is None else str(label_value).strip()
        if not label:
            raise RuntimeError(
                f"empty target in UCI {dataset['uci_id']} row {row_number}"
            )
        records.append((features, label))
    if len(records) != int(dataset["num_instances"]):
        raise RuntimeError(f"record count mismatch for UCI {dataset['uci_id']}")
    return records, {
        "csv_header_count": len(header),
        "feature_columns": feature_names,
        "target_column": target,
        "ignored_columns": [name for name in header if name not in required],
        "record_count": len(records),
    }


def parse_records(name: str, payload: bytes):
    records, summary = parse_standardized_csv(
        _DATASETS_BY_NAME[name],
        payload,
        _METADATA_BY_NAME[name],
    )
    _PARSER_SUMMARIES[name] = summary
    return records


def compile_task(name: str, records):
    task, summary = core.compile_task(name, records)
    summary.update(_PARSER_SUMMARIES[name])
    return task, summary


def protocol(preregistration: dict[str, object]) -> dict[str, object]:
    result = frontier.protocol()
    result["fresh_external_adapter"] = preregistration["adapter_protocol"]
    result["v74_lock_digest"] = V74_LOCK_DIGEST
    return result


def prepare_opened(
    preregistration: dict[str, object],
    manifest: dict[str, object],
    adapted_path: Path,
) -> list[dict[str, object]]:
    frontier.configure_module()
    _DATASETS_BY_NAME.clear()
    _METADATA_BY_NAME.clear()
    _PARSER_SUMMARIES.clear()
    adapted = {
        "lock_digest": V74_LOCK_DIGEST,
        "repository_registry_digest": V73_REGISTRY_DIGEST,
        "parent_v66_evidence_digest": core.V66_DIGEST,
        "datasets": [],
    }
    metadata_verification = []
    for dataset in manifest["datasets"]:
        name = str(dataset["name"])
        metadata_payload = json.loads(
            download(METADATA_URL.format(uci_id=int(dataset["uci_id"])))
        )
        actual = canonical_digest(metadata_payload)
        matched = actual == dataset["metadata_digest"]
        metadata_verification.append({
            "name": name,
            "uci_id": dataset["uci_id"],
            "expected_metadata_digest": dataset["metadata_digest"],
            "actual_metadata_digest": actual,
            "matched": matched,
        })
        if not matched:
            raise RuntimeError(f"metadata mismatch for {name}")
        _DATASETS_BY_NAME[name] = dataset
        _METADATA_BY_NAME[name] = metadata_payload
        adapted["datasets"].append({
            "name": name,
            "uci_id": dataset["uci_id"],
            "url": dataset["data_url"],
            "sha256": dataset["csv_sha256"],
            "bytes": dataset["csv_bytes"],
        })
    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_path.write_text(json.dumps(adapted, indent=2), encoding="utf-8")

    opened.MANIFEST = adapted_path
    opened.LOCK_DIGEST = V74_LOCK_DIGEST
    opened.REGISTRY_DIGEST = V73_REGISTRY_DIGEST
    opened.V66_DIGEST = core.V66_DIGEST
    opened.download = download
    opened.parse_records = parse_records
    opened.external.task_from_records = compile_task
    opened.compact_state = frontier.compact_state
    opened.protocol = lambda: protocol(preregistration)
    return metadata_verification


def run_reference(states_path: Path, reference_path: Path):
    preregistration, manifest = load_frozen_inputs()
    metadata_verification = prepare_opened(
        preregistration,
        manifest,
        reference_path.parent / "adapted-manifest.json",
    )
    result = opened.run(states_path, reference_path)
    result["status"] = "fresh_external_python_reference_v75"
    result["v74_lock_digest"] = V74_LOCK_DIGEST
    result["parent_v72_evidence_digest"] = V72_EVIDENCE_DIGEST
    result["parent_v68_evidence_digest"] = V72_EVIDENCE_DIGEST
    result["compiler_protocol"] = frontier.compiler_protocol()
    result["metadata_verification"] = {
        "all_hashes_match": all(row["matched"] for row in metadata_verification),
        "rows": metadata_verification,
    }
    result["frozen_external_digest"] = hashlib.sha256(
        json.dumps({
            "v74_lock_digest": V74_LOCK_DIGEST,
            "parent_v72_evidence_digest": V72_EVIDENCE_DIGEST,
            "protocol": result["protocol"],
            "dataset_summaries": result["dataset_summaries"],
            "state_input_sha256": result["state_input_sha256"],
            "state_digests": [row["state_digest"] for row in result["rows"]],
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(reference_path: Path, rust_path: Path, output_path: Path):
    preregistration, manifest = load_frozen_inputs()
    core.PREREGISTRATION = PREREGISTRATION
    result = core.validate(reference_path, rust_path, output_path)
    gate = bool(result["development_gate"])
    result["status"] = (
        "fresh_external_blind_pass_v75"
        if gate else "fresh_external_blind_rejected_v75"
    )
    result["external_gate"] = gate
    result.pop("development_gate", None)
    result["claim_scope"] = preregistration["claim_boundary"]
    result["fresh_dataset_minimum_states_passed"] = result.pop(
        "previously_zero_datasets_passed"
    )
    result["v74_lock_digest"] = V74_LOCK_DIGEST
    result["frozen_v72_commit"] = FROZEN_V72_COMMIT
    result["parent_v72_evidence_digest"] = V72_EVIDENCE_DIGEST
    result["selected_uci_ids"] = [row["uci_id"] for row in manifest["datasets"]]
    result.pop("parent_v68_evidence_digest", None)
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
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
    }, indent=2))
    if not result["external_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
