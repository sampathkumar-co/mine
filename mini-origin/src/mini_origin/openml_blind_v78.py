from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import arff

from . import clean_lower_bound_conditioned_v68 as opened
from . import label_free_frontier_v72 as frontier
from . import numeric_threshold_frontier_v70 as core


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v78-openml-blind.json"
)
MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v77-openml-hash-lock.json"
)
V77_LOCK_DIGEST = "03a64c8c5928070fb41b15d4892c2f720a909fc39c3a7f5b9597cd79f1879590"
V77_MANIFEST_SHA256 = "e9e500a6441720feff3455cfea183248b03b4fb991a5e4e8448840463c845284"
V76_REGISTRY_DIGEST = "d312c4f0b853237479d6be8a74b6bf47776722d7aea1ce00c7b9745be90d57d2"
V75_EVIDENCE_DIGEST = "db379850b2a517e16d5ea442047ac4933ad06fdcf4d6838d91fc36d72e75bc47"
LEGACY_PARENT_V68_EVIDENCE_DIGEST = "b1fc70852a2ad35d91972889eb853856cde18bca0ed02db37cd37ac333639090"
FROZEN_V75_COMMIT = "d8aa4153b69b82ccb714cfbb50d12c5137186047"
FROZEN_V77_COMMIT = "8168664e4068aa3a8b8736dc3ff13b35ecf67981"
USER_AGENT = "Mini-ORIGIN-v0.78-openml-blind/1"
RETRY_ATTEMPTS = 6
RETRY_BASE_SECONDS = 2.0

_DATASETS_BY_NAME: dict[str, dict[str, object]] = {}
_PARSER_SUMMARIES: dict[str, dict[str, object]] = {}


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


def load_frozen_inputs() -> tuple[dict[str, object], dict[str, object]]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_record_access":
        raise RuntimeError("v0.78 preregistration status changed")
    if preregistration["parent_v77_commit"] != FROZEN_V77_COMMIT:
        raise RuntimeError("frozen v0.77 commit changed")
    if preregistration["parent_v77_lock_digest"] != V77_LOCK_DIGEST:
        raise RuntimeError("v0.77 lock commitment changed")
    if preregistration["parent_v77_manifest_sha256"] != V77_MANIFEST_SHA256:
        raise RuntimeError("v0.77 manifest commitment changed")
    if preregistration["frozen_v75_commit"] != FROZEN_V75_COMMIT:
        raise RuntimeError("frozen v0.75 commit changed")
    if preregistration["parent_v75_evidence_digest"] != V75_EVIDENCE_DIGEST:
        raise RuntimeError("v0.75 evidence commitment changed")
    if preregistration["record_or_label_access_before_preregistration"] is not False:
        raise RuntimeError("records were accessed before v0.78 preregistration")
    if preregistration["solver_execution_before_preregistration"] is not False:
        raise RuntimeError("solver ran before v0.78 preregistration")
    if int(preregistration["algorithm_revisions_after_record_access"]) != 0:
        raise RuntimeError("algorithm revision budget must remain zero")
    if int(preregistration["scientific_threshold_revisions_after_record_access"]) != 0:
        raise RuntimeError("threshold revision budget must remain zero")

    manifest_bytes = MANIFEST.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != V77_MANIFEST_SHA256:
        raise RuntimeError("v0.77 manifest bytes changed")
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    if manifest["status"] != "openml_cross_source_hash_lock_v77_complete":
        raise RuntimeError("v0.77 hash lock must remain complete")
    if manifest["lock_digest"] != V77_LOCK_DIGEST:
        raise RuntimeError("unexpected v0.77 lock digest")
    if manifest["parent_v76_registry_digest"] != V76_REGISTRY_DIGEST:
        raise RuntimeError("unexpected v0.76 registry digest")
    if manifest["parent_v75_evidence_digest"] != V75_EVIDENCE_DIGEST:
        raise RuntimeError("unexpected v0.75 evidence digest")
    if int(manifest["dataset_count"]) != 7:
        raise RuntimeError("v0.77 dataset count changed")
    if manifest["byte_rejections"]:
        raise RuntimeError("v0.77 contains byte rejections")
    if manifest["selected_id_overlap"] or manifest["selected_name_overlap"]:
        raise RuntimeError("v0.77 overlaps the frozen registry")
    return preregistration, manifest


def _split_ignored(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _token(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_openml_arff(
    dataset: dict[str, object],
    payload: bytes,
) -> tuple[list[tuple[tuple[str, ...], str]], dict[str, object]]:
    if hashlib.sha256(payload).hexdigest() != str(dataset["raw_sha256"]):
        raise RuntimeError(f"SHA-256 mismatch for OpenML {dataset['dataset_id']}")
    if hashlib.md5(payload).hexdigest() != str(dataset["raw_md5"]):
        raise RuntimeError(f"MD5 mismatch for OpenML {dataset['dataset_id']}")
    if len(payload) != int(dataset["raw_bytes"]):
        raise RuntimeError(f"byte length mismatch for OpenML {dataset['dataset_id']}")

    document = arff.loads(payload.decode("utf-8-sig", errors="strict"))
    attributes = list(document.get("attributes") or [])
    attribute_names = [str(row[0]) for row in attributes]
    if not attribute_names or len(attribute_names) != len(set(attribute_names)):
        raise RuntimeError(f"missing or duplicate ARFF attributes for {dataset['name']}")
    if len(attribute_names) != int(dataset["num_features"]):
        raise RuntimeError(f"declared attribute count mismatch for {dataset['name']}")

    target = str(dataset["target_name"])
    if target != str(dataset["default_target_attribute"]):
        raise RuntimeError(f"target metadata mismatch for {dataset['name']}")
    if attribute_names.count(target) != 1:
        raise RuntimeError(f"target attribute missing for {dataset['name']}")

    ignored = _split_ignored(dataset.get("row_id_attribute"))
    ignored += _split_ignored(dataset.get("ignore_attribute"))
    ignored = list(dict.fromkeys(ignored))
    missing_ignored = [name for name in ignored if name not in attribute_names]
    if missing_ignored:
        raise RuntimeError(f"ignored attributes missing for {dataset['name']}: {missing_ignored}")

    feature_indexes = [
        index
        for index, name in enumerate(attribute_names)
        if name != target and name not in ignored
    ]
    target_index = attribute_names.index(target)
    expected_features = int(dataset["num_features"]) - 1 - len(ignored)
    if len(feature_indexes) != expected_features:
        raise RuntimeError(f"feature count mismatch for {dataset['name']}")

    rows = list(document.get("data") or [])
    if len(rows) != int(dataset["num_instances"]):
        raise RuntimeError(f"record count mismatch for {dataset['name']}")

    records: list[tuple[tuple[str, ...], str]] = []
    for row_number, row in enumerate(rows, start=1):
        if isinstance(row, dict) or len(row) != len(attribute_names):
            raise RuntimeError(f"sparse or malformed ARFF row {row_number} for {dataset['name']}")
        features = tuple(_token(row[index]) for index in feature_indexes)
        label = _token(row[target_index])
        if not label:
            raise RuntimeError(f"empty target in {dataset['name']} row {row_number}")
        records.append((features, label))

    feature_names = [attribute_names[index] for index in feature_indexes]
    return records, {
        "arff_attribute_count": len(attribute_names),
        "feature_columns": feature_names,
        "target_column": target,
        "ignored_columns": ignored,
        "record_count": len(records),
        "missing_feature_cells": sum(
            1 for features, _ in records for value in features if value == ""
        ),
    }


def parse_records(name: str, payload: bytes):
    records, summary = parse_openml_arff(_DATASETS_BY_NAME[name], payload)
    _PARSER_SUMMARIES[name] = summary
    return records


def compile_task(name: str, records):
    task, summary = core.compile_task(name, records)
    summary.update(_PARSER_SUMMARIES[name])
    return task, summary


def protocol(preregistration: dict[str, object]) -> dict[str, object]:
    result = frontier.protocol()
    result["openml_cross_source_adapter"] = preregistration["adapter_protocol"]
    result["v77_lock_digest"] = V77_LOCK_DIGEST
    result["v77_manifest_sha256"] = V77_MANIFEST_SHA256
    return result


def prepare_opened(
    preregistration: dict[str, object],
    manifest: dict[str, object],
    adapted_path: Path,
) -> None:
    frontier.configure_module()
    _DATASETS_BY_NAME.clear()
    _PARSER_SUMMARIES.clear()
    adapted = {
        "lock_digest": V77_LOCK_DIGEST,
        "repository_registry_digest": V76_REGISTRY_DIGEST,
        "parent_v66_evidence_digest": core.V66_DIGEST,
        "datasets": [],
    }
    for dataset in manifest["datasets"]:
        name = str(dataset["name"])
        _DATASETS_BY_NAME[name] = dataset
        adapted["datasets"].append({
            "name": name,
            "uci_id": dataset["dataset_id"],
            "openml_dataset_id": dataset["dataset_id"],
            "openml_task_id": dataset["task_id"],
            "url": dataset["url"],
            "sha256": dataset["raw_sha256"],
            "bytes": dataset["raw_bytes"],
        })
    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_path.write_text(json.dumps(adapted, indent=2), encoding="utf-8")

    opened.MANIFEST = adapted_path
    opened.LOCK_DIGEST = V77_LOCK_DIGEST
    opened.REGISTRY_DIGEST = V76_REGISTRY_DIGEST
    opened.V66_DIGEST = core.V66_DIGEST
    opened.download = download
    opened.parse_records = parse_records
    opened.external.task_from_records = compile_task
    opened.compact_state = frontier.compact_state
    opened.protocol = lambda: protocol(preregistration)


def run_reference(states_path: Path, reference_path: Path):
    preregistration, manifest = load_frozen_inputs()
    prepare_opened(
        preregistration,
        manifest,
        reference_path.parent / "adapted-manifest.json",
    )
    result = opened.run(states_path, reference_path)
    result["status"] = "openml_blind_python_reference_v78"
    result["v77_lock_digest"] = V77_LOCK_DIGEST
    result["v77_manifest_sha256"] = V77_MANIFEST_SHA256
    result["parent_v75_evidence_digest"] = V75_EVIDENCE_DIGEST
    result["parent_v68_evidence_digest"] = LEGACY_PARENT_V68_EVIDENCE_DIGEST
    result["compiler_protocol"] = frontier.compiler_protocol()
    result["selected_openml_dataset_ids"] = [
        row["dataset_id"] for row in manifest["datasets"]
    ]
    result["selected_openml_task_ids"] = [
        row["task_id"] for row in manifest["datasets"]
    ]
    result["frozen_external_digest"] = hashlib.sha256(
        json.dumps({
            "v77_lock_digest": V77_LOCK_DIGEST,
            "parent_v75_evidence_digest": V75_EVIDENCE_DIGEST,
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
    result["status"] = "openml_cross_source_blind_pass_v78" if gate else "openml_cross_source_blind_rejected_v78"
    result["external_gate"] = gate
    result.pop("development_gate", None)
    result["claim_scope"] = preregistration["claim_boundary"]
    result["fresh_dataset_minimum_states_passed"] = result.pop(
        "previously_zero_datasets_passed"
    )
    result["v77_lock_digest"] = V77_LOCK_DIGEST
    result["v77_manifest_sha256"] = V77_MANIFEST_SHA256
    result["frozen_v75_commit"] = FROZEN_V75_COMMIT
    result["parent_v75_evidence_digest"] = V75_EVIDENCE_DIGEST
    result["selected_openml_dataset_ids"] = [
        row["dataset_id"] for row in manifest["datasets"]
    ]
    result["selected_openml_task_ids"] = [
        row["task_id"] for row in manifest["datasets"]
    ]
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
