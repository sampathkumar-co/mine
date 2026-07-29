from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from . import clean_lower_bound_conditioned_v68 as opened
from . import label_free_frontier_v72 as frontier
from . import numeric_threshold_frontier_v70 as core


PREREGISTRATION = Path(__file__).resolve().parents[2] / "campaigns" / "v78-openml-cross-source-evaluation.json"
LOCK_EVIDENCE = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v77-openml-hash-lock.json"
LOCK_DIGEST = "03a64c8c5928070fb41b15d4892c2f720a909fc39c3a7f5b9597cd79f1879590"
REGISTRY_DIGEST = "d312c4f0b853237479d6be8a74b6bf47776722d7aea1ce00c7b9745be90d57d2"
V75_EVIDENCE_DIGEST = "db379850b2a517e16d5ea442047ac4933ad06fdcf4d6838d91fc36d72e75bc47"
FROZEN_V75_COMMIT = "d8aa4153b69b82ccb714cfbb50d12c5137186047"
USER_AGENT = "Mini-ORIGIN-v0.78-openml-cross-source/1"

_DATASETS: dict[str, dict[str, object]] = {}
_PARSER_SUMMARIES: dict[str, dict[str, object]] = {}


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as handle:
        return handle.read()


def _attribute_name(text: str) -> str:
    rest = text.strip()[10:].lstrip()
    if not rest:
        raise RuntimeError("empty @attribute declaration")
    if rest[0] in "'\"":
        quote = rest[0]
        out = []
        escaped = False
        for index, char in enumerate(rest[1:], start=1):
            if escaped:
                out.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                if not rest[index + 1 :].strip():
                    raise RuntimeError("attribute type missing")
                return "".join(out)
            else:
                out.append(char)
        raise RuntimeError("unterminated quoted attribute name")
    pieces = rest.split(None, 1)
    if len(pieces) != 2:
        raise RuntimeError("attribute type missing")
    return pieces[0]


def _split_dense_row(line: str) -> list[str]:
    values: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for char in line:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\" and quote is not None:
            escaped = True
        elif quote is not None:
            if char == quote:
                quote = None
            else:
                current.append(char)
        elif char in "'\"":
            quote = char
        elif char == ",":
            values.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if quote is not None:
        raise RuntimeError("unterminated quoted ARFF field")
    values.append("".join(current).strip())
    return values


def parse_dense_arff(dataset: dict[str, object], payload: bytes):
    text = payload.decode("utf-8-sig", errors="strict")
    attributes: list[str] = []
    data_started = False
    rows: list[list[str]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        lowered = line.lower()
        if not data_started:
            if lowered.startswith("@attribute"):
                attributes.append(_attribute_name(line))
            elif lowered == "@data":
                data_started = True
            continue
        if line.startswith("{"):
            raise RuntimeError(f"sparse ARFF row rejected at line {line_number}")
        values = _split_dense_row(line)
        if len(values) != len(attributes):
            raise RuntimeError(
                f"ARFF width mismatch for {dataset['name']} line {line_number}: "
                f"{len(values)} != {len(attributes)}"
            )
        rows.append(values)
    if not data_started or not attributes:
        raise RuntimeError(f"missing ARFF header/data section for {dataset['name']}")
    if len(attributes) != len(set(attributes)):
        raise RuntimeError(f"duplicate ARFF attributes for {dataset['name']}")
    target = str(dataset["default_target_attribute"])
    if attributes.count(target) != 1:
        raise RuntimeError(f"target attribute mismatch for {dataset['name']}")
    target_index = attributes.index(target)
    if len(attributes) != int(dataset["num_features"]):
        raise RuntimeError(f"declared feature count mismatch for {dataset['name']}")
    if len(rows) != int(dataset["num_instances"]):
        raise RuntimeError(f"record count mismatch for {dataset['name']}")
    records = []
    feature_names = [name for index, name in enumerate(attributes) if index != target_index]
    for row_number, values in enumerate(rows, start=1):
        label = values[target_index].strip()
        if not label or label == "?":
            raise RuntimeError(f"empty/missing target for {dataset['name']} row {row_number}")
        features = tuple(value.strip() for index, value in enumerate(values) if index != target_index)
        records.append((features, label))
    return records, {
        "format": "ARFF",
        "declared_attribute_count": len(attributes),
        "feature_columns": feature_names,
        "target_column": target,
        "record_count": len(records),
        "missing_feature_tokens": sum(value == "?" for features, _ in records for value in features),
    }


def parse_records(name: str, payload: bytes):
    records, summary = parse_dense_arff(_DATASETS[name], payload)
    _PARSER_SUMMARIES[name] = summary
    return records


def compile_task(name: str, records):
    task, summary = core.compile_task(name, records)
    summary.update(_PARSER_SUMMARIES[name])
    return task, summary


def load_frozen_inputs():
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    evidence = json.loads(LOCK_EVIDENCE.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_record_access":
        raise RuntimeError("v0.78 preregistration status changed")
    if prereg["parent_v77_commit"] != "8168664e4068aa3a8b8736dc3ff13b35ecf67981":
        raise RuntimeError("v0.77 parent commit changed")
    if prereg["parent_v77_lock_digest"] != LOCK_DIGEST:
        raise RuntimeError("v0.77 lock commitment changed")
    if prereg["parent_v75_evidence_digest"] != V75_EVIDENCE_DIGEST:
        raise RuntimeError("v0.75 evidence commitment changed")
    if prereg["record_or_label_access_before_preregistration"] is not False:
        raise RuntimeError("records were accessed before preregistration")
    if prereg["solver_execution_before_preregistration"] is not False:
        raise RuntimeError("solver ran before preregistration")
    if int(prereg["algorithm_revisions_after_record_access"]) != 0:
        raise RuntimeError("algorithm revision budget changed")
    if int(prereg["scientific_threshold_revisions_after_record_access"]) != 0:
        raise RuntimeError("threshold revision budget changed")
    if evidence["status"] != "openml_cross_source_hash_lock_v77_complete":
        raise RuntimeError("v0.77 hash lock incomplete")
    if evidence["lock_digest"] != LOCK_DIGEST:
        raise RuntimeError("v0.77 lock digest changed")
    ids = [int(row["dataset_id"]) for row in evidence["datasets"]]
    if ids != [int(value) for value in prereg["selected_openml_dataset_ids"]]:
        raise RuntimeError("selected OpenML suite changed")
    if len(ids) != 7 or evidence["selected_id_overlap"] or evidence["selected_name_overlap"]:
        raise RuntimeError("invalid v0.77 selected suite")
    return prereg, evidence


def protocol(prereg):
    result = frontier.protocol()
    result["openml_cross_source_adapter"] = prereg["adapter_protocol"]
    result["v77_lock_digest"] = LOCK_DIGEST
    result["source"] = "OpenML-CC18 raw ARFF"
    return result


def prepare_opened(prereg, evidence, adapted_path: Path):
    frontier.configure_module()
    _DATASETS.clear()
    _PARSER_SUMMARIES.clear()
    adapted = {
        "lock_digest": LOCK_DIGEST,
        "repository_registry_digest": REGISTRY_DIGEST,
        "parent_v66_evidence_digest": core.V66_DIGEST,
        "datasets": [],
    }
    for dataset in evidence["datasets"]:
        name = str(dataset["name"])
        _DATASETS[name] = dataset
        adapted["datasets"].append({
            "name": name,
            "uci_id": int(dataset["dataset_id"]),
            "url": str(dataset["url"]),
            "sha256": str(dataset["raw_sha256"]),
            "bytes": int(dataset["raw_bytes"]),
        })
    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    adapted_path.write_text(json.dumps(adapted, indent=2), encoding="utf-8")
    opened.MANIFEST = adapted_path
    opened.LOCK_DIGEST = LOCK_DIGEST
    opened.REGISTRY_DIGEST = REGISTRY_DIGEST
    opened.V66_DIGEST = core.V66_DIGEST
    opened.download = download
    opened.parse_records = parse_records
    opened.external.task_from_records = compile_task
    opened.compact_state = frontier.compact_state
    opened.protocol = lambda: protocol(prereg)


def run_reference(states_path: Path, reference_path: Path):
    prereg, evidence = load_frozen_inputs()
    prepare_opened(prereg, evidence, reference_path.parent / "adapted-manifest.json")
    result = opened.run(states_path, reference_path)
    result["status"] = "openml_cross_source_python_reference_v78"
    result["v77_lock_digest"] = LOCK_DIGEST
    result["parent_v75_evidence_digest"] = V75_EVIDENCE_DIGEST
    result["parent_v68_evidence_digest"] = V75_EVIDENCE_DIGEST
    result["compiler_protocol"] = frontier.compiler_protocol()
    result["frozen_external_digest"] = hashlib.sha256(json.dumps({
        "v77_lock_digest": LOCK_DIGEST,
        "parent_v75_evidence_digest": V75_EVIDENCE_DIGEST,
        "protocol": result["protocol"],
        "dataset_summaries": result["dataset_summaries"],
        "state_input_sha256": result["state_input_sha256"],
        "state_digests": [row["state_digest"] for row in result["rows"]],
    }, sort_keys=True).encode("utf-8")).hexdigest()
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(reference_path: Path, rust_path: Path, output_path: Path):
    prereg, evidence = load_frozen_inputs()
    core.PREREGISTRATION = PREREGISTRATION
    result = core.validate(reference_path, rust_path, output_path)
    gate = bool(result["development_gate"])
    result["status"] = "openml_cross_source_external_pass_v78" if gate else "openml_cross_source_external_rejected_v78"
    result["external_gate"] = gate
    result.pop("development_gate", None)
    result["claim_scope"] = prereg["claim_boundary"]
    result["fresh_dataset_minimum_states_passed"] = result.pop("previously_zero_datasets_passed")
    result["v77_lock_digest"] = LOCK_DIGEST
    result["frozen_v75_commit"] = FROZEN_V75_COMMIT
    result["parent_v75_evidence_digest"] = V75_EVIDENCE_DIGEST
    result["selected_openml_dataset_ids"] = [row["dataset_id"] for row in evidence["datasets"]]
    result.pop("parent_v68_evidence_digest", None)
    result["evidence_digest"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
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
