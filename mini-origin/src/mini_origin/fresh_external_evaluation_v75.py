from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

from . import label_free_frontier_v72 as v72
from . import numeric_threshold_frontier_v70 as core
from . import clean_lower_bound_conditioned_v68 as opened

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "external-data" / "uci-v75" / "manifest.json"
PREREGISTRATION = ROOT / "campaigns" / "v75-fresh-external-evaluation.json"
V72_EVIDENCE = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v72-label-free-frontier-pass.json"
LOCK_DIGEST = "6730d5029294a753236f77cef1be1334885a5e8cc8d114a285637987eae5fbaf"
REGISTRY_DIGEST = "3fde8f0138548928651937201cef66aa71ee41bbb792ade82b1a02337ae8b392"
V72_DIGEST = "b1fc70852a2ad35d91972889eb853856cde18bca0ed02db37cd37ac333639090"
V66_DIGEST = "3b2bb026556ff9f6321ad6a8375854ae46931e64080329c76f86f31d12c0d643"
DATASET_BY_NAME: dict[str, dict[str, object]] = {}


def parse_csv(name: str, payload: bytes):
    spec = DATASET_BY_NAME[name]
    text = payload.decode("utf-8-sig", errors="strict")
    reader = csv.reader(StringIO(text), strict=True)
    try:
        header = next(reader)
    except StopIteration as error:
        raise RuntimeError(f"empty CSV for {name}") from error
    header = [value.strip() for value in header]
    if not header or len(set(header)) != len(header) or any(not value for value in header):
        raise RuntimeError(f"invalid or duplicate CSV header for {name}: {header!r}")
    target = str(spec["target"])
    if header.count(target) != 1:
        raise RuntimeError(f"expected one target {target!r} for {name}")
    target_index = header.index(target)
    feature_indices = tuple(index for index in range(len(header)) if index != target_index)
    expected_features = int(spec["features"])
    if len(feature_indices) != expected_features:
        raise RuntimeError(
            f"unexpected feature width for {name}: {len(feature_indices)} != {expected_features}"
        )
    rows = []
    for line_number, values in enumerate(reader, start=2):
        if len(values) != len(header):
            raise RuntimeError(
                f"CSV width mismatch for {name} line {line_number}: {len(values)} != {len(header)}"
            )
        cleaned = tuple(value.strip() for value in values)
        label = cleaned[target_index]
        if not label:
            raise RuntimeError(f"empty target for {name} line {line_number}")
        rows.append((tuple(cleaned[index] for index in feature_indices), label))
    expected_rows = int(spec["rows"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"unexpected row count for {name}: {len(rows)} != {expected_rows}")
    return rows


def configure() -> tuple[dict[str, object], dict[str, object]]:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evidence = json.loads(V72_EVIDENCE.read_text(encoding="utf-8"))
    if prereg["parent_v74_lock_digest"] != LOCK_DIGEST:
        raise RuntimeError("v0.75 preregistration lock changed")
    if prereg["parent_v72_evidence_digest"] != V72_DIGEST:
        raise RuntimeError("v0.75 preregistration parent changed")
    if manifest["lock_digest"] != LOCK_DIGEST:
        raise RuntimeError("v0.74 lock digest changed")
    if manifest["repository_registry_digest"] != REGISTRY_DIGEST:
        raise RuntimeError("v0.73 registry digest changed")
    if manifest["parent_v72_evidence_digest"] != V72_DIGEST:
        raise RuntimeError("derived manifest v0.72 parent changed")
    if evidence.get("evidence_digest") != V72_DIGEST or not evidence.get("development_gate"):
        raise RuntimeError("v0.72 passing evidence changed")
    if [int(row["uci_id"]) for row in manifest["datasets"]] != [int(value) for value in prereg["datasets"]]:
        raise RuntimeError("dataset order or identity changed")
    DATASET_BY_NAME.clear()
    DATASET_BY_NAME.update({str(row["name"]): row for row in manifest["datasets"]})

    v72.configure_module()
    opened.MANIFEST = MANIFEST
    opened.LOCK_DIGEST = LOCK_DIGEST
    opened.REGISTRY_DIGEST = REGISTRY_DIGEST
    opened.V66_DIGEST = V66_DIGEST
    opened.parse_records = parse_csv
    core.opened = opened
    return prereg, manifest


def run_reference(states_path: Path, reference_path: Path) -> dict[str, object]:
    prereg, manifest = configure()
    result = opened.run(states_path, reference_path)
    result["status"] = "fresh_external_python_reference_v75"
    result["parent_v72_evidence_digest"] = V72_DIGEST
    result["parent_v74_lock_digest"] = LOCK_DIGEST
    result["adapter_protocol"] = prereg["record_adapter"]
    result["compiler_protocol"] = v72.compiler_protocol()
    result["frozen_external_digest"] = hashlib.sha256(json.dumps({
        "lock": LOCK_DIGEST,
        "v72": V72_DIGEST,
        "protocol": result["protocol"],
        "adapter": prereg["record_adapter"],
        "datasets": [(row["uci_id"], row["sha256"], row["bytes"], row["target"]) for row in manifest["datasets"]],
        "state_input_sha256": result["state_input_sha256"],
        "state_digests": [row["state_digest"] for row in result["rows"]],
    }, sort_keys=True).encode("utf-8")).hexdigest()
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(reference_path: Path, rust_path: Path, output_path: Path) -> dict[str, object]:
    prereg, _ = configure()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rust = json.loads(rust_path.read_text(encoding="utf-8"))
    rust_by_digest = {row["digest"]: row for row in rust["rows"]}
    counters = core.COUNTERS
    mismatches = []
    exact_matches = 0
    for expected in reference["rows"]:
        digest = expected["state_digest"]
        actual = rust_by_digest.get(digest)
        if actual is None:
            mismatches.append({"digest": digest, "kind": "missing-rust-row"})
            continue
        if bool(actual.get("solved")) != bool(expected["bounded_solved"]):
            mismatches.append({"digest": digest, "kind": "solved-status"})
            continue
        if not expected["bounded_solved"]:
            exact_matches += 1
            continue
        if actual.get("plan") != expected["bounded_plan"]:
            mismatches.append({"digest": digest, "kind": "plan"})
            continue
        bad = {field: [expected["bounded_stats"][field], actual.get(field)] for field in counters if int(expected["bounded_stats"][field]) != int(actual.get(field, -1))}
        if bad:
            mismatches.append({"digest": digest, "kind": "counters", "fields": bad})
            continue
        exact_matches += 1
    expected_digests = {row["state_digest"] for row in reference["rows"]}
    unexpected = sorted(set(rust_by_digest) - expected_digests)
    if unexpected:
        mismatches.append({"kind": "unexpected-rust-rows", "digests": unexpected[:20]})

    gate_values = prereg["locked_gate"]
    profiled = int(reference["profiled_state_count"])
    bounded_solved = int(reference["bounded_solved_count"])
    ladder = reference["budget_ladder_summary"]
    gate = (
        reference["archive_lock_digest"] == LOCK_DIGEST
        and reference["parent_v72_evidence_digest"] == V72_DIGEST
        and reference["archive_verification"]["all_hashes_match"]
        and int(reference["contributing_dataset_count"]) == int(gate_values["contributing_datasets"])
        and int(reference["base_state_count"]) >= int(gate_values["minimum_base_states"])
        and profiled >= int(gate_values["minimum_profiled_states"])
        and bounded_solved * 10 >= 9 * profiled
        and int(reference["both_plain_bounded_count"]) >= int(gate_values["minimum_both_plain_bounded"])
        and int(reference["plain_bounded_objective_mismatch_count"]) == 0
        and int(reference["bounded_only_count"]) >= int(gate_values["minimum_bounded_only"])
        and int(reference["current_bounded_plan_mismatch_count"]) == 0
        and int(reference["bounded_expansion_regression_count"]) == 0
        and int(reference["states_with_lower_bound_pruning"]) >= int(gate_values["minimum_states_with_lower_bound_pruning"])
        and float(reference["aggregate_bounded_reduction_fraction"]) >= float(gate_values["minimum_aggregate_bounded_reduction_fraction"])
        and int(reference["dominated_queries_removed"]) >= int(gate_values["minimum_dominated_queries_removed"])
        and int(reference["root_incomparable_classes"]) >= int(gate_values["minimum_root_incomparable_classes"])
        and reference["expansion_ratio_median"] is not None
        and float(reference["expansion_ratio_median"]) >= float(gate_values["minimum_median_plain_bounded_ratio"])
        and reference["expansion_ratio_p90"] is not None
        and float(reference["expansion_ratio_p90"]) >= float(gate_values["minimum_p90_plain_bounded_ratio"])
        and int(ladder["50000"]["bounded_solved"]) >= int(ladder["50000"]["plain_solved"]) + int(gate_values["minimum_50k_solve_advantage"])
        and not mismatches
        and exact_matches == profiled
    )
    result = {
        "status": "fresh_external_label_free_pass_v75" if gate else "fresh_external_label_free_rejected_v75",
        "development_gate": gate,
        "claim_scope": "Clean preregistered external-data evaluation of the frozen v0.72 label-free representation and exact lower-bound mechanism, with independent Rust replay. A pass is not outside-human reproduction, publication novelty, peer review, acceptance, or a world-level claim.",
        "parent_v74_lock_digest": LOCK_DIGEST,
        "parent_v72_evidence_digest": V72_DIGEST,
        "frozen_external_digest": reference["frozen_external_digest"],
        "contributing_dataset_count": reference["contributing_dataset_count"],
        "base_state_count": reference["base_state_count"],
        "profiled_state_count": profiled,
        "bounded_solved_count": bounded_solved,
        "both_plain_bounded_count": reference["both_plain_bounded_count"],
        "bounded_only_count": reference["bounded_only_count"],
        "plain_bounded_objective_mismatch_count": reference["plain_bounded_objective_mismatch_count"],
        "current_bounded_plan_mismatch_count": reference["current_bounded_plan_mismatch_count"],
        "bounded_expansion_regression_count": reference["bounded_expansion_regression_count"],
        "states_with_lower_bound_pruning": reference["states_with_lower_bound_pruning"],
        "aggregate_bounded_reduction_fraction": reference["aggregate_bounded_reduction_fraction"],
        "dominated_queries_removed": reference["dominated_queries_removed"],
        "root_incomparable_classes": reference["root_incomparable_classes"],
        "expansion_ratio_median": reference["expansion_ratio_median"],
        "expansion_ratio_p90": reference["expansion_ratio_p90"],
        "budget_ladder_summary": ladder,
        "rust_total_milliseconds": rust.get("total_milliseconds"),
        "rust_exact_match_count": exact_matches,
        "rust_mismatch_count": len(mismatches),
        "rust_mismatches": mismatches,
        "dataset_summaries": reference["dataset_summaries"],
        "archive_verification": reference["archive_verification"],
        "protocol": reference["protocol"],
    }
    result["evidence_digest"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode("utf-8")).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--states", type=Path, required=True)
    ref.add_argument("--reference", type=Path, required=True)
    val = sub.add_parser("validate")
    val.add_argument("--reference", type=Path, required=True)
    val.add_argument("--rust", type=Path, required=True)
    val.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reference":
        result = run_reference(args.states, args.reference)
    else:
        result = validate(args.reference, args.rust, args.output)
    print(json.dumps({key: result.get(key) for key in ("status", "development_gate", "contributing_dataset_count", "base_state_count", "profiled_state_count", "bounded_solved_count", "bounded_only_count", "expansion_ratio_median", "rust_mismatch_count")}, indent=2))
    if args.command == "validate" and not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
