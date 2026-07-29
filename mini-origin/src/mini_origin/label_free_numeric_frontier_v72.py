from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import conditioned_cell_frontier_v60 as conditioned
from . import label_free_selector_certificate_v71 as selector
from . import numeric_threshold_frontier_v70 as original
from . import numeric_threshold_repaired_v70 as repaired
from . import response_cost_export_v57 as export_v57


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v72-label-free-threshold-frontier-preregistration.json"
)
PARENT_V68 = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v68-clean-lower-bound-rejected.json"
)
V71_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v71-label-free-selector-pass.json"
)
V71_DIGEST = "58382c7aa2bcd7c28fec54642fcf76cc88e5059e0eb62e942d6cccf33d6ddfe2"
PARENT_V68_DIGEST = "b2ff35cbc40d0c2828fa26a3057c245d5c794f4ea9164b3f560c7bcfba50448b"


def compiler_protocol() -> dict[str, object]:
    result = dict(repaired.compiler_protocol())
    result["oversized_cell_sampling"] = (
        "SHA-256(task name, salt, sampled row index, compiled response row); "
        "labels excluded"
    )
    result["labels_or_costs_used"] = False
    return result


def protocol() -> dict[str, object]:
    result = dict(original._ORIGINAL_PROTOCOL())
    result["query_compiler"] = compiler_protocol()
    result["state_selector"] = (
        "v0.60 conditioned-cell traversal and structural criteria with only "
        "sample_allowed replaced by the v0.71 label-free sampler"
    )
    result["lower_bound_solver"] = "byte-identical v0.65 planner"
    return result


def compact_state(task: object, allowed: int, remaining: int, seed: int):
    row = export_v57.compact_state(task, allowed, remaining, seed)
    base_digest = hashlib.sha256(
        f"v72:{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    row["base_digest"] = base_digest
    row["digest"] = hashlib.sha256(
        f"{base_digest}:{seed}:label-free-threshold-v72".encode("utf-8")
    ).hexdigest()
    return row


def configure() -> None:
    conditioned.sample_allowed = selector.label_free_sample_allowed
    original.external.deterministic_sample = repaired.label_free_sample
    original.compiler_protocol = compiler_protocol
    original.protocol = protocol
    original.compact_state = compact_state


def verify_frozen_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_V68.read_text(encoding="utf-8"))
    v71 = json.loads(V71_EVIDENCE.read_text(encoding="utf-8"))
    v70_preregistration = json.loads(
        original.PREREGISTRATION.read_text(encoding="utf-8")
    )
    if parent.get("development_gate"):
        raise RuntimeError("v0.68 parent must remain rejected")
    if parent.get("evidence_digest") != PARENT_V68_DIGEST:
        raise RuntimeError("v0.68 evidence digest changed")
    if preregistration.get("parent_v68_evidence_digest") != PARENT_V68_DIGEST:
        raise RuntimeError("v0.72 parent v0.68 digest changed")
    if not v71.get("development_gate") or v71.get("evidence_digest") != V71_DIGEST:
        raise RuntimeError("v0.71 label-free selector evidence changed")
    if preregistration.get("parent_v71_evidence_digest") != V71_DIGEST:
        raise RuntimeError("v0.72 parent v0.71 digest changed")
    if preregistration.get("locked_gate") != v70_preregistration.get("locked_gate"):
        raise RuntimeError("v0.72 gate differs from frozen v0.70 gate")
    if preregistration.get("exact_budget") != v70_preregistration.get("exact_budget"):
        raise RuntimeError("v0.72 budget differs from frozen v0.70 budget")
    return preregistration, parent, v71


def run_reference(states_path: Path, reference_path: Path) -> dict[str, object]:
    preregistration, parent, v71 = verify_frozen_inputs()
    configure()
    result = original.run_reference(states_path, reference_path)
    result["status"] = "label_free_numeric_threshold_python_reference_v72"
    result["parent_v68_evidence_digest"] = parent["evidence_digest"]
    result["parent_v71_evidence_digest"] = v71["evidence_digest"]
    result["compiler_protocol"] = compiler_protocol()
    result["protocol"] = protocol()
    result["frozen_external_digest"] = hashlib.sha256(
        json.dumps(
            {
                "parent_v68_evidence_digest": parent["evidence_digest"],
                "parent_v71_evidence_digest": v71["evidence_digest"],
                "archive_lock_digest": result["archive_lock_digest"],
                "protocol": result["protocol"],
                "dataset_summaries": result["dataset_summaries"],
                "state_input_sha256": result["state_input_sha256"],
                "state_digests": [row["state_digest"] for row in result["rows"]],
                "locked_gate": preregistration["locked_gate"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(reference_path: Path, rust_path: Path, output_path: Path) -> dict[str, object]:
    preregistration, _, v71 = verify_frozen_inputs()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("parent_v71_evidence_digest") != V71_DIGEST:
        raise RuntimeError("reference was not produced with v0.71 selector evidence")
    configure()
    result = original.validate(reference_path, rust_path, output_path)
    result["status"] = (
        "label_free_numeric_threshold_frontier_v72_pass"
        if result["development_gate"]
        else "label_free_numeric_threshold_frontier_v72_rejected"
    )
    result["claim_scope"] = preregistration["claim_boundary"]
    result["parent_v71_evidence_digest"] = v71["evidence_digest"]
    result["compiler_protocol"] = compiler_protocol()
    result["protocol"] = protocol()
    result.pop("evidence_digest", None)
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    reference_parser = subparsers.add_parser("reference")
    reference_parser.add_argument("--states", type=Path, required=True)
    reference_parser.add_argument("--reference", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
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
        "gate": result["development_gate"],
        "datasets": result["contributing_dataset_count"],
        "base_states": result["base_state_count"],
        "profiled_states": result["profiled_state_count"],
        "bounded_solved": result["bounded_solved_count"],
        "plain_solved": result["both_plain_bounded_count"],
        "bounded_only": result["bounded_only_count"],
        "median": result["expansion_ratio_median"],
        "rust_mismatches": result["rust_mismatch_count"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
