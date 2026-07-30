from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pmlb_blind_v82 as parent
from . import response_lattice_closure_v85 as repair


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v85-response-lattice-closure.json"
)
IMPLEMENTATION_AMENDMENT = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v85-response-lattice-implementation-amendment.json"
)


def _install() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    amendment = json.loads(IMPLEMENTATION_AMENDMENT.read_text(encoding="utf-8"))
    if preregistration["status"] != "preregistered_before_implementation_or_dataset_evaluation":
        raise RuntimeError("v0.85 preregistration status changed")
    if preregistration["single_allowed_algorithmic_change"]["name"] != "response_lattice_closure_generator":
        raise RuntimeError("v0.85 allowed algorithmic change changed")
    if preregistration["opened_data_gate"]["contributing_dataset_count"] != 7:
        raise RuntimeError("v0.85 all-seven-dataset gate changed")
    if preregistration["opened_data_gate"]["threshold_source"] != "unchanged v0.82 preregistration":
        raise RuntimeError("v0.85 threshold source changed")
    if amendment["status"] != "implementation_amendment_before_selector_integration_or_opened_data_evaluation":
        raise RuntimeError("v0.85 implementation amendment status changed")
    repair.configure_module()
    parent.selector = repair
    return preregistration


def run_reference(states_path: Path, reference_path: Path):
    preregistration = _install()
    result = parent.run_reference(states_path, reference_path)
    result["status"] = "pmlb_opened_data_python_reference_v85"
    result["development_data_status"] = "opened"
    result["fresh_external_evidence"] = False
    result["selector_revision"] = "response-lattice-closure-v85"
    result["implementation_amendment"] = IMPLEMENTATION_AMENDMENT.name
    result["claim_scope"] = preregistration["claim_boundary"]
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(reference_path: Path, rust_path: Path, output_path: Path):
    preregistration = _install()
    result = parent.validate(reference_path, rust_path, output_path)
    gate = bool(result["external_gate"])
    result["status"] = (
        "response_lattice_closure_development_pass_v85"
        if gate
        else "response_lattice_closure_development_rejected_v85"
    )
    result["development_gate"] = gate
    result.pop("external_gate", None)
    result["development_data_status"] = "opened"
    result["fresh_external_evidence"] = False
    result["selector_revision"] = "response-lattice-closure-v85"
    result["implementation_amendment"] = IMPLEMENTATION_AMENDMENT.name
    result["claim_scope"] = preregistration["claim_boundary"]
    result["evidence_digest"] = parent.canonical_digest(result)
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
        }, indent=2))
        return
    result = validate(args.reference, args.rust, args.output)
    print(json.dumps({
        "status": result["status"],
        "gate": result["development_gate"],
        "datasets": result["contributing_dataset_count"],
        "base_states": result["base_state_count"],
        "profiled_states": result["profiled_state_count"],
        "rust_mismatches": result["rust_mismatch_count"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
