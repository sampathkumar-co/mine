from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import near_small_query_coverage_v83 as repair
from . import pmlb_blind_v82 as parent


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v83-near-small-query-coverage.json"
)


def _install() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "opened_data_development_preregistered_after_v82_rejection":
        raise RuntimeError("v0.83 preregistration status changed")
    if int(preregistration["selector_change"]["old_adaptive_limit"]) != 10:
        raise RuntimeError("v0.83 old selector boundary changed")
    if int(preregistration["selector_change"]["new_adaptive_limit"]) != 12:
        raise RuntimeError("v0.83 new selector boundary changed")
    if preregistration["development_data_status"] != "opened":
        raise RuntimeError("v0.83 must remain an opened-data development campaign")
    if preregistration["fresh_external_evidence"] is not False:
        raise RuntimeError("v0.83 cannot be classified as fresh external evidence")
    parent.selector = repair
    return preregistration


def run_reference(states_path: Path, reference_path: Path):
    preregistration = _install()
    result = parent.run_reference(states_path, reference_path)
    result["status"] = "pmlb_opened_data_python_reference_v83"
    result["development_data_status"] = "opened"
    result["fresh_external_evidence"] = False
    result["selector_revision"] = "near-small-query-coverage-v83"
    result["adaptive_query_limit"] = repair.NEAR_SMALL_QUERY_LIMIT
    result["claim_scope"] = preregistration["claim_boundary"]
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(reference_path: Path, rust_path: Path, output_path: Path):
    preregistration = _install()
    result = parent.validate(reference_path, rust_path, output_path)
    gate = bool(result["external_gate"])
    result["status"] = (
        "near_small_query_coverage_development_pass_v83"
        if gate
        else "near_small_query_coverage_development_rejected_v83"
    )
    result["development_gate"] = gate
    result.pop("external_gate", None)
    result["development_data_status"] = "opened"
    result["fresh_external_evidence"] = False
    result["selector_revision"] = "near-small-query-coverage-v83"
    result["adaptive_query_limit"] = repair.NEAR_SMALL_QUERY_LIMIT
    result["claim_scope"] = preregistration["claim_boundary"]
    result["evidence_digest"] = repair.canonical_digest(result)
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
