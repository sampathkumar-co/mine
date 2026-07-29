from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import clean_lower_bound_conditioned_v68 as reference
from . import clean_lower_bound_validate_v68 as validator


REPAIRED_LOCK_DIGEST = "c599849c7ba26fdb3e241221dcae3d3feb26d47ad217da53f0d06b97ebc1e02b"


def configure() -> None:
    reference.LOCK_DIGEST = REPAIRED_LOCK_DIGEST
    validator.LOCK_DIGEST = REPAIRED_LOCK_DIGEST


def main() -> None:
    configure()
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
        result = reference.run(args.states, args.reference)
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
    result = validator.validate(args.reference, args.rust, args.output)
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
