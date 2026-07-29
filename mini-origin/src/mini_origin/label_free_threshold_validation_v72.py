from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import conditioned_cell_frontier_v60 as conditioned
from . import label_free_selector_certificate_v71 as selector
from . import numeric_threshold_repaired_v70 as numeric

PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v72-label-free-threshold-validation.json"
)


def configure() -> None:
    # The only selector change relative to v0.70: labels are excluded from
    # oversized-cell sampling. The numeric compiler and exact solvers remain frozen.
    conditioned.sample_allowed = selector.label_free_sample_allowed


def verify_preregistration() -> dict[str, object]:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg["parent_v71_evidence_digest"] != "58382c7aa2bcd7c28fec54642fcf76cc88e5059e0eb62e942d6cccf33d6ddfe2":
        raise RuntimeError("unexpected v0.71 certificate digest")
    if prereg["parent_v68_evidence_digest"] != "b2ff35cbc40d0c2828fa26a3057c245d5c794f4ea9164b3f560c7bcfba50448b":
        raise RuntimeError("unexpected v0.68 evidence digest")
    return prereg


def main() -> None:
    verify_preregistration()
    configure()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("--states", type=Path, required=True)
    reference.add_argument("--reference", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--reference", type=Path, required=True)
    validate.add_argument("--rust", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reference":
        result = numeric.run_reference(args.states, args.reference)
        print(json.dumps({
            "status": result["status"],
            "datasets": result["contributing_dataset_count"],
            "base_states": result["base_state_count"],
            "profiled_states": result["profiled_state_count"],
            "bounded_solved": result["bounded_solved_count"],
        }, indent=2))
        return
    result = numeric.validate(args.reference, args.rust, args.output)
    # Preserve the v0.70 gate exactly. v0.72 changes only label dependence.
    print(json.dumps({
        "status": result["status"],
        "gate": result["development_gate"],
        "datasets": result["contributing_dataset_count"],
        "base_states": result["base_state_count"],
        "profiled_states": result["profiled_state_count"],
        "bounded_solved": result["bounded_solved_count"],
        "bounded_only": result["bounded_only_count"],
        "median": result["expansion_ratio_median"],
        "rust_mismatches": result["rust_mismatch_count"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
