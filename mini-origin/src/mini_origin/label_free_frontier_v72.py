from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import label_free_selector_certificate_v71 as selector
from . import numeric_threshold_frontier_v70 as core

_V70_CONFIGURE_PARENT = core.configure_parent
_V70_PROTOCOL = core.protocol
_V70_COMPACT_STATE = core.compact_state

# Imported for the feature-only record sampler; v0.72 rebuilds the protocol.
from . import numeric_threshold_repaired_v70 as repaired


PARENT_V71_COMMIT = "5594d601356382e45875d26a82052f300d745fe7"
PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v72-label-free-frontier-revalidation.json"
)


def compiler_protocol() -> dict[str, object]:
    return {
        "maximum_sampled_records": core.external.MAX_RECORDS,
        "sampling": (
            "rank original rows by SHA-256(dataset name, feature vector) with "
            "original row index as the duplicate-feature tie-break; labels excluded"
        ),
        "missing_tokens": sorted(core.MISSING_TOKENS),
        "numeric_detection": "all nonmissing tokens parse as finite Decimal",
        "low_cardinality_exact_cutoff": core.LOW_CARDINALITY_EXACT_CUTOFF,
        "quantile_denominator": core.QUANTILE_DENOMINATOR,
        "quantile_numerators": list(core.QUANTILE_NUMERATORS),
        "quantile_rank_rule": "ceil(k*n/8)-1 on sorted nonmissing sampled values",
        "numeric_response_values": ["missing", "le", "gt"],
        "categorical_response": "preserve exact token",
        "labels_or_costs_used": False,
        "query_cap": None,
    }


def protocol() -> dict[str, object]:
    core.compiler_protocol = compiler_protocol
    result = dict(_V70_PROTOCOL())
    result["record_sampler"] = (
        "v0.70 feature-only SHA-256 sampler; labels excluded"
    )
    result["state_selector"] = (
        "v0.60 conditioned-cell selector with the v0.71 feature-only "
        "oversized-cell sampler; labels excluded"
    )
    result["selector_certificate_parent"] = PARENT_V71_COMMIT
    return result


def compact_state(task: object, allowed: int, remaining: int, seed: int):
    row = _V70_COMPACT_STATE(task, allowed, remaining, seed)
    base_digest = hashlib.sha256(
        f"v72:{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    row["base_digest"] = base_digest
    row["digest"] = hashlib.sha256(
        f"{base_digest}:{seed}:label-free-frontier-v72".encode("utf-8")
    ).hexdigest()
    return row


def configure_parent() -> None:
    selector.configure_selector()
    core.external.deterministic_sample = repaired.label_free_sample
    core.compiler_protocol = compiler_protocol
    core.protocol = protocol
    core.compact_state = compact_state
    _V70_CONFIGURE_PARENT()


def configure_module() -> None:
    core.PREREGISTRATION = PREREGISTRATION
    core.compiler_protocol = compiler_protocol
    core.configure_parent = configure_parent
    core.protocol = protocol
    core.compact_state = compact_state


def run_reference(states_path: Path, reference_path: Path) -> dict[str, object]:
    configure_module()
    result = core.run_reference(states_path, reference_path)
    result["status"] = "label_free_frontier_python_reference_v72"
    result["parent_v71_commit"] = PARENT_V71_COMMIT
    result["selector_protocol"] = protocol()["state_selector"]
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(
    reference_path: Path,
    rust_path: Path,
    output_path: Path,
) -> dict[str, object]:
    configure_module()
    result = core.validate(reference_path, rust_path, output_path)
    gate = bool(result["development_gate"])
    result["status"] = (
        "label_free_frontier_revalidation_pass"
        if gate else "label_free_frontier_revalidation_rejected"
    )
    result["claim_scope"] = (
        "Opened-data revalidation of the v0.70 numeric-threshold compiler "
        "and exact lower-bound planner after replacing the final "
        "label-dependent oversized-cell sampling key with the certified "
        "v0.71 feature-only key. The original v0.70 scientific thresholds "
        "remain unchanged. A pass is not fresh external validation, "
        "outside-human reproduction, publication novelty, peer review, "
        "acceptance, or a world-level claim."
    )
    result["parent_v71_commit"] = PARENT_V71_COMMIT
    result["selector_protocol"] = protocol()["state_selector"]
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
