from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
from pathlib import Path

import numpy as np

from . import clean_lower_bound_conditioned_v68 as opened
from . import clean_lower_bound_validate_v68 as parent_validator
from . import conditioned_cell_frontier_v60 as conditioned
from . import external_response_cost_v58 as external
from . import response_cost_export_v57 as export_v57
from . import response_cost_lower_bound_v65 as bounded
from . import response_cost_pareto_v56 as response
from . import state_policy_v34 as state


LOCK_DIGEST = "c599849c7ba26fdb3e241221dcae3d3feb26d47ad217da53f0d06b97ebc1e02b"
REGISTRY_DIGEST = "b88fcb352c2f80af8bc89a3a7576b9cd384800b67d1b168534ad26df9985b6c1"
V66_DIGEST = "3b2bb026556ff9f6321ad6a8375854ae46931e64080329c76f86f31d12c0d643"
PARENT_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v68-clean-lower-bound-rejected.json"
)
PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v70-numeric-threshold-frontier-preregistration.json"
)
MISSING_TOKENS = frozenset(("", "?", "na", "nan", "none", "null"))
LOW_CARDINALITY_EXACT_CUTOFF = 8
QUANTILE_DENOMINATOR = 8
QUANTILE_NUMERATORS = tuple(range(1, 8))
COUNTERS = parent_validator.COUNTERS
_ORIGINAL_TASK_FROM_RECORDS = external.task_from_records
_ORIGINAL_PROTOCOL = opened.protocol


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS


def finite_decimal(value: str) -> Decimal | None:
    if is_missing(value):
        return None
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation:
        raise ValueError(value)
    if not parsed.is_finite():
        raise ValueError(value)
    return parsed


def decimal_name(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def quantile_thresholds(values: list[Decimal]) -> tuple[Decimal, ...]:
    ordered = sorted(values)
    if not ordered:
        return ()
    maximum = ordered[-1]
    thresholds = []
    for numerator in QUANTILE_NUMERATORS:
        rank = int(
            (Decimal(numerator * len(ordered)) / QUANTILE_DENOMINATOR)
            .to_integral_value(rounding=ROUND_CEILING)
        ) - 1
        rank = max(0, min(rank, len(ordered) - 1))
        threshold = ordered[rank]
        if threshold >= maximum:
            continue
        if threshold not in thresholds:
            thresholds.append(threshold)
    return tuple(thresholds)


def compile_task(
    name: str,
    records: list[tuple[tuple[str, ...], str]],
):
    sampled = external.deterministic_sample(name, records)
    if not sampled:
        raise RuntimeError(f"no sampled records for {name}")
    width = len(sampled[0][0])
    if any(len(features) != width for features, _ in sampled):
        raise RuntimeError(f"inconsistent width for {name}")

    compiled_names: list[str] = []
    response_columns: list[tuple[str, ...]] = []
    numeric_columns = 0
    threshold_columns = 0
    exact_columns = 0
    threshold_queries = 0
    per_column = []

    for feature_index in range(width):
        tokens = [features[feature_index].strip() for features, _ in sampled]
        nonmissing_tokens = [token for token in tokens if not is_missing(token)]
        parsed_values: list[Decimal] = []
        numeric = bool(nonmissing_tokens)
        if numeric:
            try:
                parsed_values = [finite_decimal(token) for token in nonmissing_tokens]  # type: ignore[list-item]
            except ValueError:
                numeric = False
                parsed_values = []
        distinct_nonmissing = len(set(nonmissing_tokens))

        if numeric:
            numeric_columns += 1
        if numeric and distinct_nonmissing > LOW_CARDINALITY_EXACT_CUTOFF:
            thresholds = quantile_thresholds(parsed_values)
            local_count = 0
            for threshold in thresholds:
                responses = []
                for token in tokens:
                    if is_missing(token):
                        responses.append("missing")
                    else:
                        value = finite_decimal(token)
                        assert value is not None
                        responses.append("le" if value <= threshold else "gt")
                if len(set(responses)) <= 1:
                    continue
                compiled_names.append(
                    f"f{feature_index}<=${decimal_name(threshold)}".replace("$", "")
                )
                response_columns.append(tuple(responses))
                local_count += 1
            if local_count:
                threshold_columns += 1
                threshold_queries += local_count
                per_column.append({
                    "feature_index": feature_index,
                    "kind": "numeric-threshold",
                    "distinct_nonmissing": distinct_nonmissing,
                    "queries": local_count,
                })
                continue

        compiled_names.append(f"f{feature_index}=exact")
        response_columns.append(tuple(
            "missing" if is_missing(token) else token
            for token in tokens
        ))
        exact_columns += 1
        per_column.append({
            "feature_index": feature_index,
            "kind": "exact",
            "numeric": numeric,
            "distinct_nonmissing": distinct_nonmissing,
            "queries": 1,
        })

    if not response_columns:
        raise RuntimeError(f"compiler produced no queries for {name}")
    compiled_rows = tuple(
        tuple(column[row_index] for column in response_columns)
        for row_index in range(len(sampled))
    )
    labels = tuple(label for _, label in sampled)
    task = state.base.make_task(
        name,
        tuple(compiled_names),
        compiled_rows,
        labels,
    )
    summary = {
        "raw_records": len(records),
        "distinct_records": len(set(records)),
        "sampled_records": len(sampled),
        "original_features": width,
        "compiled_queries": len(compiled_names),
        "numeric_columns": numeric_columns,
        "threshold_columns": threshold_columns,
        "exact_columns": exact_columns,
        "threshold_queries": threshold_queries,
        "labels": len(set(labels)),
        "compiler_columns": per_column,
    }
    return task, summary


def compiler_protocol() -> dict[str, object]:
    return {
        "maximum_sampled_distinct_records": external.MAX_RECORDS,
        "missing_tokens": sorted(MISSING_TOKENS),
        "numeric_detection": "all nonmissing tokens parse as finite Decimal",
        "low_cardinality_exact_cutoff": LOW_CARDINALITY_EXACT_CUTOFF,
        "quantile_denominator": QUANTILE_DENOMINATOR,
        "quantile_numerators": list(QUANTILE_NUMERATORS),
        "quantile_rank_rule": "ceil(k*n/8)-1 on sorted nonmissing sampled values",
        "numeric_response_values": ["missing", "le", "gt"],
        "categorical_response": "preserve exact token",
        "labels_or_costs_used": False,
        "query_cap": None,
    }


def protocol() -> dict[str, object]:
    result = dict(_ORIGINAL_PROTOCOL())
    result["query_compiler"] = compiler_protocol()
    result["state_selector"] = "byte-identical v0.60 conditioned-cell selector"
    result["lower_bound_solver"] = "byte-identical v0.65 planner"
    return result


def compact_state(task: object, allowed: int, remaining: int, seed: int):
    row = export_v57.compact_state(task, allowed, remaining, seed)
    base_digest = hashlib.sha256(
        f"v70:{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    row["base_digest"] = base_digest
    row["digest"] = hashlib.sha256(
        f"{base_digest}:{seed}:numeric-threshold-v70".encode("utf-8")
    ).hexdigest()
    return row


def configure_parent() -> None:
    opened.LOCK_DIGEST = LOCK_DIGEST
    opened.REGISTRY_DIGEST = REGISTRY_DIGEST
    opened.V66_DIGEST = V66_DIGEST
    opened.external.task_from_records = compile_task
    opened.compact_state = compact_state
    opened.protocol = protocol


def run_reference(states_path: Path, reference_path: Path) -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_EVIDENCE.read_text(encoding="utf-8"))
    if parent["development_gate"]:
        raise RuntimeError("v0.68 parent must remain rejected")
    if (
        parent["evidence_digest"]
        != preregistration["parent_v68_evidence_digest"]
    ):
        raise RuntimeError("v0.68 parent evidence changed")
    configure_parent()
    result = opened.run(states_path, reference_path)
    result["status"] = "numeric_threshold_python_reference_v70"
    result["parent_v68_evidence_digest"] = parent["evidence_digest"]
    result["compiler_protocol"] = compiler_protocol()
    result["frozen_external_digest"] = hashlib.sha256(
        json.dumps({
            "parent_v68_evidence_digest": parent["evidence_digest"],
            "archive_lock_digest": result["archive_lock_digest"],
            "protocol": result["protocol"],
            "dataset_summaries": result["dataset_summaries"],
            "state_input_sha256": result["state_input_sha256"],
            "state_digests": [row["state_digest"] for row in result["rows"]],
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(
    reference_path: Path,
    rust_path: Path,
    output_path: Path,
) -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rust = json.loads(rust_path.read_text(encoding="utf-8"))
    rust_by_digest = {row["digest"]: row for row in rust["rows"]}
    mismatches = []
    exact_matches = 0
    for expected in reference["rows"]:
        digest = expected["state_digest"]
        actual = rust_by_digest.get(digest)
        if actual is None:
            mismatches.append({"digest": digest, "kind": "missing-rust-row"})
            continue
        if bool(actual.get("solved")) != bool(expected["bounded_solved"]):
            mismatches.append({
                "digest": digest,
                "kind": "solved-status",
                "python": expected["bounded_solved"],
                "rust": actual.get("solved"),
            })
            continue
        if not expected["bounded_solved"]:
            exact_matches += 1
            continue
        if actual["plan"] != expected["bounded_plan"]:
            mismatches.append({
                "digest": digest,
                "kind": "plan",
                "python": expected["bounded_plan"],
                "rust": actual["plan"],
            })
            continue
        bad = {
            field: {
                "python": expected["bounded_stats"][field],
                "rust": actual.get(field),
            }
            for field in COUNTERS
            if int(expected["bounded_stats"][field])
            != int(actual.get(field, -1))
        }
        if bad:
            mismatches.append({
                "digest": digest,
                "kind": "counters",
                "fields": bad,
            })
            continue
        exact_matches += 1
    expected_digests = {row["state_digest"] for row in reference["rows"]}
    unexpected = sorted(set(rust_by_digest) - expected_digests)
    if unexpected:
        mismatches.append({
            "kind": "unexpected-rust-rows",
            "digests": unexpected[:20],
        })

    gate_values = preregistration["locked_gate"]
    summaries = {
        row["task"]: row for row in reference["dataset_summaries"]
    }
    previously_zero_ok = all(
        int(summaries[name]["selected_states"])
        >= int(gate_values["minimum_states_from_each_previously_zero_dataset"])
        for name in gate_values["previously_zero_datasets"]
    )
    profiled = int(reference["profiled_state_count"])
    bounded_solved = int(reference["bounded_solved_count"])
    ladder = reference["budget_ladder_summary"]
    gate = (
        reference["archive_verification"]["all_hashes_match"]
        and int(reference["contributing_dataset_count"])
        == int(gate_values["contributing_datasets"])
        and previously_zero_ok
        and int(reference["base_state_count"])
        >= int(gate_values["minimum_base_states"])
        and profiled >= int(gate_values["minimum_profiled_states"])
        and bounded_solved * 10 >= 9 * profiled
        and int(reference["both_plain_bounded_count"])
        >= int(gate_values["minimum_both_plain_bounded"])
        and int(reference["plain_bounded_objective_mismatch_count"])
        == int(gate_values["plain_bounded_objective_mismatches"])
        and int(reference["bounded_only_count"])
        >= int(gate_values["minimum_bounded_only"])
        and int(reference["current_bounded_plan_mismatch_count"])
        == int(gate_values["current_bounded_plan_mismatches"])
        and int(reference["bounded_expansion_regression_count"])
        == int(gate_values["bounded_expansion_regressions"])
        and int(reference["states_with_lower_bound_pruning"])
        >= int(gate_values["minimum_states_with_lower_bound_pruning"])
        and float(reference["aggregate_bounded_reduction_fraction"])
        >= float(gate_values["minimum_aggregate_bounded_reduction_fraction"])
        and int(reference["dominated_queries_removed"])
        >= int(gate_values["minimum_dominated_queries_removed"])
        and int(reference["root_incomparable_classes"])
        >= int(gate_values["minimum_root_incomparable_classes"])
        and reference["expansion_ratio_median"] is not None
        and float(reference["expansion_ratio_median"])
        >= float(gate_values["minimum_median_plain_bounded_ratio"])
        and reference["expansion_ratio_p90"] is not None
        and float(reference["expansion_ratio_p90"])
        >= float(gate_values["minimum_p90_plain_bounded_ratio"])
        and int(ladder["50000"]["bounded_solved"])
        >= int(ladder["50000"]["plain_solved"])
        + int(gate_values["minimum_50k_solve_advantage"])
        and not mismatches
        and exact_matches == profiled
    )
    result = {
        "status": (
            "numeric_threshold_frontier_development_pass"
            if gate else "numeric_threshold_frontier_development_rejected"
        ),
        "development_gate": gate,
        "claim_scope": (
            "Label-free numeric-threshold query compilation combined with the "
            "unchanged v0.60 selector and independently reproduced v0.66 exact "
            "lower-bound solver, evaluated on opened v0.68 archives only. A pass "
            "freezes a representation for a later fresh campaign and is not "
            "external validation or a world-level claim."
        ),
        "parent_v68_evidence_digest": reference["parent_v68_evidence_digest"],
        "archive_lock_digest": reference["archive_lock_digest"],
        "frozen_external_digest": reference["frozen_external_digest"],
        "compiler_protocol": reference["compiler_protocol"],
        "contributing_dataset_count": reference["contributing_dataset_count"],
        "previously_zero_datasets_passed": previously_zero_ok,
        "base_state_count": reference["base_state_count"],
        "profiled_state_count": profiled,
        "bounded_solved_count": bounded_solved,
        "both_plain_bounded_count": reference["both_plain_bounded_count"],
        "bounded_only_count": reference["bounded_only_count"],
        "plain_bounded_objective_mismatch_count": reference[
            "plain_bounded_objective_mismatch_count"
        ],
        "current_bounded_plan_mismatch_count": reference[
            "current_bounded_plan_mismatch_count"
        ],
        "bounded_expansion_regression_count": reference[
            "bounded_expansion_regression_count"
        ],
        "states_with_lower_bound_pruning": reference[
            "states_with_lower_bound_pruning"
        ],
        "current_query_expansions": reference["current_query_expansions"],
        "bounded_query_expansions": reference["bounded_query_expansions"],
        "aggregate_bounded_reduction_fraction": reference[
            "aggregate_bounded_reduction_fraction"
        ],
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
        "protocol": reference["protocol"],
    }
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
