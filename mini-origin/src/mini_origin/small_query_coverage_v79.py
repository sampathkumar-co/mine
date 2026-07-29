from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import conditioned_cell_frontier_v60 as conditioned
from . import label_free_frontier_v72 as frontier
from . import numeric_threshold_frontier_v70 as core
from . import openml_blind_v78 as parent


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v79-small-query-coverage.json"
)
_PARENT_SELECT_STATES = conditioned.select_states
_V72_PROTOCOL = frontier.protocol
_V72_COMPACT_STATE = frontier.compact_state
_V72_CONFIGURE_MODULE = frontier.configure_module
SMALL_QUERY_LIMIT = 10
SMALL_QUERY_SAMPLE_SIZES = (24, 20, 16, 12, 10, 8)
SMALL_QUERY_ALLOWED_SEEDS = 6
SMALL_QUERY_REMAINING_SEEDS = 8

def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def effective_limits(task: object) -> tuple[int, int]:
    minimum_raw = min(
        conditioned.MIN_RAW_QUERIES,
        max(conditioned.MIN_PARTITION_CLASSES + 1, task.query_count - 1),
    )
    minimum_redundancy = min(
        conditioned.MIN_REDUNDANCY,
        max(2, minimum_raw - conditioned.MIN_PARTITION_CLASSES),
    )
    return minimum_raw, minimum_redundancy


def parent_state_set_digest(task: object, rows: list[tuple[int, int, int]]) -> str:
    digests = sorted(
        hashlib.sha256(f"v68:{task.name}:{allowed}:{remaining}".encode("utf-8")).hexdigest()
        for allowed, remaining, _ in rows
    )
    return canonical_digest(digests)

def adaptive_select_states(task: object):
    if task.query_count > SMALL_QUERY_LIMIT:
        rows, summary = _PARENT_SELECT_STATES(task)
        summary.update({
            "adaptive_small_query_mode": False,
            "effective_min_raw_queries": conditioned.MIN_RAW_QUERIES,
            "effective_min_redundancy": conditioned.MIN_REDUNDANCY,
            "selected_state_set_digest": parent_state_set_digest(task, rows),
        })
        return rows, summary

    minimum_raw, minimum_redundancy = effective_limits(task)
    candidates: dict[tuple[int, int], int] = {}
    cells = conditioned.conditioned_cells(task)
    for cell, path_remaining, path in cells:
        cell_size = cell.bit_count()
        allowed_variants = []
        if 8 <= cell_size <= 24:
            allowed_variants.append(cell)
        for size in SMALL_QUERY_SAMPLE_SIZES:
            if cell_size < size:
                continue
            for seed in conditioned.PATH_SEEDS[:SMALL_QUERY_ALLOWED_SEEDS]:
                allowed_variants.append(
                    conditioned.sample_allowed(
                        task, cell, size, f"{path}:{seed}:{size}"
                    )
                )
        for allowed in sorted(set(allowed_variants)):
            for seed in conditioned.PATH_SEEDS[:SMALL_QUERY_REMAINING_SEEDS]:
                remaining, representatives = conditioned.choose_remaining(
                    task, allowed, path_remaining, f"{path}:{seed}"
                )
                raw = remaining.bit_count()
                if (
                    conditioned.MIN_PARTITION_CLASSES
                    <= representatives
                    <= conditioned.MAX_PARTITION_CLASSES
                    and minimum_raw <= raw <= conditioned.MAX_RAW_QUERIES
                    and raw - representatives >= minimum_redundancy
                ):
                    candidates[(allowed, remaining)] = representatives
    rows = [
        (allowed, remaining, representatives)
        for (allowed, remaining), representatives in candidates.items()
    ]
    rows.sort(
        key=lambda row: conditioned.structural_rank(
            task.name, row[0], row[1], row[2]
        )
    )
    rows = rows[:conditioned.MAX_STATES_PER_TASK]
    return rows, {
        "conditioned_cells": len(cells),
        "structural_candidates": len(candidates),
        "selected_states": len(rows),
        "adaptive_small_query_mode": True,
        "effective_min_raw_queries": minimum_raw,
        "effective_min_redundancy": minimum_redundancy,
        "allowed_seed_count": SMALL_QUERY_ALLOWED_SEEDS,
        "remaining_seed_count": SMALL_QUERY_REMAINING_SEEDS,
        "sample_sizes": list(SMALL_QUERY_SAMPLE_SIZES),
        "selected_state_set_digest": parent_state_set_digest(task, rows),
    }

def protocol() -> dict[str, object]:
    result = dict(_V72_PROTOCOL())
    result["state_selector"] = (
        "v0.72 label-free conditioned-cell selector unchanged for tasks with "
        "more than 10 compiled queries; query-count-aware raw/redundancy floors "
        "and broader label-free allowed-set sampling for tasks with 10 or fewer"
    )
    result["small_query_limit"] = SMALL_QUERY_LIMIT
    result["small_query_sample_sizes"] = list(SMALL_QUERY_SAMPLE_SIZES)
    result["small_query_allowed_seeds"] = SMALL_QUERY_ALLOWED_SEEDS
    result["small_query_remaining_seeds"] = SMALL_QUERY_REMAINING_SEEDS
    return result


def compact_state(task: object, allowed: int, remaining: int, seed: int):
    row = _V72_COMPACT_STATE(task, allowed, remaining, seed)
    base_digest = hashlib.sha256(
        f"v79:{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    row["base_digest"] = base_digest
    row["digest"] = hashlib.sha256(
        f"{base_digest}:{seed}:small-query-coverage-v79".encode("utf-8")
    ).hexdigest()
    return row

def install_v79_components() -> None:
    _V72_CONFIGURE_MODULE()
    frontier.protocol = protocol
    frontier.compact_state = compact_state
    conditioned.select_states = adaptive_select_states


def configure_module() -> None:
    install_v79_components()
    frontier.configure_module = install_v79_components


def label_independence_certificate() -> dict[str, object]:
    configure_module()
    manifest = json.loads(parent.MANIFEST.read_text(encoding="utf-8-sig"))
    dataset = next(
        row for row in manifest["datasets"]
        if row["name"] == "analcatdata_dmft"
    )
    payload = parent.download(str(dataset["url"]))
    records, _ = parent.parse_openml_arff(dataset, payload)
    features = [feature_row for feature_row, _ in records]
    rows = []
    for shift in range(8):
        first = [
            (feature_row, str(index % 6))
            for index, feature_row in enumerate(features)
        ]
        second = [
            (feature_row, str((index * (shift + 2) + shift + 1) % 6))
            for index, feature_row in enumerate(features)
        ]
        name = f"v79-opened-label-certificate-{shift}"
        first_task, _ = core.compile_task(name, first)
        second_task, _ = core.compile_task(name, second)
        first_selected, _ = adaptive_select_states(first_task)
        second_selected, _ = adaptive_select_states(second_task)
        rows.append({
            "shift": shift,
            "query_count": first_task.query_count,
            "first_selected": len(first_selected),
            "second_selected": len(second_selected),
            "equal": first_selected == second_selected,
        })
    mismatches = [row for row in rows if not row["equal"]]
    minimum_selected = min(
        min(row["first_selected"], row["second_selected"])
        for row in rows
    )
    return {
        "pair_count": len(rows),
        "mismatch_count": len(mismatches),
        "minimum_selected": minimum_selected,
        "all_equal": not mismatches,
        "rows": rows,
    }


def normal_state_digest_check(
    preregistration: dict[str, object],
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    expected = preregistration["normal_task_state_set_digests"]
    actual = {
        row["task"]: row.get("selected_state_set_digest")
        for row in summaries
        if row["task"] in expected
    }
    mismatches = [
        {
            "task": task,
            "expected": digest,
            "actual": actual.get(task),
        }
        for task, digest in expected.items()
        if actual.get(task) != digest
    ]
    return {
        "expected": expected,
        "actual": actual,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def run_reference(states_path: Path, reference_path: Path):
    configure_module()
    result = parent.run_reference(states_path, reference_path)
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    result["status"] = "small_query_coverage_python_reference_v79"
    result["parent_v78_first_attempt_run"] = preregistration[
        "parent_v78_first_attempt_run"
    ]
    result["selector_protocol"] = protocol()["state_selector"]
    result["normal_state_digest_check"] = normal_state_digest_check(
        preregistration, result["dataset_summaries"]
    )
    result["label_independence_certificate"] = label_independence_certificate()
    result["v79_development_digest"] = hashlib.sha256(
        json.dumps({
            "parent_v78_frozen_external_digest": preregistration[
                "parent_v78_frozen_external_digest"
            ],
            "protocol": result["protocol"],
            "dataset_summaries": result["dataset_summaries"],
            "state_input_sha256": result["state_input_sha256"],
            "state_digests": [row["state_digest"] for row in result["rows"]],
            "normal_state_digest_check": result["normal_state_digest_check"],
            "label_independence_certificate": result[
                "label_independence_certificate"
            ],
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

def validate(reference_path: Path, rust_path: Path, output_path: Path):
    configure_module()
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    core.PREREGISTRATION = PREREGISTRATION
    result = core.validate(reference_path, rust_path, output_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    normal_check = reference["normal_state_digest_check"]
    label_check = reference["label_independence_certificate"]
    extra_gate = (
        int(normal_check["mismatch_count"])
        == int(
            preregistration["development_gate"][
                "normal_task_state_set_digest_mismatches"
            ]
        )
        and int(label_check["mismatch_count"])
        == int(
            preregistration["development_gate"][
                "label_independence_mismatches"
            ]
        )
        and int(label_check["minimum_selected"])
        >= int(
            preregistration["development_gate"][
                "minimum_label_certificate_selected_states"
            ]
        )
    )
    base_gate = bool(result["development_gate"])
    gate = bool(base_gate and extra_gate)
    result["status"] = (
        "small_query_coverage_development_pass_v79"
        if gate
        else "small_query_coverage_development_rejected_v79"
    )
    result["development_gate"] = gate
    result["base_validator_gate"] = base_gate
    result["normal_state_digest_check"] = normal_check
    result["label_independence_certificate"] = label_check
    result["selector_protocol"] = protocol()["state_selector"]
    result["parent_v78_first_attempt_run"] = preregistration[
        "parent_v78_first_attempt_run"
    ]
    result["claim_scope"] = preregistration["claim_boundary"]
    result["v79_development_digest"] = reference["v79_development_digest"]
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
        "gate": result["development_gate"],
        "datasets": result["contributing_dataset_count"],
        "base_states": result["base_state_count"],
        "profiled_states": result["profiled_state_count"],
        "bounded_solved": result["bounded_solved_count"],
        "plain_solved": result["both_plain_bounded_count"],
        "bounded_only": result["bounded_only_count"],
        "median": result["expansion_ratio_median"],
        "rust_mismatches": result["rust_mismatch_count"],
        "normal_state_mismatches": result[
            "normal_state_digest_check"
        ]["mismatch_count"],
        "label_independence_mismatches": result[
            "label_independence_certificate"
        ]["mismatch_count"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
