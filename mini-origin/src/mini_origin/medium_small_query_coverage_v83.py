from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import conditioned_cell_frontier_v60 as conditioned
from . import numeric_threshold_frontier_v70 as core
from . import pmlb_blind_v82 as parent
from . import small_query_coverage_v79 as v79


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v83-medium-small-query-coverage.json"
)
V82_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v82-pmlb-blind-rejected.json"
)
V82_EVIDENCE_SHA256 = "f3b0b4367e8e5e7c8dfba92f914f6a78596d7294d20dcdc960b5d78ab45ae4a0"
V82_EVIDENCE_DIGEST = "6c5e6e2c48a164c1c38c4cfff205b6ad33e6d44891b6e953ca1623baa6c72863"
FROZEN_V82_COMMIT = "19b5ae7fd97d92c75451269e78a032d0f298c8d7"
SOLAR_TASK = "_deprecated_solar_flare_1"
MEDIUM_SMALL_MIN = 11
MEDIUM_SMALL_MAX = 16
SAMPLE_SIZES = (24, 20, 16, 12, 10, 8)
ALLOWED_SEEDS = 6
REMAINING_SEEDS = 8


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_inputs() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "opened_data_development_preregistration":
        raise RuntimeError("v0.83 preregistration status changed")
    if preregistration["parent_v82_commit"] != FROZEN_V82_COMMIT:
        raise RuntimeError("frozen v0.82 commit changed")
    if preregistration["parent_v82_evidence_sha256"] != V82_EVIDENCE_SHA256:
        raise RuntimeError("v0.82 evidence SHA changed")
    if preregistration["parent_v82_evidence_digest"] != V82_EVIDENCE_DIGEST:
        raise RuntimeError("v0.82 evidence digest changed")
    if preregistration["fresh_blind_claim"] is not False:
        raise RuntimeError("v0.83 cannot be a fresh blind claim")
    if int(preregistration["exact_solver_revisions"]) != 0:
        raise RuntimeError("exact solver revision budget changed")
    if int(preregistration["compiler_revisions"]) != 0:
        raise RuntimeError("compiler revision budget changed")
    if int(preregistration["selector_revisions"]) != 1:
        raise RuntimeError("selector revision budget changed")
    fallback = preregistration["fallback_selector"]
    if fallback["dataset_specific_exceptions"] is not False:
        raise RuntimeError("dataset-specific fallback introduced")
    if fallback["label_or_cost_use"] is not False:
        raise RuntimeError("fallback label boundary changed")
    if fallback["preserve_nonzero_parent_selection_exactly"] is not True:
        raise RuntimeError("parent state preservation changed")

    evidence_bytes = V82_EVIDENCE.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != V82_EVIDENCE_SHA256:
        raise RuntimeError("v0.82 evidence bytes changed")
    evidence = json.loads(evidence_bytes.decode("utf-8"))
    if evidence["status"] != "pmlb_cross_source_blind_rejected_v82":
        raise RuntimeError("v0.82 parent must remain rejected")
    if evidence["evidence_digest"] != V82_EVIDENCE_DIGEST:
        raise RuntimeError("unexpected v0.82 evidence digest")
    if evidence["adapter_verification_passed"] is not True:
        raise RuntimeError("v0.82 adapter verification changed")
    if int(evidence["rust_mismatch_count"]) != 0:
        raise RuntimeError("v0.82 Rust mismatch count changed")

    v82_preregistration, manifest, v79_evidence = parent.load_frozen_inputs()
    return preregistration, v82_preregistration, manifest, evidence


def fallback_limits(task: object) -> tuple[int, int]:
    minimum_raw = min(10, max(7, int(task.query_count) - 4))
    minimum_redundancy = min(4, max(2, minimum_raw - 6))
    return minimum_raw, minimum_redundancy


def label_free_sample_allowed(
    task: object,
    cell: int,
    size: int,
    salt: str,
) -> int:
    indices: list[int] = []
    pending = cell
    while pending:
        bit = pending & -pending
        indices.append(bit.bit_length() - 1)
        pending ^= bit
    indices.sort(
        key=lambda index: conditioned.hash_token(
            "v83-label-free", task.name, salt, index, task.rows[index]
        )
    )
    mask = 0
    for index in indices[:size]:
        mask |= 1 << index
    return mask


def fallback_select_states(task: object):
    minimum_raw, minimum_redundancy = fallback_limits(task)
    candidates: dict[tuple[int, int], int] = {}
    cells = conditioned.conditioned_cells(task)
    for cell, path_remaining, path in cells:
        cell_size = cell.bit_count()
        allowed_variants: list[int] = []
        if 8 <= cell_size <= 24:
            allowed_variants.append(cell)
        for size in SAMPLE_SIZES:
            if cell_size < size:
                continue
            for seed in conditioned.PATH_SEEDS[:ALLOWED_SEEDS]:
                allowed_variants.append(
                    label_free_sample_allowed(task, cell, size, f"{path}:{seed}:{size}")
                )
        for allowed in sorted(set(allowed_variants)):
            for seed in conditioned.PATH_SEEDS[:REMAINING_SEEDS]:
                remaining, representatives = conditioned.choose_remaining(
                    task, allowed, path_remaining, f"v83:{path}:{seed}"
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
        "adaptive_small_query_mode": False,
        "medium_small_zero_candidate_fallback": True,
        "effective_min_raw_queries": minimum_raw,
        "effective_min_redundancy": minimum_redundancy,
        "allowed_seed_count": ALLOWED_SEEDS,
        "remaining_seed_count": REMAINING_SEEDS,
        "sample_sizes": list(SAMPLE_SIZES),
        "selected_state_set_digest": v79.parent_state_set_digest(task, rows),
    }


def select_states(task: object):
    rows, summary = v79.adaptive_select_states(task)
    if rows or not (MEDIUM_SMALL_MIN <= int(task.query_count) <= MEDIUM_SMALL_MAX):
        summary = dict(summary)
        summary["medium_small_zero_candidate_fallback"] = False
        return rows, summary
    return fallback_select_states(task)


def protocol() -> dict[str, object]:
    result = dict(v79.protocol())
    result["state_selector"] = (
        "frozen v0.79 selector, plus a label-free zero-candidate fallback only "
        "when 11 to 16 compiled queries produce no parent states"
    )
    result["medium_small_fallback_range"] = [MEDIUM_SMALL_MIN, MEDIUM_SMALL_MAX]
    result["medium_small_sample_sizes"] = list(SAMPLE_SIZES)
    result["medium_small_allowed_seeds"] = ALLOWED_SEEDS
    result["medium_small_remaining_seeds"] = REMAINING_SEEDS
    result["medium_small_label_free_sampler"] = True
    result["preserve_nonzero_parent_selection_exactly"] = True
    return result


compact_state = v79.compact_state
frontier = v79.frontier


def install_v83_components() -> None:
    v79.install_v79_components()
    frontier.protocol = protocol
    conditioned.select_states = select_states
    frontier.configure_module = install_v83_components


def configure_module() -> None:
    install_v83_components()


def prepare_opened(
    v82_preregistration: dict[str, object],
    manifest: dict[str, object],
    adapted_path: Path,
) -> None:
    parent.selector = sys.modules[__name__]
    configure_module()
    parent.prepare_opened(v82_preregistration, manifest, adapted_path)


def frozen_state_digest_check(
    preregistration: dict[str, object],
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    expected = preregistration["frozen_nonzero_state_set_digests"]
    actual = {
        str(row["task"]): row.get("selected_state_set_digest")
        for row in summaries
        if str(row["task"]) in expected
    }
    mismatches = [
        {"task": task, "expected": digest, "actual": actual.get(task)}
        for task, digest in expected.items()
        if actual.get(task) != digest
    ]
    return {
        "expected": expected,
        "actual": actual,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def solar_records(manifest: dict[str, object]):
    dataset = next(
        row for row in manifest["datasets"]
        if str(row["name"]) == SOLAR_TASK
    )
    payload = parent.download(str(dataset["raw_url"]))
    records, _ = parent.parse_pmlb_table(dataset, payload)
    return records


def label_independence_certificate(manifest: dict[str, object]) -> dict[str, object]:
    configure_module()
    records = solar_records(manifest)
    features = [feature_row for feature_row, _ in records]
    rows: list[dict[str, object]] = []
    for shift in range(8):
        first = [
            (feature_row, str(index % 5))
            for index, feature_row in enumerate(features)
        ]
        second = [
            (feature_row, str((index * (shift + 2) + shift + 1) % 5))
            for index, feature_row in enumerate(features)
        ]
        name = f"v83-opened-solar-label-certificate-{shift}"
        first_task, _ = core.compile_task(name, first)
        second_task, _ = core.compile_task(name, second)
        first_selected, first_summary = select_states(first_task)
        second_selected, second_summary = select_states(second_task)
        rows.append({
            "shift": shift,
            "query_count": first_task.query_count,
            "first_selected": len(first_selected),
            "second_selected": len(second_selected),
            "first_candidates": first_summary["structural_candidates"],
            "second_candidates": second_summary["structural_candidates"],
            "equal": first_selected == second_selected,
        })
    mismatches = [row for row in rows if not row["equal"]]
    minimum_selected = min(
        min(int(row["first_selected"]), int(row["second_selected"]))
        for row in rows
    )
    return {
        "pair_count": len(rows),
        "mismatch_count": len(mismatches),
        "minimum_selected": minimum_selected,
        "all_equal": not mismatches,
        "rows": rows,
    }


def run_reference(states_path: Path, reference_path: Path):
    preregistration, v82_preregistration, manifest, evidence = load_inputs()
    prepare_opened(
        v82_preregistration,
        manifest,
        reference_path.parent / "adapted-manifest.json",
    )
    result = parent.opened.run(states_path, reference_path)
    result["status"] = "medium_small_query_coverage_python_reference_v83"
    result["parent_v82_evidence_digest"] = V82_EVIDENCE_DIGEST
    result["parent_v82_evidence_sha256"] = V82_EVIDENCE_SHA256
    result["parent_v68_evidence_digest"] = parent.LEGACY_PARENT_V68_EVIDENCE_DIGEST
    result["compiler_protocol"] = frontier.compiler_protocol()
    result["selector_protocol"] = protocol()["state_selector"]
    result["selected_pmlb_datasets"] = [
        str(row["name"]) for row in manifest["datasets"]
    ]
    result["frozen_nonzero_state_digest_check"] = frozen_state_digest_check(
        preregistration, result["dataset_summaries"]
    )
    result["label_independence_certificate"] = label_independence_certificate(
        manifest
    )
    result["v83_development_digest"] = canonical_digest({
        "parent_v82_evidence_digest": V82_EVIDENCE_DIGEST,
        "protocol": result["protocol"],
        "dataset_summaries": result["dataset_summaries"],
        "state_input_sha256": result["state_input_sha256"],
        "state_digests": [row["state_digest"] for row in result["rows"]],
        "frozen_state_check": result["frozen_nonzero_state_digest_check"],
        "label_certificate": result["label_independence_certificate"],
    })
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(reference_path: Path, rust_path: Path, output_path: Path):
    preregistration, _, manifest, evidence = load_inputs()
    core.PREREGISTRATION = PREREGISTRATION
    result = core.validate(reference_path, rust_path, output_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    state_check = reference["frozen_nonzero_state_digest_check"]
    label_check = reference["label_independence_certificate"]
    summaries = {
        str(row["task"]): row for row in reference["dataset_summaries"]
    }
    adapter_verified = (
        len(summaries) == len(manifest["datasets"])
        and all(
            row.get("raw_sha256_verified") is True
            and row.get("raw_bytes_verified") is True
            and int(row.get("record_count", -1)) > 0
            for row in summaries.values()
        )
    )
    gate_values = preregistration["development_gate"]
    extra_gate = (
        int(state_check["mismatch_count"])
        == int(gate_values["frozen_nonzero_state_set_digest_mismatches"])
        and int(label_check["mismatch_count"])
        == int(gate_values["label_independence_mismatches"])
        and int(label_check["minimum_selected"])
        >= int(gate_values["minimum_fallback_certificate_states"])
        and int(summaries[SOLAR_TASK]["selected_states"])
        >= int(gate_values["minimum_solar_flare_states"])
        and adapter_verified is bool(gate_values["adapter_verification_passed"])
    )
    base_gate = bool(result["development_gate"])
    gate = bool(base_gate and extra_gate)
    result["status"] = (
        "medium_small_query_coverage_development_pass_v83"
        if gate else "medium_small_query_coverage_development_rejected_v83"
    )
    result["development_gate"] = gate
    result["base_validator_gate"] = base_gate
    result["claim_scope"] = preregistration["claim_boundary"]
    result["frozen_nonzero_state_digest_check"] = state_check
    result["label_independence_certificate"] = label_check
    result["adapter_verification_passed"] = adapter_verified
    result["solar_flare_selected_states"] = int(
        summaries[SOLAR_TASK]["selected_states"]
    )
    result["parent_v82_evidence_digest"] = V82_EVIDENCE_DIGEST
    result["parent_v82_evidence_sha256"] = V82_EVIDENCE_SHA256
    result["parent_v82_status"] = evidence["status"]
    result["selector_protocol"] = protocol()["state_selector"]
    result["v83_development_digest"] = reference["v83_development_digest"]
    result["evidence_digest"] = canonical_digest(result)
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
            "solar_states": next(
                row["selected_states"] for row in result["dataset_summaries"]
                if row["task"] == SOLAR_TASK
            ),
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
        "frozen_state_mismatches": result[
            "frozen_nonzero_state_digest_check"
        ]["mismatch_count"],
        "label_independence_mismatches": result[
            "label_independence_certificate"
        ]["mismatch_count"],
        "solar_states": result["solar_flare_selected_states"],
        "adapter_verified": result["adapter_verification_passed"],
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


