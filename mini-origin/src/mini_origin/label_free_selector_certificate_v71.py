from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import conditioned_cell_frontier_v60 as conditioned
from . import numeric_threshold_repaired_v70 as numeric


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v71-label-free-selector-certificate.json"
)
TASK_COUNT = 64
SEED_START = 71_001
RECORDS_PER_TASK = 420
FEATURES_PER_TASK = 18
_ORIGINAL_SAMPLE_ALLOWED = conditioned.sample_allowed


def hash_token(*values: object) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def label_free_sample_allowed(task: object, cell: int, size: int, salt: str) -> int:
    indices = []
    pending = cell
    while pending:
        bit = pending & -pending
        index = bit.bit_length() - 1
        pending ^= bit
        indices.append(index)
    indices.sort(
        key=lambda index: hash_token(
            task.name,
            salt,
            index,
            task.rows[index],
        )
    )
    mask = 0
    for index in indices[:size]:
        mask |= 1 << index
    return mask


def configure_selector() -> None:
    conditioned.sample_allowed = label_free_sample_allowed


def synthetic_records(seed: int, relabel: bool):
    rows = []
    for index in range(RECORDS_PER_TASK):
        features = []
        for feature in range(FEATURES_PER_TASK):
            # Mixed continuous and low-cardinality values.  Feature values are
            # entirely independent of labels and deterministic from seed/index.
            if feature % 5 == 0:
                value = str((index * (feature + 3) + seed) % 7)
            elif feature % 5 == 1:
                value = f"{((index * (feature + 11) + seed * 3) % 10007) / 97:.6f}"
            elif feature % 5 == 2:
                value = f"{((index * index + seed + feature * 13) % 20011) / 113:.6f}"
            elif feature % 5 == 3:
                value = "?" if (index + seed + feature) % 53 == 0 else f"{((index * 17 + seed * 5 + feature) % 30011) / 131:.6f}"
            else:
                value = chr(ord("A") + ((index + feature + seed) % 4))
            features.append(value)
        if relabel:
            label = f"R{(index * 7 + seed) % 5}"
        else:
            label = f"L{(index + seed) % 3}"
        rows.append((tuple(features), label))
    return rows


def selected_signature(task: object):
    selected, summary = conditioned.select_states(task)
    return tuple(selected), summary


def run() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    configure_selector()
    rows = []
    sampled_mismatches = []
    response_mismatches = []
    query_count_mismatches = []
    cell_mismatches = []
    selected_mismatches = []
    oversized_pairs = 0
    selected_pairs = 0

    for seed in range(SEED_START, SEED_START + TASK_COUNT):
        name = f"v71-label-free-{seed}"
        first_records = synthetic_records(seed, False)
        second_records = synthetic_records(seed, True)

        first_sample = numeric.label_free_sample(name, first_records)
        second_sample = numeric.label_free_sample(name, second_records)
        first_features = tuple(features for features, _ in first_sample)
        second_features = tuple(features for features, _ in second_sample)
        sampled_equal = first_features == second_features
        if not sampled_equal:
            sampled_mismatches.append(seed)

        first_task, first_summary = numeric.compile_task(name, first_records)
        second_task, second_summary = numeric.compile_task(name, second_records)
        responses_equal = first_task.rows == second_task.rows
        queries_equal = first_task.query_count == second_task.query_count
        if not responses_equal:
            response_mismatches.append(seed)
        if not queries_equal:
            query_count_mismatches.append(seed)

        first_cells = tuple(sorted(conditioned.conditioned_cells(first_task)))
        second_cells = tuple(sorted(conditioned.conditioned_cells(second_task)))
        cells_equal = first_cells == second_cells
        if not cells_equal:
            cell_mismatches.append(seed)
        has_oversized = any(cell.bit_count() > 24 for cell, _, _ in first_cells)
        oversized_pairs += int(has_oversized)

        first_selected, first_selection = selected_signature(first_task)
        second_selected, second_selection = selected_signature(second_task)
        selected_equal = (
            first_selected == second_selected
            and first_selection == second_selection
        )
        if not selected_equal:
            selected_mismatches.append(seed)
        has_selected = bool(first_selected)
        selected_pairs += int(has_selected)

        rows.append({
            "seed": seed,
            "sampled_features_equal": sampled_equal,
            "compiled_responses_equal": responses_equal,
            "compiled_query_counts_equal": queries_equal,
            "conditioned_cells_equal": cells_equal,
            "selected_states_equal": selected_equal,
            "oversized_cells_present": has_oversized,
            "selected_states_present": has_selected,
            "compiled_queries": first_task.query_count,
            "conditioned_cells": len(first_cells),
            "selected_states": len(first_selected),
            "first_label_count": first_summary["labels"],
            "second_label_count": second_summary["labels"],
        })

    gate_values = preregistration["locked_gate"]
    gate = (
        len(rows) == int(gate_values["task_pairs"])
        and len(sampled_mismatches)
        == int(gate_values["sampled_feature_table_mismatches"])
        and len(response_mismatches)
        == int(gate_values["compiled_response_table_mismatches"])
        and len(query_count_mismatches)
        == int(gate_values["compiled_query_count_mismatches"])
        and len(cell_mismatches)
        == int(gate_values["conditioned_cell_mismatches"])
        and len(selected_mismatches)
        == int(gate_values["selected_state_mismatches"])
        and oversized_pairs
        >= int(gate_values["minimum_pairs_with_oversized_cells"])
        and selected_pairs
        >= int(gate_values["minimum_pairs_with_selected_states"])
    )
    result = {
        "status": (
            "label_free_selector_certificate_pass"
            if gate else "label_free_selector_certificate_rejected"
        ),
        "development_gate": gate,
        "claim_scope": preregistration["claim_boundary"],
        "protocol": preregistration["protocol"],
        "task_pair_count": len(rows),
        "sampled_feature_table_mismatch_count": len(sampled_mismatches),
        "compiled_response_table_mismatch_count": len(response_mismatches),
        "compiled_query_count_mismatch_count": len(query_count_mismatches),
        "conditioned_cell_mismatch_count": len(cell_mismatches),
        "selected_state_mismatch_count": len(selected_mismatches),
        "pairs_with_oversized_cells": oversized_pairs,
        "pairs_with_selected_states": selected_pairs,
        "mismatches": {
            "sampled_features": sampled_mismatches,
            "compiled_responses": response_mismatches,
            "compiled_query_counts": query_count_mismatches,
            "conditioned_cells": cell_mismatches,
            "selected_states": selected_mismatches,
        },
        "rows": rows,
    }
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "task_pairs": result["task_pair_count"],
        "oversized_pairs": result["pairs_with_oversized_cells"],
        "selected_pairs": result["pairs_with_selected_states"],
        "mismatches": sum((
            result["sampled_feature_table_mismatch_count"],
            result["compiled_response_table_mismatch_count"],
            result["compiled_query_count_mismatch_count"],
            result["conditioned_cell_mismatch_count"],
            result["selected_state_mismatch_count"],
        )),
    }, indent=2))
    if not result["development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
