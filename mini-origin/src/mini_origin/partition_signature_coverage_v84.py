from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import conditioned_cell_frontier_v60 as conditioned
from . import label_free_frontier_v72 as frontier
from . import near_small_query_coverage_v83 as parent
from . import response_cost_pareto_v56 as response


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v84-partition-signature-coverage.json"
)

compact_state = parent.compact_state


def _structural_hash(*values: object) -> str:
    """Hash structural values only; dataset identity and labels are forbidden."""
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def partition_signature_groups(
    task: object, allowed: int, remaining: int
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Group separating queries by their exact truth partition on allowed rows."""
    groups: dict[tuple[int, ...], list[int]] = {}
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        signature = tuple(response.partition(task, allowed, query))
        if len(signature) > 1:
            groups.setdefault(signature, []).append(query)
    rows = [
        (signature, tuple(sorted(queries)))
        for signature, queries in groups.items()
    ]
    rows.sort(key=lambda row: _structural_hash(row[0], row[1]))
    return tuple(rows)


def complete_class_masks(
    task: object,
    allowed: int,
    remaining: int,
    minimum_raw: int,
    minimum_redundancy: int,
) -> tuple[tuple[int, int], ...]:
    """Build deterministic masks containing whole signature-equivalence classes.

    The full class set is attempted first. Leave-one-class-out variants are then
    considered in structural order. This is deliberately bounded and introduces
    no new tuning parameter or solver-derived ranking.
    """
    groups = partition_signature_groups(task, allowed, remaining)
    if not groups:
        return ()

    selections = [groups]
    selections.extend(groups[:index] + groups[index + 1 :] for index in range(len(groups)))
    candidates: dict[int, int] = {}
    for selected in selections:
        representatives = len(selected)
        if not (
            conditioned.MIN_PARTITION_CLASSES
            <= representatives
            <= conditioned.MAX_PARTITION_CLASSES
        ):
            continue
        mask = 0
        for _, queries in selected:
            for query in queries:
                mask |= 1 << query
        raw = mask.bit_count()
        if not (minimum_raw <= raw <= conditioned.MAX_RAW_QUERIES):
            continue
        if raw - representatives < minimum_redundancy:
            continue
        candidates[mask] = representatives

    row_count = task.full_mask.bit_count()
    return tuple(
        sorted(
            candidates.items(),
            key=lambda row: (
                -(row[0].bit_count() - row[1]),
                -row[0].bit_count(),
                _structural_hash(row_count, task.query_count, allowed, row[0], row[1]),
            ),
        )
    )


def _signature_fallback_states(task: object):
    minimum_raw, minimum_redundancy = parent.parent.effective_limits(task)
    candidates: dict[tuple[int, int], int] = {}
    cells = conditioned.conditioned_cells(task)
    for cell, path_remaining, path in cells:
        cell_size = cell.bit_count()
        allowed_variants = []
        if 8 <= cell_size <= 24:
            allowed_variants.append(cell)
        for size in parent.parent.SMALL_QUERY_SAMPLE_SIZES:
            if cell_size < size:
                continue
            for seed in conditioned.PATH_SEEDS[: parent.parent.SMALL_QUERY_ALLOWED_SEEDS]:
                allowed_variants.append(
                    conditioned.sample_allowed(task, cell, size, f"{path}:{seed}:{size}")
                )
        for allowed in sorted(set(allowed_variants)):
            for remaining, representatives in complete_class_masks(
                task,
                allowed,
                path_remaining,
                minimum_raw,
                minimum_redundancy,
            ):
                candidates[(allowed, remaining)] = representatives

    rows = [
        (allowed, remaining, representatives)
        for (allowed, remaining), representatives in candidates.items()
    ]
    row_count = task.full_mask.bit_count()
    rows.sort(
        key=lambda row: (
            -row[0].bit_count(),
            -(row[1].bit_count() - row[2]),
            -row[1].bit_count(),
            _structural_hash(row_count, task.query_count, row[0], row[1], row[2]),
        )
    )
    rows = rows[: conditioned.MAX_STATES_PER_TASK]
    return rows, {
        "conditioned_cells": len(cells),
        "structural_candidates": len(candidates),
        "selected_states": len(rows),
        "partition_signature_fallback": True,
        "effective_min_raw_queries": minimum_raw,
        "effective_min_redundancy": minimum_redundancy,
        "selected_state_set_digest": parent.state_set_digest(task, rows),
        "selector_revision": "partition-signature-coverage-v84",
    }


def select_states(task: object):
    """Preserve every contributing v0.83 state set; repair only zero-candidate cases."""
    rows, summary = parent.adaptive_select_states(task)
    if rows:
        result = dict(summary)
        result["partition_signature_fallback"] = False
        result["selector_revision"] = "partition-signature-coverage-v84"
        return rows, result
    return _signature_fallback_states(task)


def protocol() -> dict[str, object]:
    result = dict(parent.protocol())
    result["state_selector"] = (
        "v0.83 selector unchanged for contributing tasks; deterministic complete "
        "partition-signature classes are used only when v0.83 yields zero states"
    )
    result["partition_signature_fallback"] = "zero-candidate-only"
    result["development_data_status"] = "opened"
    return result


def install_v84_components() -> None:
    parent.configure_module()
    frontier.protocol = protocol
    conditioned.select_states = select_states


def configure_module() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "preregistered_before_implementation_or_evaluation":
        raise RuntimeError("v0.84 preregistration status changed")
    install_v84_components()
    frontier.configure_module = install_v84_components
