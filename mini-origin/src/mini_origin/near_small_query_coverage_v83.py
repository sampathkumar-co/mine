from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import conditioned_cell_frontier_v60 as conditioned
from . import label_free_frontier_v72 as frontier
from . import small_query_coverage_v79 as parent


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v83-near-small-query-coverage.json"
)
NEAR_SMALL_QUERY_LIMIT = 12


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def state_set_digest(task: object, rows: list[tuple[int, int, int]]) -> str:
    digests = sorted(
        hashlib.sha256(
            f"v68:{task.name}:{allowed}:{remaining}".encode("utf-8")
        ).hexdigest()
        for allowed, remaining, _ in rows
    )
    return canonical_digest(digests)


def adaptive_select_states(task: object):
    """Extend only the generic v0.79 adaptive boundary from 10 to 12 queries."""
    if task.query_count <= parent.SMALL_QUERY_LIMIT:
        return parent.adaptive_select_states(task)
    if task.query_count > NEAR_SMALL_QUERY_LIMIT:
        rows, summary = parent._PARENT_SELECT_STATES(task)
        summary.update({
            "adaptive_small_query_mode": False,
            "effective_min_raw_queries": conditioned.MIN_RAW_QUERIES,
            "effective_min_redundancy": conditioned.MIN_REDUNDANCY,
            "selected_state_set_digest": state_set_digest(task, rows),
            "selector_revision": "near-small-query-coverage-v83",
        })
        return rows, summary

    minimum_raw, minimum_redundancy = parent.effective_limits(task)
    candidates: dict[tuple[int, int], int] = {}
    cells = conditioned.conditioned_cells(task)
    for cell, path_remaining, path in cells:
        cell_size = cell.bit_count()
        allowed_variants = []
        if 8 <= cell_size <= 24:
            allowed_variants.append(cell)
        for size in parent.SMALL_QUERY_SAMPLE_SIZES:
            if cell_size < size:
                continue
            for seed in conditioned.PATH_SEEDS[: parent.SMALL_QUERY_ALLOWED_SEEDS]:
                allowed_variants.append(
                    conditioned.sample_allowed(
                        task, cell, size, f"{path}:{seed}:{size}"
                    )
                )
        for allowed in sorted(set(allowed_variants)):
            for seed in conditioned.PATH_SEEDS[: parent.SMALL_QUERY_REMAINING_SEEDS]:
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
    rows = rows[: conditioned.MAX_STATES_PER_TASK]
    return rows, {
        "conditioned_cells": len(cells),
        "structural_candidates": len(candidates),
        "selected_states": len(rows),
        "adaptive_small_query_mode": True,
        "effective_min_raw_queries": minimum_raw,
        "effective_min_redundancy": minimum_redundancy,
        "allowed_seed_count": parent.SMALL_QUERY_ALLOWED_SEEDS,
        "remaining_seed_count": parent.SMALL_QUERY_REMAINING_SEEDS,
        "sample_sizes": list(parent.SMALL_QUERY_SAMPLE_SIZES),
        "selected_state_set_digest": state_set_digest(task, rows),
        "selector_revision": "near-small-query-coverage-v83",
    }


def protocol() -> dict[str, object]:
    result = dict(parent.protocol())
    result["state_selector"] = (
        "v0.79 selector unchanged except that its generic query-count-aware "
        "adaptive path applies to tasks with 12 or fewer compiled queries"
    )
    result["small_query_limit"] = NEAR_SMALL_QUERY_LIMIT
    result["development_data_status"] = "opened"
    return result


def install_v83_components() -> None:
    parent.configure_module()
    frontier.protocol = protocol
    conditioned.select_states = adaptive_select_states


def configure_module() -> None:
    install_v83_components()
    frontier.configure_module = install_v83_components
