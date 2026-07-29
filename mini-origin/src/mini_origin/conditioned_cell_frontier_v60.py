from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import external_response_cost_v58 as external
from . import response_cost_pareto_v56 as response


PROFILE_SEEDS = (6001, 6002, 6003)
PATH_SEEDS = tuple(range(60101, 60113))
MAX_DEPTH = 3
MAX_QUERY_CHOICES = 6
MAX_CELLS_PER_DEPTH = 96
SAMPLE_SIZES = (24, 20, 16, 12)
MAX_STATES_PER_TASK = 12
MIN_PARTITION_CLASSES = 6
MAX_PARTITION_CLASSES = 16
MIN_RAW_QUERIES = 10
MAX_RAW_QUERIES = 64
MIN_REDUNDANCY = 4
BUDGET_LADDER = (10_000, 50_000, 250_000, 500_000)


def hash_token(*values: object) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def separating_queries(task: object, allowed: int, remaining: int) -> list[int]:
    result = []
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        if len(response.partition(task, allowed, query)) > 1:
            result.append(query)
    return result


def conditioned_cells(task: object) -> list[tuple[int, int, str]]:
    initial_remaining = (1 << task.query_count) - 1
    frontier = [(task.full_mask, initial_remaining, "root")]
    collected: dict[tuple[int, int], str] = {}
    for depth in range(1, MAX_DEPTH + 1):
        next_rows: dict[tuple[int, int], str] = {}
        for allowed, remaining, path in frontier:
            queries = separating_queries(task, allowed, remaining)
            queries.sort(key=lambda query: hash_token(task.name, depth, path, query))
            for query in queries[:MAX_QUERY_CHOICES]:
                next_remaining = remaining & ~(1 << query)
                children = response.partition(task, allowed, query)
                for child_index, child in enumerate(children):
                    if child.bit_count() < 8:
                        continue
                    child_path = f"{path}/q{query}/c{child_index}"
                    key = child, next_remaining
                    previous = next_rows.get(key)
                    if previous is None or child_path < previous:
                        next_rows[key] = child_path
                    collected.setdefault(key, child_path)
        ranked = sorted(
            ((allowed, remaining, path) for (allowed, remaining), path in next_rows.items()),
            key=lambda row: hash_token(task.name, depth, row[0], row[1], row[2]),
        )
        frontier = ranked[:MAX_CELLS_PER_DEPTH]
    return [
        (allowed, remaining, path)
        for (allowed, remaining), path in collected.items()
    ]


def sample_allowed(task: object, cell: int, size: int, salt: str) -> int:
    indices = []
    pending = cell
    while pending:
        bit = pending & -pending
        index = bit.bit_length() - 1
        pending ^= bit
        indices.append(index)
    indices.sort(
        key=lambda index: hash_token(
            task.name, salt, index, task.rows[index], task.labels[index]
        )
    )
    mask = 0
    for index in indices[:size]:
        mask |= 1 << index
    return mask


def query_groups(task: object, allowed: int, remaining: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    pending = remaining
    while pending:
        bit = pending & -pending
        query = bit.bit_length() - 1
        pending ^= bit
        signature = response.partition(task, allowed, query)
        if len(signature) > 1:
            groups.setdefault(signature, []).append(query)
    return [
        (signature, tuple(sorted(queries)))
        for signature, queries in groups.items()
    ]


def choose_remaining(task: object, allowed: int, remaining: int, salt: str) -> tuple[int, int]:
    groups = query_groups(task, allowed, remaining)
    groups.sort(key=lambda row: (
        -len(row[1]),
        hash_token(task.name, salt, row[0], row[1]),
    ))
    selected = []
    raw = 0
    for signature, queries in groups:
        if len(selected) >= MAX_PARTITION_CLASSES:
            break
        if raw + len(queries) > MAX_RAW_QUERIES:
            continue
        selected.append((signature, queries))
        raw += len(queries)
    mask = 0
    for _, queries in selected:
        for query in queries:
            mask |= 1 << query
    return mask, len(selected)


def structural_rank(task_name: str, allowed: int, remaining: int, representatives: int) -> tuple[int, int, int, str]:
    raw = remaining.bit_count()
    return (
        -allowed.bit_count(),
        -(raw - representatives),
        -raw,
        hash_token("v60", task_name, allowed, remaining),
    )


def select_states(task: object) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    candidates: dict[tuple[int, int], int] = {}
    cells = conditioned_cells(task)
    for cell, path_remaining, path in cells:
        cell_size = cell.bit_count()
        allowed_variants = []
        if 8 <= cell_size <= 24:
            allowed_variants.append(cell)
        for size in SAMPLE_SIZES:
            if cell_size >= size:
                for seed in PATH_SEEDS[:3]:
                    allowed_variants.append(
                        sample_allowed(task, cell, size, f"{path}:{seed}:{size}")
                    )
        for allowed in sorted(set(allowed_variants)):
            for seed in PATH_SEEDS[:4]:
                remaining, representatives = choose_remaining(
                    task, allowed, path_remaining, f"{path}:{seed}"
                )
                raw = remaining.bit_count()
                if (
                    MIN_PARTITION_CLASSES <= representatives <= MAX_PARTITION_CLASSES
                    and MIN_RAW_QUERIES <= raw <= MAX_RAW_QUERIES
                    and raw - representatives >= MIN_REDUNDANCY
                ):
                    candidates[(allowed, remaining)] = representatives
    rows = [
        (allowed, remaining, representatives)
        for (allowed, remaining), representatives in candidates.items()
    ]
    rows.sort(key=lambda row: structural_rank(task.name, row[0], row[1], row[2]))
    return rows[:MAX_STATES_PER_TASK], {
        "conditioned_cells": len(cells),
        "structural_candidates": len(rows),
        "selected_states": min(len(rows), MAX_STATES_PER_TASK),
    }


def load_tasks() -> tuple[list[tuple[object, list[tuple[int, int, int]]]], list[dict[str, object]]]:
    manifest = json.loads(external.MANIFEST.read_text(encoding="utf-8"))
    tasks = []
    summaries = []
    for dataset in manifest["datasets"]:
        payload = external.download(str(dataset["url"]))
        if hashlib.sha256(payload).hexdigest() != dataset["sha256"]:
            raise RuntimeError(f"archive mismatch for {dataset['name']}")
        records = external.parse_records(str(dataset["name"]), payload)
        task, summary = external.task_from_records(str(dataset["name"]), records)
        selected, selection = select_states(task)
        summary.update(selection)
        summary["task"] = task.name
        summaries.append(summary)
        tasks.append((task, selected))
    return tasks, summaries


def run() -> dict[str, object]:
    tasks, summaries = load_tasks()
    rows = []
    base_states = set()
    for task, selected in tasks:
        for allowed, remaining, representatives in selected:
            base_digest = hash_token("v60", task.name, allowed, remaining)
            base_states.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile = response.profile_for_task(task, seed)
                row = response.evaluate_state(task, profile, allowed, remaining)
                row.update({
                    "task": task.name,
                    "base_state_digest": base_digest,
                    "profile_seed": seed,
                    "structural_partition_representatives": representatives,
                    "state_digest": hash_token(base_digest, seed, "conditioned-v60"),
                })
                rows.append(row)
    solved = [row for row in rows if row["pareto_solved"]]
    both = [row for row in solved if row["plain_solved"]]
    pareto_only = [row for row in solved if not row["plain_solved"]]
    ratios = [float(row["expansion_ratio_lower_bound"]) for row in solved]
    ladder = {
        str(budget): {
            key: sum(int(row["budget_ladder"][str(budget)][key]) for row in rows)
            for key in ("pareto_solved", "plain_solved")
        }
        for budget in BUDGET_LADDER
    }
    contributing = sum(int(summary["selected_states"] > 0) for summary in summaries)
    dominated = sum(
        int(row["pareto_stats"]["dominated_queries_removed"])
        for row in solved
    )
    incomparable = sum(
        int(row["root_pareto_certificate"]["incomparable_pareto_classes"])
        for row in rows
    )
    gate = (
        contributing >= 5
        and len(base_states) >= 50
        and len(rows) >= 150
        and len(solved) >= int(0.9 * len(rows))
        and len(both) >= 40
        and all(row["matched_if_both"] for row in both)
        and len(pareto_only) >= 25
        and dominated >= 1000
        and incomparable > 0
        and ratios
        and float(np.median(ratios)) >= 10.0
        and float(np.quantile(ratios, 0.9)) >= 30.0
        and ladder["50000"]["pareto_solved"] >= (
            ladder["50000"]["plain_solved"] + 20
        )
    )
    payload = {
        "status": (
            "conditioned_cell_frontier_pass"
            if gate else "conditioned_cell_frontier_rejected"
        ),
        "development_gate": gate,
        "claim_scope": (
            "Solver-independent conditioning paths are used to create descendant-like "
            "states on already opened external datasets. A pass freezes a benchmark "
            "generator for a separate blind suite; it is not external confirmation."
        ),
        "protocol": {
            "profile_seeds": list(PROFILE_SEEDS),
            "path_seeds": list(PATH_SEEDS),
            "maximum_depth": MAX_DEPTH,
            "maximum_query_choices": MAX_QUERY_CHOICES,
            "maximum_cells_per_depth": MAX_CELLS_PER_DEPTH,
            "sample_sizes": list(SAMPLE_SIZES),
            "max_states_per_task": MAX_STATES_PER_TASK,
            "partition_class_range": [MIN_PARTITION_CLASSES, MAX_PARTITION_CLASSES],
            "raw_query_range": [MIN_RAW_QUERIES, MAX_RAW_QUERIES],
            "minimum_redundancy": MIN_REDUNDANCY,
            "budget": response.BUDGET,
            "budget_ladder": list(BUDGET_LADDER),
        },
        "dataset_summaries": summaries,
        "contributing_dataset_count": contributing,
        "base_state_count": len(base_states),
        "profiled_state_count": len(rows),
        "pareto_solved_count": len(solved),
        "both_solved_count": len(both),
        "pareto_only_count": len(pareto_only),
        "plan_mismatch_count": sum(int(not row["matched_if_both"]) for row in both),
        "dominated_queries_removed": dominated,
        "root_incomparable_classes": incomparable,
        "expansion_ratio_median": float(np.median(ratios)) if ratios else None,
        "expansion_ratio_p90": float(np.quantile(ratios, 0.9)) if ratios else None,
        "budget_ladder_summary": ladder,
        "rows": rows,
    }
    payload["frozen_frontier_digest"] = hashlib.sha256(json.dumps({
        "protocol": payload["protocol"],
        "dataset_summaries": summaries,
        "state_digests": [row["state_digest"] for row in rows],
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "gate": result["development_gate"],
        "datasets": result["contributing_dataset_count"],
        "base_states": result["base_state_count"],
        "profiled_states": result["profiled_state_count"],
        "pareto_solved": result["pareto_solved_count"],
        "plain_solved": result["both_solved_count"],
        "pareto_only": result["pareto_only_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
