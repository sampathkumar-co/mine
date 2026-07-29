from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import external_response_cost_v58 as external
from . import response_cost_pareto_v56 as response


PROFILE_SEEDS = (5901, 5902, 5903)
SUBSET_SEEDS = tuple(range(59101, 59125))
CANDIDATE_SIZES = (12, 16, 20, 24)
MAX_STATES_PER_TASK = 12
MIN_PARTITION_CLASSES = 8
MAX_PARTITION_CLASSES = 16
MIN_RAW_QUERIES = 18
MAX_RAW_QUERIES = 64
MIN_REDUNDANCY = 6
BUDGET_LADDER = (10_000, 50_000, 250_000, 500_000)


def candidate_rank(task: object, seed: int, index: int) -> str:
    return hashlib.sha256(
        json.dumps(
            [task.name, seed, index, task.rows[index], task.labels[index]],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def choose_allowed(task: object, size: int, seed: int) -> int:
    ranked = sorted(
        range(task.candidate_count),
        key=lambda index: candidate_rank(task, seed, index),
    )
    mask = 0
    for index in ranked[: min(size, len(ranked))]:
        mask |= 1 << index
    return mask


def group_queries(task: object, allowed: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for query in range(task.query_count):
        signature = response.partition(task, allowed, query)
        if len(signature) > 1:
            groups.setdefault(signature, []).append(query)
    return [
        (signature, tuple(sorted(queries)))
        for signature, queries in groups.items()
    ]


def class_rank(
    task_name: str,
    seed: int,
    signature: tuple[int, ...],
    queries: tuple[int, ...],
) -> tuple[int, str]:
    digest = hashlib.sha256(
        f"{task_name}:{seed}:{signature}:{queries}".encode("utf-8")
    ).hexdigest()
    return -len(queries), digest


def choose_remaining(task: object, allowed: int, seed: int) -> tuple[int, int]:
    groups = group_queries(task, allowed)
    groups.sort(key=lambda row: class_rank(task.name, seed, row[0], row[1]))
    selected: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    raw = 0
    for signature, queries in groups:
        if len(selected) >= MAX_PARTITION_CLASSES:
            break
        if raw + len(queries) > MAX_RAW_QUERIES:
            continue
        selected.append((signature, queries))
        raw += len(queries)
    remaining = 0
    for _, queries in selected:
        for query in queries:
            remaining |= 1 << query
    return remaining, len(selected)


def structural_rank(
    task_name: str,
    allowed: int,
    remaining: int,
    representatives: int,
) -> tuple[int, int, int, str]:
    raw = remaining.bit_count()
    return (
        -allowed.bit_count(),
        -(raw - representatives),
        -raw,
        hashlib.sha256(
            f"v59:{task_name}:{allowed}:{remaining}".encode("utf-8")
        ).hexdigest(),
    )


def select_states(task: object) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    candidates: dict[tuple[int, int], int] = {}
    for size in CANDIDATE_SIZES:
        if task.candidate_count < size:
            continue
        for seed in SUBSET_SEEDS:
            allowed = choose_allowed(task, size, seed)
            remaining, representatives = choose_remaining(task, allowed, seed)
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
    rows.sort(
        key=lambda row: structural_rank(task.name, row[0], row[1], row[2])
    )
    return rows[:MAX_STATES_PER_TASK], {
        "structural_candidates": len(rows),
        "selected_states": min(len(rows), MAX_STATES_PER_TASK),
    }


def load_opened_v58_tasks() -> tuple[list[object], list[dict[str, object]]]:
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
    tasks, summaries = load_opened_v58_tasks()
    rows = []
    base_states = set()
    for task, selected in tasks:
        for allowed, remaining, representatives in selected:
            base_digest = hashlib.sha256(
                f"v59:{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            base_states.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile = response.profile_for_task(task, seed)
                row = response.evaluate_state(task, profile, allowed, remaining)
                row.update({
                    "task": task.name,
                    "base_state_digest": base_digest,
                    "profile_seed": seed,
                    "structural_partition_representatives": representatives,
                    "state_digest": hashlib.sha256(
                        f"{base_digest}:{seed}:structural-v59".encode("utf-8")
                    ).hexdigest(),
                })
                rows.append(row)
    solved = [row for row in rows if row["pareto_solved"]]
    both = [row for row in solved if row["plain_solved"]]
    pareto_only = [row for row in solved if not row["plain_solved"]]
    ratios = [float(row["expansion_ratio_lower_bound"]) for row in solved]
    ladder = {
        str(budget): {
            key: sum(
                int(row["budget_ladder"][str(budget)][key])
                for row in rows
            )
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
        contributing >= 6
        and len(base_states) >= 60
        and len(rows) >= 180
        and len(solved) >= int(0.9 * len(rows))
        and len(both) >= 60
        and all(row["matched_if_both"] for row in both)
        and len(pareto_only) >= 30
        and dominated >= 1000
        and incomparable > 0
        and ratios
        and float(np.median(ratios)) >= 5.0
        and float(np.quantile(ratios, 0.9)) >= 20.0
        and ladder["50000"]["pareto_solved"] >= (
            ladder["50000"]["plain_solved"] + 25
        )
    )
    payload = {
        "status": (
            "structural_subtable_frontier_pass"
            if gate else "structural_subtable_frontier_rejected"
        ),
        "development_gate": gate,
        "claim_scope": (
            "A solver-independent structural subtable generator is developed on the "
            "already opened v0.58 datasets. It may justify a fresh blind external gate, "
            "but is not itself external confirmation or a world-class claim."
        ),
        "protocol": {
            "profile_seeds": list(PROFILE_SEEDS),
            "subset_seeds": list(SUBSET_SEEDS),
            "candidate_sizes": list(CANDIDATE_SIZES),
            "max_states_per_task": MAX_STATES_PER_TASK,
            "partition_class_range": [
                MIN_PARTITION_CLASSES, MAX_PARTITION_CLASSES
            ],
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
