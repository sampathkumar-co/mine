from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import select_holdouts as locked
from selector import (
    DATASET_REVISION,
    EXCLUSIONS,
    MAX_PER_FAMILY,
    MIN_FAMILIES,
    SEED,
    SOURCE_COMMIT,
    TASK_COUNT,
    classify,
    select_tasks,
    task_score,
)


def source_inventory_layout_v2() -> tuple[set[str], dict[str, object]]:
    payload, headers, raw = locked.fetch_json(locked.SOURCE_TREE_URL, "LEXIGEN-v4-name-only-selector-layout-v2")
    if not isinstance(payload, dict) or payload.get("truncated"):
        raise RuntimeError("source tree metadata is missing or truncated")
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError("source tree metadata lacks a tree list")

    tasks: set[str] = set()
    matched_paths: list[str] = []
    for entry in tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = str(entry.get("path", ""))
        parts = path.split("/")
        if (
            len(parts) == 3
            and parts[0] == "AlgoTuneTasks"
            and parts[1]
            and parts[2] == f"{parts[1]}.py"
        ):
            tasks.add(parts[1])
            matched_paths.append(path)
    if not tasks:
        raise RuntimeError("source tree contains no frozen-layout task names")

    evidence = {
        "url": locked.SOURCE_TREE_URL,
        "commit": SOURCE_COMMIT,
        "tree_sha": payload.get("sha"),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "etag": headers.get("etag"),
        "layout_rule": "AlgoTuneTasks/<task>/<task>.py",
        "matched_path_count": len(matched_paths),
        "matched_paths_sha256": hashlib.sha256("\n".join(sorted(matched_paths)).encode()).hexdigest(),
        "task_contents_opened": False,
    }
    return tasks, evidence


def main() -> None:
    source, source_evidence = source_inventory_layout_v2()
    dataset, dataset_evidence = locked.dataset_inventory()
    common = sorted((source & dataset) - EXCLUSIONS)
    rows = [{"task": task, "family": classify(task), "score": task_score(task)} for task in common]
    selected = select_tasks(rows)
    family_counts = Counter(row["family"] for row in selected)

    inventory_payload = {
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "eligible": sorted((row["task"], row["family"], row["score"]) for row in rows),
    }
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "selection_seed": SEED,
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "source_metadata": source_evidence,
        "dataset_metadata": dataset_evidence,
        "source_task_count": len(source),
        "dataset_task_count": len(dataset),
        "eligible_common_task_count": len(common),
        "inventory_sha256": hashlib.sha256(json.dumps(inventory_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "excluded_tasks": sorted(EXCLUSIONS),
        "task_count": TASK_COUNT,
        "minimum_distinct_families": MIN_FAMILIES,
        "maximum_per_family": MAX_PER_FAMILY,
        "selected": selected,
        "selected_family_counts": dict(sorted(family_counts.items())),
        "enumerator_amendment": {
            "classification": "infrastructure_only_frozen_snapshot_layout_fix",
            "original_run_id": 30372778744,
            "diagnostic_run_ids": [30373004021, 30373226269, 30373399055],
            "source_path_diagnostic_artifact": 8693791651,
            "source_layout_diagnostic_artifact": 8693867527,
            "scientific_rules_changed": False
        },
        "task_contents_opened": False,
        "reports_opened": False,
        "public_solvers_opened": False,
        "data_manifests_opened": False,
        "data_payloads_opened": False,
    }
    if len(selected) != TASK_COUNT or len(family_counts) < MIN_FAMILIES:
        raise RuntimeError("selection violates the frozen task-count or diversity gate")
    output = Path("selection-evidence-layout-v2")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "inventory_sha256": report["inventory_sha256"], "source_task_count": len(source), "dataset_task_count": len(dataset)}, indent=2))


if __name__ == "__main__":
    main()
