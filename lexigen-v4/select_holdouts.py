from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

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

SOURCE_TREE_URL = f"https://api.github.com/repos/oripress/AlgoTune/git/trees/{SOURCE_COMMIT}?recursive=1"
DATASET_TREE_URL = (
    "https://huggingface.co/api/datasets/oripress/AlgoTune/tree/"
    f"{DATASET_REVISION}/data?recursive=false&expand=false&limit=1000"
)


def fetch_json(url: str, user_agent: str) -> tuple[object, dict[str, str], bytes]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/vnd.github+json, application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        raw = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
    return json.loads(raw), headers, raw


def source_inventory() -> tuple[set[str], dict[str, object]]:
    payload, headers, raw = fetch_json(SOURCE_TREE_URL, "LEXIGEN-v4-name-only-selector")
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
        if len(parts) == 3 and parts[0] == "AlgoTuneTasks" and parts[2] == "task.py":
            tasks.add(parts[1])
            matched_paths.append(path)
    if not tasks:
        raise RuntimeError("source tree contains no task names")
    evidence = {
        "url": SOURCE_TREE_URL,
        "commit": SOURCE_COMMIT,
        "tree_sha": payload.get("sha"),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "etag": headers.get("etag"),
        "matched_path_count": len(matched_paths),
        "matched_paths_sha256": hashlib.sha256("\n".join(sorted(matched_paths)).encode()).hexdigest(),
        "task_contents_opened": False,
    }
    return tasks, evidence


def dataset_inventory() -> tuple[set[str], dict[str, object]]:
    payload, headers, raw = fetch_json(DATASET_TREE_URL, "LEXIGEN-v4-name-only-selector")
    if not isinstance(payload, list):
        raise RuntimeError("dataset tree metadata is not a list")
    tasks: set[str] = set()
    directory_rows: list[tuple[str, str]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", ""))
        entry_type = str(entry.get("type", ""))
        if entry_type == "directory" and path.startswith("data/"):
            name = path.split("/", 1)[1]
            if name and "/" not in name:
                tasks.add(name)
                directory_rows.append((name, str(entry.get("oid", ""))))
    if not tasks:
        raise RuntimeError("dataset tree contains no task directories")
    evidence = {
        "url": DATASET_TREE_URL,
        "revision": DATASET_REVISION,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "etag": headers.get("etag"),
        "directory_count": len(directory_rows),
        "directory_identity_sha256": hashlib.sha256(json.dumps(sorted(directory_rows), separators=(",", ":")).encode()).hexdigest(),
        "manifest_contents_opened": False,
        "payloads_opened": False,
    }
    return tasks, evidence


def main() -> None:
    source, source_evidence = source_inventory()
    dataset, dataset_evidence = dataset_inventory()
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
        "task_contents_opened": False,
        "reports_opened": False,
        "public_solvers_opened": False,
        "data_manifests_opened": False,
        "data_payloads_opened": False,
    }
    if len(selected) != TASK_COUNT or len(family_counts) < MIN_FAMILIES:
        raise RuntimeError("selection violates the frozen task-count or diversity gate")
    output = Path("selection-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "inventory_sha256": report["inventory_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
