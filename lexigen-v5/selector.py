from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

SOURCE_COMMIT = "dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"
DATASET_REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
SEED = "LEXIGEN-V5-CAUSAL-TRANSFER-2026-08-23-A"
TASK_COUNT = 10
MIN_FAMILIES = 8
MAX_PER_FAMILY = 2
SOURCE_TREE_URL = f"https://api.github.com/repos/oripress/AlgoTune/git/trees/{SOURCE_COMMIT}?recursive=1"
DATASET_TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{DATASET_REVISION}/data?recursive=false&expand=false&limit=1000"

FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cryptography_encoding", (r"cipher", r"encrypt", r"decrypt", r"hash", r"sha", r"base\d+", r"codec", r"encoding", r"compression")),
    ("linear_algebra", (r"matrix", r"svd", r"eigen", r"cholesky", r"qr", r"least_squares", r"linear_system", r"tensor", r"product")),
    ("numerical_optimization", (r"optimi", r"projection", r"simplex", r"portfolio", r"resource", r"flow", r"assignment", r"transport", r"cvar")),
    ("graph_discrete", (r"graph", r"shortest", r"path", r"tree", r"clique", r"color", r"matching", r"articulation", r"network", r"mst", r"cycle")),
    ("combinatorial", (r"cover", r"subset", r"set_", r"knapsack", r"permutation", r"combination", r"partition", r"scheduling", r"integer", r"factor")),
    ("signal_processing", (r"fft", r"fourier", r"filter", r"signal", r"convolution", r"correlation", r"wavelet", r"spectral", r"audio", r"image", r"dst", r"dct")),
    ("statistics", (r"stat", r"mean", r"median", r"quantile", r"regression", r"probab", r"distribution", r"sampling", r"variance", r"entropy")),
    ("scientific_computing", (r"integrat", r"differential", r"ode", r"pde", r"physics", r"simulation", r"scientific", r"nbody", r"monte_carlo")),
    ("string_sequence", (r"string", r"sequence", r"edit_distance", r"alignment", r"substring", r"prefix", r"suffix", r"token")),
    ("geometry", (r"geometry", r"convex_hull", r"distance", r"point", r"polygon", r"mesh", r"spatial")),
    ("machine_learning", (r"cluster", r"classification", r"neural", r"learning", r"pca", r"embedding", r"nearest", r"kmeans")),
)


def fetch_json(url: str, user_agent: str) -> tuple[object, dict[str, str], bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/vnd.github+json, application/json"})
    with urllib.request.urlopen(request, timeout=240) as response:
        raw = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
    return json.loads(raw), headers, raw


def classify(task_name: str) -> str:
    lowered = task_name.lower()
    for family, patterns in FAMILY_RULES:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return family
    return "miscellaneous"


def task_score(task_name: str) -> str:
    return hashlib.sha256(f"{SEED}\0{task_name}".encode()).hexdigest()


def exclusions() -> set[str]:
    path = Path(__file__).with_name("CONTAMINATION_EXCLUSIONS.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {str(x) for x in payload["combined_exclusions"]}
    if len(values) != int(payload["combined_exclusion_count"]):
        raise RuntimeError("contamination exclusion count mismatch")
    if not payload.get("frozen_before_holdout_selection"):
        raise RuntimeError("contamination exclusions are not marked frozen")
    return values


def source_inventory() -> tuple[set[str], dict[str, object]]:
    payload, headers, raw = fetch_json(SOURCE_TREE_URL, "LEXIGEN-v5-name-only-selector")
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
        if len(parts) == 3 and parts[0] == "AlgoTuneTasks" and parts[1] and parts[2] == f"{parts[1]}.py":
            tasks.add(parts[1])
            matched_paths.append(path)
    if not tasks:
        raise RuntimeError("source tree contains no frozen-layout task names")
    return tasks, {
        "url": SOURCE_TREE_URL,
        "commit": SOURCE_COMMIT,
        "tree_sha": payload.get("sha"),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "etag": headers.get("etag"),
        "layout_rule": "AlgoTuneTasks/<task>/<task>.py",
        "matched_path_count": len(matched_paths),
        "matched_paths_sha256": hashlib.sha256("\n".join(sorted(matched_paths)).encode()).hexdigest(),
        "task_contents_opened": False,
    }


def dataset_inventory() -> tuple[set[str], dict[str, object]]:
    payload, headers, raw = fetch_json(DATASET_TREE_URL, "LEXIGEN-v5-name-only-selector")
    if not isinstance(payload, list):
        raise RuntimeError("dataset tree metadata is not a list")
    tasks: set[str] = set()
    identities: list[tuple[str, str]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", ""))
        if str(entry.get("type", "")) == "directory" and path.startswith("data/"):
            name = path.split("/", 1)[1]
            if name and "/" not in name:
                tasks.add(name)
                identities.append((name, str(entry.get("oid", ""))))
    if not tasks:
        raise RuntimeError("dataset tree contains no task directories")
    return tasks, {
        "url": DATASET_TREE_URL,
        "revision": DATASET_REVISION,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "etag": headers.get("etag"),
        "directory_count": len(identities),
        "directory_identity_sha256": hashlib.sha256(json.dumps(sorted(identities), separators=(",", ":")).encode()).hexdigest(),
        "manifest_contents_opened": False,
        "payloads_opened": False,
    }


def possible_family_count(ordered: list[dict[str, str]], start: int, counts: Counter[str]) -> int:
    remaining = {row["family"] for row in ordered[start:] if counts[row["family"]] < MAX_PER_FAMILY}
    return len(set(counts) | remaining)


def select_tasks(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (row["score"], row["task"]))

    def search(index: int, chosen: list[dict[str, str]], counts: Counter[str]) -> list[dict[str, str]] | None:
        if len(chosen) == TASK_COUNT:
            return chosen[:] if len(counts) >= MIN_FAMILIES else None
        needed = TASK_COUNT - len(chosen)
        if len(ordered) - index < needed or possible_family_count(ordered, index, counts) < MIN_FAMILIES:
            return None
        for position in range(index, len(ordered)):
            row = ordered[position]
            family = row["family"]
            if counts[family] >= MAX_PER_FAMILY:
                continue
            chosen.append(row)
            counts[family] += 1
            result = search(position + 1, chosen, counts)
            if result is not None:
                return result
            counts[family] -= 1
            if counts[family] == 0:
                del counts[family]
            chosen.pop()
        return None

    selected = search(0, [], Counter())
    if selected is None:
        raise RuntimeError("no admissible ten-task v5 selection satisfies frozen diversity constraints")
    return selected


def main() -> None:
    source, source_evidence = source_inventory()
    dataset, dataset_evidence = dataset_inventory()
    excluded = exclusions()
    common = sorted((source & dataset) - excluded)
    rows = [{"task": task, "family": classify(task), "score": task_score(task)} for task in common]
    selected = select_tasks(rows)
    family_counts = Counter(row["family"] for row in selected)
    inventory_payload = {
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "eligible": sorted((row["task"], row["family"], row["score"]) for row in rows),
    }
    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "selection_seed": SEED,
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "source_metadata": source_evidence,
        "dataset_metadata": dataset_evidence,
        "source_task_count": len(source),
        "dataset_task_count": len(dataset),
        "eligible_common_task_count": len(common),
        "inventory_sha256": hashlib.sha256(json.dumps(inventory_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "excluded_tasks": sorted(excluded),
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
    if len(selected) != TASK_COUNT or len(family_counts) < MIN_FAMILIES or any(count > MAX_PER_FAMILY for count in family_counts.values()):
        raise RuntimeError("v5 selection violates frozen task-count/diversity constraints")
    output = Path("selection-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "inventory_sha256": report["inventory_sha256"], "eligible_count": len(common)}, indent=2))


if __name__ == "__main__":
    main()
