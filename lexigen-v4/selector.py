from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

SOURCE_COMMIT = "dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"
DATASET_REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
SEED = "LEXIGEN-V4-GENERALIZATION-2026-07-28-A"
TASK_COUNT = 8
MIN_FAMILIES = 6
MAX_PER_FAMILY = 2

EXCLUSIONS = {
    "numerical_integration",
    "water_filling",
    "polynomial_mixed",
    "vector_quantization",
    "integer_factorization",
    "chacha_encryption",
    "outer_product",
    "base64_encoding",
    "articulation_points",
    "cvar_projection",
    "kmeans",
    "procrustes",
    "sha256_hashing",
}

# These rules are frozen before inventory access and use task names only.
FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cryptography_encoding", (r"cipher", r"encrypt", r"decrypt", r"hash", r"sha", r"base\d+", r"codec", r"encoding", r"compression")),
    ("linear_algebra", (r"matrix", r"svd", r"eigen", r"cholesky", r"qr", r"least_squares", r"linear_system", r"tensor", r"product")),
    ("numerical_optimization", (r"optimi", r"projection", r"simplex", r"portfolio", r"resource", r"flow", r"assignment", r"transport", r"cvar")),
    ("graph_discrete", (r"graph", r"shortest", r"path", r"tree", r"clique", r"color", r"matching", r"articulation", r"network", r"mst", r"cycle")),
    ("combinatorial", (r"cover", r"subset", r"set_", r"knapsack", r"permutation", r"combination", r"partition", r"scheduling", r"integer", r"factor")),
    ("signal_processing", (r"fft", r"fourier", r"filter", r"signal", r"convolution", r"correlation", r"wavelet", r"spectral", r"audio", r"image")),
    ("statistics", (r"stat", r"mean", r"median", r"quantile", r"regression", r"probab", r"distribution", r"sampling", r"variance", r"entropy")),
    ("scientific_computing", (r"integrat", r"differential", r"ode", r"pde", r"physics", r"simulation", r"scientific", r"nbody", r"monte_carlo")),
    ("string_sequence", (r"string", r"sequence", r"edit_distance", r"alignment", r"substring", r"prefix", r"suffix", r"token")),
    ("geometry", (r"geometry", r"convex_hull", r"distance", r"point", r"polygon", r"mesh", r"spatial")),
    ("machine_learning", (r"cluster", r"classification", r"neural", r"learning", r"pca", r"embedding", r"nearest", r"kmeans")),
)


def classify(task_name: str) -> str:
    lowered = task_name.lower()
    for family, patterns in FAMILY_RULES:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return family
    return "miscellaneous"


def task_score(task_name: str) -> str:
    return hashlib.sha256(f"{SEED}\0{task_name}".encode("utf-8")).hexdigest()


def source_inventory(source_root: Path) -> set[str]:
    root = source_root / "AlgoTuneTasks"
    if not root.is_dir():
        raise RuntimeError(f"missing AlgoTuneTasks directory: {root}")
    tasks: set[str] = set()
    for task_file in root.glob("*/task.py"):
        tasks.add(task_file.parent.name)
    if not tasks:
        raise RuntimeError("source task inventory is empty")
    return tasks


def dataset_inventory() -> set[str]:
    url = (
        "https://huggingface.co/api/datasets/oripress/AlgoTune/tree/"
        f"{DATASET_REVISION}/data?recursive=false&expand=false&limit=1000"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-selector"})
    with urllib.request.urlopen(request, timeout=180) as response:
        entries = json.loads(response.read())
    tasks: set[str] = set()
    for entry in entries:
        path = str(entry.get("path", ""))
        entry_type = str(entry.get("type", ""))
        if entry_type == "directory" and path.startswith("data/"):
            name = path.split("/", 1)[1]
            if name and "/" not in name:
                tasks.add(name)
    if not tasks:
        raise RuntimeError("dataset task inventory is empty")
    return tasks


def _possible_family_count(ordered: list[dict[str, str]], start: int, counts: Counter[str]) -> int:
    remaining = {row["family"] for row in ordered[start:] if counts[row["family"]] < MAX_PER_FAMILY}
    return len(set(counts) | remaining)


def select_tasks(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (row["score"], row["task"]))

    def search(index: int, chosen: list[dict[str, str]], counts: Counter[str]) -> list[dict[str, str]] | None:
        if len(chosen) == TASK_COUNT:
            return chosen[:] if len(counts) >= MIN_FAMILIES else None
        needed = TASK_COUNT - len(chosen)
        if len(ordered) - index < needed:
            return None
        if _possible_family_count(ordered, index, counts) < MIN_FAMILIES:
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
        raise RuntimeError("no admissible eight-task selection satisfies the frozen diversity constraints")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = source_inventory(args.source_root)
    dataset = dataset_inventory()
    common = sorted((source & dataset) - EXCLUSIONS)
    rows = [
        {"task": task, "family": classify(task), "score": task_score(task)}
        for task in common
    ]
    inventory_payload = {
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "tasks": sorted({row["task"]: row["family"] for row in rows}.items()),
    }
    inventory_json = json.dumps(inventory_payload, sort_keys=True, separators=(",", ":"))
    selected = select_tasks(rows)
    family_counts = Counter(row["family"] for row in selected)
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "selection_seed": SEED,
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "source_task_count": len(source),
        "dataset_task_count": len(dataset),
        "eligible_common_task_count": len(common),
        "inventory_sha256": hashlib.sha256(inventory_json.encode()).hexdigest(),
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
    }
    if len(selected) != TASK_COUNT or len(family_counts) < MIN_FAMILIES:
        raise RuntimeError("selector produced an invalid campaign")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "inventory_sha256": report["inventory_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
