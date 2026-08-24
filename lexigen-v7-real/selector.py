from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"
DATASET_REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
SEED = "LEXIGEN-V7-REAL-MECHANISM-GENESIS-2026-08-24-B"
TASK_COUNT = 5
MIN_FAMILIES = 5
MAX_PER_FAMILY = 1
FORBIDDEN_HOLDOUT_FAMILIES = {"graph_discrete", "numerical_optimization", "linear_algebra", "signal_processing"}
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


def fetch_json(url: str, agent: str) -> tuple[object, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": agent, "Accept": "application/vnd.github+json, application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        raw = r.read()
    return json.loads(raw), raw


def classify(name: str) -> str:
    low = name.lower()
    for family, patterns in FAMILY_RULES:
        if any(re.search(p, low) for p in patterns):
            return family
    return "miscellaneous"


def score(name: str) -> str:
    return hashlib.sha256(f"{SEED}\0{name}".encode()).hexdigest()


def exclusions() -> set[str]:
    p = json.loads(Path(__file__).with_name("CONTAMINATION_EXCLUSIONS.json").read_text())
    values = {str(x) for x in p["excluded_tasks"]}
    if len(values) != int(p["excluded_task_count"]):
        raise RuntimeError("exclusion-count mismatch")
    if not p.get("frozen_before_holdout_selection"):
        raise RuntimeError("exclusions not frozen")
    return values


def source_tasks() -> tuple[set[str], str]:
    payload, raw = fetch_json(SOURCE_TREE_URL, "LEXIGEN-v7-real-selector-r2")
    if not isinstance(payload, dict) or payload.get("truncated"):
        raise RuntimeError("source tree missing/truncated")
    out = set()
    for e in payload.get("tree", []):
        if not isinstance(e, dict) or e.get("type") != "blob":
            continue
        path = str(e.get("path", ""))
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "AlgoTuneTasks" and parts[2] == f"{parts[1]}.py":
            out.add(parts[1])
    return out, hashlib.sha256(raw).hexdigest()


def dataset_tasks() -> tuple[set[str], str]:
    payload, raw = fetch_json(DATASET_TREE_URL, "LEXIGEN-v7-real-selector-r2")
    if not isinstance(payload, list):
        raise RuntimeError("dataset tree is not a list")
    out = set()
    for e in payload:
        if isinstance(e, dict) and str(e.get("type")) == "directory":
            path = str(e.get("path", ""))
            if path.startswith("data/") and path.count("/") == 1:
                out.add(path.split("/", 1)[1])
    return out, hashlib.sha256(raw).hexdigest()


def main() -> None:
    src, src_sha = source_tasks()
    data, data_sha = dataset_tasks()
    excluded = exclusions()
    rows = []
    for task in sorted((src & data) - excluded):
        family = classify(task)
        if family in FORBIDDEN_HOLDOUT_FAMILIES:
            continue
        rows.append({"task": task, "family": family, "score": score(task)})
    ordered = sorted(rows, key=lambda r: (r["score"], r["task"]))
    selected = []
    seen = set()
    for row in ordered:
        if row["family"] in seen:
            continue
        selected.append(row)
        seen.add(row["family"])
        if len(selected) == TASK_COUNT:
            break
    if len(selected) != TASK_COUNT or len(seen) != MIN_FAMILIES:
        raise RuntimeError(f"could not select five distinct eligible families: got {selected}")
    report = {
        "campaign": "LEXIGEN V7 Real Mechanism-Genesis Pilot R2",
        "stage": "name_only_holdout_selection_r2",
        "selection_seed": SEED,
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "source_inventory_response_sha256": src_sha,
        "dataset_inventory_response_sha256": data_sha,
        "excluded_task_count": len(excluded),
        "forbidden_holdout_families": sorted(FORBIDDEN_HOLDOUT_FAMILIES),
        "eligible_task_count": len(rows),
        "selected": selected,
        "selected_family_count": len(seen),
        "task_contents_opened": False,
        "descriptions_opened": False,
        "manifests_opened": False,
        "payloads_opened": False,
        "reports_opened": False,
        "public_solvers_opened": False,
    }
    out = Path("selection-evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
