from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "max_common_subgraph"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task3-training-metadata"})
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"metadata fetch exhausted retries: {url}") from last


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def analyse_matrix(raw: object, label: str) -> tuple[int, int, float, int, int]:
    if not isinstance(raw, list) or any(not isinstance(row, list) for row in raw):
        raise RuntimeError(f"{label} is not an inline list-of-lists adjacency matrix")
    n = len(raw)
    if any(len(row) != n for row in raw):
        raise RuntimeError(f"{label} is not square")
    degrees: list[int] = []
    for i, row in enumerate(raw):
        if int(row[i]) != 0:
            raise RuntimeError(f"{label} has nonzero diagonal")
        degree = 0
        for j, value in enumerate(row):
            bit = 1 if int(value) else 0
            if j != i:
                degree += bit
            if j > i and bit != (1 if int(raw[j][i]) else 0):
                raise RuntimeError(f"{label} is not symmetric")
        degrees.append(degree)
    edges = sum(degrees) // 2
    density = 2.0 * edges / (n * (n - 1)) if n > 1 else 0.0
    return n, edges, density, min(degrees) if degrees else 0, max(degrees) if degrees else 0


def main() -> None:
    entries = json.loads(fetch(TREE_URL))
    files = [entry for entry in entries if entry.get("type") == "file"]
    train = [entry for entry in files if str(entry["path"]).endswith("_train.jsonl")]
    test = [entry for entry in files if str(entry["path"]).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError(f"expected one train and one test manifest, received {len(train)} and {len(test)}")
    train_entry, test_entry = train[0], test[0]
    train_name = Path(str(train_entry["path"])).name
    test_name = Path(str(test_entry["path"])).name

    raw = fetch(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")

    a_sizes: list[int] = []
    b_sizes: list[int] = []
    a_edges: list[int] = []
    b_edges: list[int] = []
    a_density: list[float] = []
    b_density: list[float] = []
    a_degree_min: list[int] = []
    a_degree_max: list[int] = []
    b_degree_min: list[int] = []
    b_degree_max: list[int] = []
    problem_keys = Counter()
    encoding_counts = Counter()

    for row in rows:
        problem = row.get("problem")
        if not isinstance(problem, dict):
            raise RuntimeError("training row problem is not a dictionary")
        problem_keys.update(problem)
        if set(problem) != {"A", "B"}:
            raise RuntimeError(f"unexpected problem keys: {sorted(problem)}")
        an, ae, ad, amin, amax = analyse_matrix(problem["A"], "A")
        bn, be, bd, bmin, bmax = analyse_matrix(problem["B"], "B")
        a_sizes.append(an)
        b_sizes.append(bn)
        a_edges.append(ae)
        b_edges.append(be)
        a_density.append(ad)
        b_density.append(bd)
        a_degree_min.append(amin)
        a_degree_max.append(amax)
        b_degree_min.append(bmin)
        b_degree_max.append(bmax)
        encoding_counts["inline_two_square_adjacency_matrices"] += 1

    report = {
        "task": TASK,
        "dataset_revision": REVISION,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_entry.get("oid"),
        "train_manifest_git_blob_sha1": git_blob(raw),
        "train_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "train_manifest_size": len(raw),
        "test_manifest_name": test_name,
        "test_manifest_tree_oid": test_entry.get("oid"),
        "training_records": len(rows),
        "problem_keys": sorted(problem_keys),
        "problem_encoding_counts": dict(encoding_counts),
        "A_node_count_min": min(a_sizes),
        "A_node_count_max": max(a_sizes),
        "A_node_count_values": sorted(set(a_sizes)),
        "B_node_count_min": min(b_sizes),
        "B_node_count_max": max(b_sizes),
        "B_node_count_values": sorted(set(b_sizes)),
        "same_size_record_count": sum(a == b for a, b in zip(a_sizes, b_sizes)),
        "A_edge_count_min": min(a_edges),
        "A_edge_count_max": max(a_edges),
        "B_edge_count_min": min(b_edges),
        "B_edge_count_max": max(b_edges),
        "A_density_min": min(a_density),
        "A_density_max": max(a_density),
        "B_density_min": min(b_density),
        "B_density_max": max(b_density),
        "A_degree_min": min(a_degree_min),
        "A_degree_max": max(a_degree_max),
        "B_degree_min": min(b_degree_min),
        "B_degree_max": max(b_degree_max),
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
        "candidate_execution_count": 0,
        "reference_execution_count": 0,
        "training_revision_consumed": False
    }
    output = Path("metadata-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
