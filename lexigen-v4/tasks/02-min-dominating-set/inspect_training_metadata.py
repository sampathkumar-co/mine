from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "min_dominating_set"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task2-training-metadata"})
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


def component_count(matrix: list[list[int]]) -> int:
    n = len(matrix)
    unseen = set(range(n))
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        stack = [start]
        while stack:
            u = stack.pop()
            for v, edge in enumerate(matrix[u]):
                if edge and v in unseen:
                    unseen.remove(v)
                    stack.append(v)
    return count


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

    node_counts: list[int] = []
    edge_counts: list[int] = []
    component_counts: list[int] = []
    degree_values: list[int] = []
    isolated_counts: list[int] = []
    densities: list[float] = []
    encoding_types = Counter()

    for row in rows:
        problem = row.get("problem")
        if not isinstance(problem, list) or any(not isinstance(r, list) for r in problem):
            encoding_types[type(problem).__name__] += 1
            raise RuntimeError("expected inline adjacency-matrix list-of-lists training representation")
        encoding_types["inline_square_list_of_lists"] += 1
        n = len(problem)
        if any(len(r) != n for r in problem):
            raise RuntimeError("training adjacency matrix is not square")
        matrix = [[1 if int(value) else 0 for value in r] for r in problem]
        for i in range(n):
            if matrix[i][i] != 0:
                raise RuntimeError("training adjacency matrix has nonzero diagonal")
            for j in range(i + 1, n):
                if matrix[i][j] != matrix[j][i]:
                    raise RuntimeError("training adjacency matrix is not symmetric")
        degrees = [sum(r) for r in matrix]
        edges = sum(degrees) // 2
        node_counts.append(n)
        edge_counts.append(edges)
        component_counts.append(component_count(matrix))
        degree_values.extend(degrees)
        isolated_counts.append(sum(value == 0 for value in degrees))
        densities.append((2.0 * edges / (n * (n - 1))) if n > 1 else 0.0)

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
        "problem_encoding_counts": dict(encoding_types),
        "node_count_min": min(node_counts),
        "node_count_max": max(node_counts),
        "node_count_values": sorted(set(node_counts)),
        "edge_count_min": min(edge_counts),
        "edge_count_max": max(edge_counts),
        "density_min": min(densities),
        "density_max": max(densities),
        "component_count_min": min(component_counts),
        "component_count_max": max(component_counts),
        "connected_record_count": sum(value == 1 for value in component_counts),
        "disconnected_record_count": sum(value > 1 for value in component_counts),
        "degree_min": min(degree_values) if degree_values else 0,
        "degree_max": max(degree_values) if degree_values else 0,
        "isolated_vertices_min": min(isolated_counts),
        "isolated_vertices_max": max(isolated_counts),
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
        "candidate_execution_count": 0,
        "reference_execution_count": 0,
        "training_revision_consumed": False
    }
    output = Path("metadata-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
