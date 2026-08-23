from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "communicability"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task1-training-metadata"})
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


def component_count(adjacency_list: list[list[int]]) -> int:
    n = len(adjacency_list)
    unseen = set(range(n))
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adjacency_list[u]:
                vertex = int(v)
                if vertex in unseen:
                    unseen.remove(vertex)
                    stack.append(vertex)
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
    undirected_edges: list[int] = []
    component_counts: list[int] = []
    degree_values: list[int] = []
    problem_keys = Counter()
    encoding_types = Counter()
    for row in rows:
        problem = row.get("problem")
        if not isinstance(problem, dict):
            raise RuntimeError("training row problem is not a dictionary")
        problem_keys.update(problem)
        adjacency = problem.get("adjacency_list")
        if not isinstance(adjacency, list) or any(not isinstance(neighbors, list) for neighbors in adjacency):
            encoding_types[type(adjacency).__name__] += 1
            raise RuntimeError("expected inline adjacency_list training representation")
        encoding_types["inline_list_of_lists"] += 1
        normalised = [[int(v) for v in neighbors] for neighbors in adjacency]
        n = len(normalised)
        node_counts.append(n)
        degree_values.extend(len(neighbors) for neighbors in normalised)
        undirected_edges.append(sum(len(neighbors) for neighbors in normalised) // 2)
        component_counts.append(component_count(normalised))

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
        "problem_encoding_counts": dict(encoding_types),
        "node_count_min": min(node_counts),
        "node_count_max": max(node_counts),
        "node_count_values": sorted(set(node_counts)),
        "edge_count_min": min(undirected_edges),
        "edge_count_max": max(undirected_edges),
        "component_count_min": min(component_counts),
        "component_count_max": max(component_counts),
        "connected_record_count": sum(value == 1 for value in component_counts),
        "disconnected_record_count": sum(value > 1 for value in component_counts),
        "degree_min": min(degree_values) if degree_values else 0,
        "degree_max": max(degree_values) if degree_values else 0,
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
