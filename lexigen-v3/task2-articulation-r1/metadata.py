from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "articulation_points"
TREE_URL = (
    "https://huggingface.co/api/datasets/oripress/AlgoTune/tree/"
    f"{REVISION}/data/{TASK}"
)
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str, *, attempts: int = 8) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LEXIGEN-v3-articulation-metadata-only"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
        except Exception as exc:
            last_error = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"metadata fetch exhausted retries: {url}") from last_error


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def describe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if "__type__" in value:
            result: dict[str, Any] = {"descriptor_type": value.get("__type__")}
            for key in ("size", "shape", "dtype", "bin_path", "npy_path"):
                if key in value:
                    result[key] = value[key]
            return result
        return {"python_type": "dict", "keys": sorted(value)}
    if isinstance(value, list):
        first = value[0] if value else None
        return {
            "python_type": "list",
            "length": len(value),
            "first_item_type": type(first).__name__ if first is not None else None,
            "first_item_length": len(first) if isinstance(first, list) else None,
        }
    return {"python_type": type(value).__name__, "value": value if isinstance(value, (int, float, str, bool)) else None}


def decode_problem(problem: dict[str, Any]) -> tuple[int, int]:
    num_nodes = problem.get("num_nodes")
    edges = problem.get("edges")
    if not isinstance(num_nodes, int):
        raise RuntimeError(f"expected inline integer num_nodes, received {type(num_nodes).__name__}")
    if not isinstance(edges, list):
        raise RuntimeError(f"expected inline edge list, received {type(edges).__name__}")
    if any(not isinstance(edge, list) or len(edge) != 2 for edge in edges):
        raise RuntimeError("edge list contains a non-pair")
    return num_nodes, len(edges)


def main() -> None:
    entries = json.loads(fetch(TREE_URL))
    files = [entry for entry in entries if entry.get("type") == "file"]
    train_entries = [entry for entry in files if entry["path"].endswith("_train.jsonl")]
    test_entries = [entry for entry in files if entry["path"].endswith("_test.jsonl")]
    if len(train_entries) != 1 or len(test_entries) != 1:
        raise RuntimeError("expected exactly one train and one test manifest")
    train_entry = train_entries[0]
    test_entry = test_entries[0]
    train_name = Path(train_entry["path"]).name
    test_name = Path(test_entry["path"]).name

    train_raw = fetch(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in train_raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")

    sizes: list[int] = []
    edge_counts: list[int] = []
    for row in rows:
        problem = row.get("problem")
        if not isinstance(problem, dict):
            raise RuntimeError("expected mapping problem")
        nodes, edges = decode_problem(problem)
        sizes.append(nodes)
        edge_counts.append(edges)

    first_problem = rows[0]["problem"]
    report = {
        "task": TASK,
        "dataset_revision": REVISION,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_entry.get("oid"),
        "train_manifest_lfs": train_entry.get("lfs"),
        "train_manifest_resolved_git_blob": git_blob(train_raw),
        "train_manifest_resolved_sha256": hashlib.sha256(train_raw).hexdigest(),
        "test_manifest_name": test_name,
        "test_manifest_tree_oid": test_entry.get("oid"),
        "test_manifest_lfs": test_entry.get("lfs"),
        "training_records": len(rows),
        "row_keys": sorted(rows[0]),
        "problem_keys": sorted(first_problem),
        "problem_descriptors": {
            key: describe(value) for key, value in first_problem.items()
        },
        "num_nodes": {
            "minimum": min(sizes),
            "maximum": max(sizes),
            "unique": sorted(set(sizes)),
        },
        "edge_count": {
            "minimum": min(edge_counts),
            "maximum": max(edge_counts),
            "mean": sum(edge_counts) / len(edge_counts),
        },
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
    }
    Path("metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
