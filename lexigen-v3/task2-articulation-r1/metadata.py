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
            for key in (
                "size",
                "shape",
                "dtype",
                "bin_path",
                "npy_path",
                "length",
                "item_type",
            ):
                if key in value:
                    result[key] = value[key]
            result["keys"] = sorted(key for key in value if key != "data_b64")
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
    return {
        "python_type": type(value).__name__,
        "value": value if isinstance(value, (int, float, str, bool)) else None,
    }


def inline_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def sequence_length(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        shape = value.get("shape")
        if isinstance(shape, list) and shape and isinstance(shape[0], int):
            return shape[0]
        size = value.get("size")
        if isinstance(size, int):
            return size
        length = value.get("length")
        if isinstance(length, int):
            return length
    return None


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
    if not all(isinstance(row.get("problem"), dict) for row in rows):
        raise RuntimeError("expected every training problem to be a mapping")

    problems = [row["problem"] for row in rows]
    first_problem = problems[0]
    node_values = [inline_int(problem.get("num_nodes")) for problem in problems]
    edge_counts = [sequence_length(problem.get("edges")) for problem in problems]
    descriptor_sets = {
        key: sorted(
            {
                json.dumps(describe(problem.get(key)), sort_keys=True)
                for problem in problems
            }
        )
        for key in sorted(first_problem)
    }

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
        "first_problem_descriptors": {
            key: describe(value) for key, value in first_problem.items()
        },
        "all_descriptor_forms": descriptor_sets,
        "inline_num_nodes": {
            "available_for_all_records": all(value is not None for value in node_values),
            "minimum": min(value for value in node_values if value is not None) if any(value is not None for value in node_values) else None,
            "maximum": max(value for value in node_values if value is not None) if any(value is not None for value in node_values) else None,
            "unique": sorted({value for value in node_values if value is not None}),
        },
        "edge_count_from_inline_or_shape_metadata": {
            "available_for_all_records": all(value is not None for value in edge_counts),
            "minimum": min(value for value in edge_counts if value is not None) if any(value is not None for value in edge_counts) else None,
            "maximum": max(value for value in edge_counts if value is not None) if any(value is not None for value in edge_counts) else None,
            "mean": (sum(value for value in edge_counts if value is not None) / sum(value is not None for value in edge_counts)) if any(value is not None for value in edge_counts) else None,
        },
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
    }
    Path("metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
