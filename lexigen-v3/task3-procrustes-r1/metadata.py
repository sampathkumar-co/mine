from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "procrustes"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-procrustes-metadata"})
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code != 429:
                raise
        except Exception as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"metadata fetch failed: {url}") from last


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def describe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "descriptor_type": value.get("__type__"),
            "shape": value.get("shape"),
            "dtype": value.get("dtype"),
            "npy_path": value.get("npy_path"),
            "keys": sorted(value),
        }
    if isinstance(value, list):
        return {
            "python_type": "list",
            "rows": len(value),
            "columns": len(value[0]) if value and isinstance(value[0], list) else None,
        }
    return {"python_type": type(value).__name__}


def dimension(value: Any) -> int | None:
    if isinstance(value, dict):
        shape = value.get("shape")
        if isinstance(shape, list) and len(shape) == 2 and shape[0] == shape[1]:
            return int(shape[0])
    if isinstance(value, list) and value and isinstance(value[0], list) and len(value) == len(value[0]):
        return len(value)
    return None


def main() -> None:
    entries = json.loads(fetch(TREE_URL))
    files = [entry for entry in entries if entry.get("type") == "file"]
    train = [entry for entry in files if str(entry["path"]).endswith("_train.jsonl")]
    test = [entry for entry in files if str(entry["path"]).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError("expected one train and one test manifest")
    train_entry, test_entry = train[0], test[0]
    train_name = Path(train_entry["path"]).name
    test_name = Path(test_entry["path"]).name
    raw = fetch(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")

    dimensions: list[int] = []
    descriptor_pairs = set()
    for row in rows:
        problem = row["problem"]
        if sorted(problem) != ["A", "B"]:
            raise RuntimeError(f"unexpected problem keys: {sorted(problem)}")
        a_desc = describe(problem["A"])
        b_desc = describe(problem["B"])
        descriptor_pairs.add(json.dumps([a_desc, b_desc], sort_keys=True))
        a_dim = dimension(problem["A"])
        b_dim = dimension(problem["B"])
        if a_dim is not None and b_dim is not None:
            if a_dim != b_dim:
                raise RuntimeError("available matrix dimension metadata disagrees")
            dimensions.append(a_dim)

    report = {
        "task": TASK,
        "dataset_revision": REVISION,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_entry.get("oid"),
        "train_manifest_git_blob_sha1": git_blob(raw),
        "train_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "test_manifest_name": test_name,
        "test_manifest_tree_oid": test_entry.get("oid"),
        "training_records": len(rows),
        "problem_keys": ["A", "B"],
        "first_A_descriptor": describe(rows[0]["problem"]["A"]),
        "first_B_descriptor": describe(rows[0]["problem"]["B"]),
        "descriptor_pair_forms": [json.loads(value) for value in sorted(descriptor_pairs)],
        "inline_dimension_metadata_available": len(dimensions) == len(rows),
        "dimension_min": min(dimensions) if dimensions else None,
        "dimension_max": max(dimensions) if dimensions else None,
        "dimension_values": sorted(set(dimensions)),
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
    }
    Path("metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
