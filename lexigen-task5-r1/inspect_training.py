from __future__ import annotations

import hashlib
import json
import tempfile
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "outer_product"
TREE_API = (
    "https://huggingface.co/api/datasets/oripress/AlgoTune/tree/"
    f"{REVISION}/data/{TASK}?recursive=false&expand=false"
)
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
OUTPUT = Path("training-metadata.json")


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v2-task5-metadata"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def tuple_items(problem: object) -> list[dict[str, object]]:
    if not isinstance(problem, dict) or problem.get("__type__") != "tuple":
        raise TypeError(f"expected tagged tuple problem, received {problem}")
    for key in ("items", "values", "data", "value"):
        value = problem.get(key)
        if isinstance(value, list):
            return value
    raise TypeError(f"tagged tuple has unsupported keys: {sorted(problem)}")


def inspect() -> dict[str, object]:
    tree = json.loads(request_bytes(TREE_API))
    if not isinstance(tree, list):
        raise TypeError(f"expected tree list, received {type(tree).__name__}")
    files = [entry for entry in tree if entry.get("type") == "file"]
    train = next(entry for entry in files if str(entry["path"]).endswith("_train.jsonl"))
    test = next(entry for entry in files if str(entry["path"]).endswith("_test.jsonl"))

    train_name = Path(str(train["path"])).name
    train_raw = request_bytes(f"{BASE}/{urllib.parse.quote(train_name)}?download=true")
    resolved_blob = git_blob(train_raw)
    rows = [json.loads(line) for line in train_raw.decode().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("empty training manifest")

    first_problem = rows[0]["problem"]
    items = tuple_items(first_problem)
    if len(items) != 2:
        raise ValueError(f"expected two tuple items, received {len(items)}")
    descriptors: list[tuple[str, str]] = []
    for index, value in enumerate(items):
        if not isinstance(value, dict) or value.get("__type__") != "ndarray_ref":
            raise TypeError(f"unexpected tuple item {index}: {value}")
        descriptors.append((f"item_{index}", str(value["npy_path"])))

    first_arrays: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for key, relative in descriptors:
            payload = root / Path(relative).name
            payload.write_bytes(request_bytes(f"{BASE}/{relative}?download=true"))
            array = np.load(payload, allow_pickle=False)
            first_arrays[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "c_contiguous": bool(array.flags.c_contiguous),
                "f_contiguous": bool(array.flags.f_contiguous),
                "nbytes": int(array.nbytes),
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
            }

    k_values = sorted({int(row["k"]) for row in rows if row.get("k") is not None})
    return {
        "status": "success",
        "task": TASK,
        "dataset_revision": REVISION,
        "directory_entries": tree,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train["oid"],
        "train_manifest_resolved_git_blob_sha1": resolved_blob,
        "tree_oid_matches_resolved_git_blob": bool(train["oid"] == resolved_blob),
        "test_manifest_name": Path(str(test["path"])).name,
        "expected_test_manifest_tree_oid": test["oid"],
        "training_records": len(rows),
        "row_keys": sorted(rows[0]),
        "problem_encoding": {"__type__": "tuple", "item_count": len(items)},
        "k_values": k_values,
        "seed_min": min(int(row["seed"]) for row in rows),
        "seed_max": max(int(row["seed"]) for row in rows),
        "first_training_arrays": first_arrays,
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
    }


def main() -> None:
    try:
        summary = inspect()
    except Exception as exc:
        summary = {
            "status": "failure",
            "task": TASK,
            "dataset_revision": REVISION,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "test_manifest_downloaded": False,
            "test_payloads_downloaded": 0,
        }
        OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        raise
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
