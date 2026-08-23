from __future__ import annotations

import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "eigenvalues_complex"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task4-training-metadata"})
            with urllib.request.urlopen(req, timeout=240) as response:
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


def load_training_matrix(problem: object) -> list[list[float]]:
    if not isinstance(problem, dict) or problem.get("__type__") != "ndarray_ref":
        raise RuntimeError(f"unexpected training problem serialization: {type(problem).__name__}")
    rel = problem.get("npy_path")
    if not isinstance(rel, str) or not rel.startswith("_npy_data/") or ".." in rel:
        raise RuntimeError("invalid training ndarray_ref path")
    raw = fetch(f"{BASE}/{rel}?download=true")
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[0] != arr.shape[1]:
        raise RuntimeError(f"training matrix is not nonempty square: shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise RuntimeError(f"training matrix dtype is nonnumeric: {arr.dtype}")
    return np.asarray(arr, dtype=np.float64).tolist()


def main() -> None:
    entries = json.loads(fetch(TREE_URL))
    files = [e for e in entries if e.get("type") == "file"]
    train = [e for e in files if str(e["path"]).endswith("_train.jsonl")]
    test = [e for e in files if str(e["path"]).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError(f"expected one train and one test manifest, received {len(train)} and {len(test)}")
    train_entry, test_entry = train[0], test[0]
    train_name = Path(str(train_entry["path"])).name
    test_name = Path(str(test_entry["path"])).name

    raw = fetch(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")

    sizes: list[int] = []
    abs_max: list[float] = []
    frob_sq: list[float] = []
    exact_symmetric = 0
    finite_entries = 0
    total_entries = 0

    for row in rows:
        matrix = load_training_matrix(row.get("problem"))
        n = len(matrix)
        sizes.append(n)
        flat = [float(x) for r in matrix for x in r]
        finite_entries += sum(math.isfinite(x) for x in flat)
        total_entries += len(flat)
        abs_max.append(max(abs(x) for x in flat))
        frob_sq.append(sum(x * x for x in flat))
        if all(matrix[i][j] == matrix[j][i] for i in range(n) for j in range(n)):
            exact_symmetric += 1

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
        "training_npy_payloads_downloaded": len(rows),
        "matrix_size_min": min(sizes),
        "matrix_size_max": max(sizes),
        "matrix_size_values": sorted(set(sizes)),
        "exact_symmetric_records": exact_symmetric,
        "finite_entry_fraction": finite_entries / total_entries,
        "entry_abs_max_min": min(abs_max),
        "entry_abs_max_max": max(abs_max),
        "frob_norm_min": math.sqrt(min(frob_sq)),
        "frob_norm_max": math.sqrt(max(frob_sq)),
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
        "candidate_execution_count": 0,
        "reference_execution_count": 0,
        "eigensolver_execution_count": 0,
        "training_revision_consumed": False,
    }
    out = Path("metadata-evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
