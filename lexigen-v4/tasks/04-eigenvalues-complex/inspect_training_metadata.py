from __future__ import annotations

import base64
import hashlib
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
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task4-training-metadata-r1b"})
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


def decode_ndarray_wrapper(value: object) -> np.ndarray | None:
    if isinstance(value, list):
        try:
            arr = np.asarray(value, dtype=np.float64)
        except Exception:
            return None
        return arr if arr.ndim == 2 else None
    if not isinstance(value, dict):
        return None
    kind = value.get("__type__")
    if kind == "ndarray_b64":
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(x) for x in value["shape"])
        payload = base64.b64decode(str(value["data_b64"]).encode("ascii"), validate=True)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(payload) != expected_bytes:
            raise RuntimeError(f"ndarray_b64 byte length mismatch: {len(payload)} != {expected_bytes}")
        arr = np.frombuffer(payload, dtype=dtype).reshape(shape)
        return np.asarray(arr, dtype=np.float64)
    if kind == "ndarray":
        arr = np.asarray(value.get("data"), dtype=np.float64)
        shape = tuple(int(x) for x in value.get("shape", arr.shape))
        if tuple(arr.shape) != shape:
            raise RuntimeError(f"inline ndarray shape mismatch: {arr.shape} != {shape}")
        return arr
    if kind == "ndarray_ref":
        raise RuntimeError("metadata stage refuses external ndarray_ref sidecar downloads")
    return None


def numeric_matrix(problem: object) -> tuple[np.ndarray, str]:
    direct = decode_ndarray_wrapper(problem)
    if direct is not None:
        return direct, "direct"
    if isinstance(problem, dict):
        decoded: list[tuple[str, np.ndarray]] = []
        for key, value in problem.items():
            arr = decode_ndarray_wrapper(value)
            if arr is not None:
                decoded.append((str(key), arr))
        if len(decoded) == 1:
            key, arr = decoded[0]
            return arr, f"dict_key:{key}"
        raise RuntimeError(f"expected exactly one matrix field in problem dict; found {[key for key, _ in decoded]}")
    raise RuntimeError(f"unable to identify one numeric matrix from problem type {type(problem).__name__}")


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
    frob_norms: list[float] = []
    exact_symmetric = 0
    finite_entries = 0
    total_entries = 0
    problem_type_counts: dict[str, int] = {}
    encoding_counts: dict[str, int] = {}

    for row in rows:
        problem = row.get("problem")
        problem_type_counts[type(problem).__name__] = problem_type_counts.get(type(problem).__name__, 0) + 1
        matrix, location = numeric_matrix(problem)
        encoding_counts[location] = encoding_counts.get(location, 0) + 1
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
            raise RuntimeError(f"training matrix is not nonempty and square: {matrix.shape}")
        n = int(matrix.shape[0])
        sizes.append(n)
        finite = np.isfinite(matrix)
        finite_entries += int(np.count_nonzero(finite))
        total_entries += int(matrix.size)
        if not bool(np.all(finite)):
            raise RuntimeError("training matrix contains a nonfinite value")
        abs_max.append(float(np.max(np.abs(matrix))))
        frob_norms.append(float(np.linalg.norm(matrix, ord="fro")))
        if bool(np.array_equal(matrix, matrix.T)):
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
        "problem_type_counts": problem_type_counts,
        "matrix_encoding_location_counts": encoding_counts,
        "matrix_size_min": min(sizes),
        "matrix_size_max": max(sizes),
        "matrix_size_values": sorted(set(sizes)),
        "exact_symmetric_records": exact_symmetric,
        "finite_entry_fraction": finite_entries / total_entries,
        "entry_abs_max_min": min(abs_max),
        "entry_abs_max_max": max(abs_max),
        "frob_norm_min": min(frob_norms),
        "frob_norm_max": max(frob_norms),
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
        "candidate_execution_count": 0,
        "reference_execution_count": 0,
        "eigensolver_execution_count": 0,
        "training_revision_consumed": False,
        "parser_correction": "supports AlgoTune ndarray_b64/ndarray wrappers; explicitly refuses ndarray_ref sidecars",
    }
    out = Path("metadata-evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
