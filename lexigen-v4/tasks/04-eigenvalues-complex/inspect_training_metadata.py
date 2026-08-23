from __future__ import annotations

import ast
import base64
import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
from collections import Counter
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
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task4-training-metadata-r1c"})
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


def npy_header_only(relative_path: str) -> tuple[tuple[int, ...], str, int]:
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise RuntimeError(f"unsafe npy_path: {relative_path}")
    url = f"{BASE}/{relative_path}?download=true"
    req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task4-training-metadata-r1c", "Range": "bytes=0-511"})
    with urllib.request.urlopen(req, timeout=240) as response:
        status = getattr(response, "status", None)
        content_range = response.headers.get("Content-Range")
        if status != 206 or not content_range:
            raise RuntimeError(f"server did not honor header-only range request for {relative_path}: status={status}")
        prefix = response.read(512)
    if prefix[:6] != b"\x93NUMPY":
        raise RuntimeError(f"invalid npy magic for {relative_path}")
    major, minor = prefix[6], prefix[7]
    if major == 1:
        header_len = struct.unpack("<H", prefix[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", prefix[8:12])[0]
        header_start = 12
    else:
        raise RuntimeError(f"unsupported npy version {major}.{minor}")
    total_header = header_start + header_len
    if total_header > len(prefix):
        req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task4-training-metadata-r1c", "Range": f"bytes=0-{total_header - 1}"})
        with urllib.request.urlopen(req, timeout=240) as response:
            status = getattr(response, "status", None)
            if status != 206:
                raise RuntimeError(f"server did not honor extended header-only range request for {relative_path}")
            prefix = response.read(total_header)
    header = ast.literal_eval(prefix[header_start:total_header].decode("latin1").strip())
    shape = tuple(int(x) for x in header["shape"])
    dtype = str(np.dtype(header["descr"]))
    return shape, dtype, total_header


def matrix_metadata(problem: object) -> tuple[tuple[int, ...], str, str, int]:
    value = problem
    location = "direct"
    if isinstance(problem, dict) and "__type__" not in problem:
        candidates = [(str(k), v) for k, v in problem.items() if isinstance(v, (dict, list))]
        typed = [(k, v) for k, v in candidates if isinstance(v, dict) and str(v.get("__type__", "")).startswith("ndarray")]
        if len(typed) == 1:
            location, value = typed[0][0], typed[0][1]
        elif len(candidates) == 1:
            location, value = candidates[0][0], candidates[0][1]
    if isinstance(value, list):
        arr = np.asarray(value)
        return tuple(int(x) for x in arr.shape), str(arr.dtype), f"inline_list:{location}", 0
    if not isinstance(value, dict):
        raise RuntimeError(f"unsupported problem matrix type {type(value).__name__}")
    kind = value.get("__type__")
    if kind == "ndarray_b64":
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(x) for x in value["shape"])
        raw = base64.b64decode(str(value["data_b64"]).encode("ascii"), validate=True)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected:
            raise RuntimeError("ndarray_b64 byte length mismatch")
        return shape, str(dtype), f"ndarray_b64:{location}", len(raw)
    if kind == "ndarray":
        arr = np.asarray(value.get("data"), dtype=np.dtype(str(value.get("dtype", "float64"))))
        return tuple(int(x) for x in arr.shape), str(arr.dtype), f"ndarray_inline:{location}", int(arr.nbytes)
    if kind == "ndarray_ref":
        shape, dtype, header_bytes = npy_header_only(str(value["npy_path"]))
        return shape, dtype, f"ndarray_ref_header_only:{location}", header_bytes
    raise RuntimeError(f"unsupported ndarray wrapper: {kind!r}")


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

    shapes: list[tuple[int, ...]] = []
    dtypes: Counter[str] = Counter()
    encodings: Counter[str] = Counter()
    header_bytes_total = 0
    for row in rows:
        shape, dtype, encoding, header_bytes = matrix_metadata(row.get("problem"))
        if len(shape) != 2 or shape[0] <= 0 or shape[0] != shape[1]:
            raise RuntimeError(f"training matrix is not nonempty and square: {shape}")
        shapes.append(shape)
        dtypes[dtype] += 1
        encodings[encoding] += 1
        header_bytes_total += header_bytes

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
        "matrix_shape_values": [list(s) for s in sorted(set(shapes))],
        "matrix_size_min": min(s[0] for s in shapes),
        "matrix_size_max": max(s[0] for s in shapes),
        "dtype_counts": dict(dtypes),
        "encoding_counts": dict(encodings),
        "training_sidecar_access": "NumPy header byte ranges only; numerical payload not read",
        "training_sidecar_header_bytes_read_total": header_bytes_total,
        "test_manifest_downloaded": False,
        "test_sidecars_accessed": 0,
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
