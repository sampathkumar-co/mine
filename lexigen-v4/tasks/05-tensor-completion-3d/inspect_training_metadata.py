from __future__ import annotations

import ast
import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "tensor_completion_3d"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
USER_AGENT = "LEXIGEN-v4-task5-training-metadata-r1"


def fetch(url: str, *, range_header: str | None = None) -> tuple[bytes, int | None, str | None]:
    last: Exception | None = None
    for attempt in range(8):
        try:
            headers = {"User-Agent": USER_AGENT}
            if range_header is not None:
                headers["Range"] = range_header
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=240) as response:
                return response.read(), getattr(response, "status", None), response.headers.get("Content-Range")
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


def npy_header_only(relative_path: str) -> tuple[tuple[int, ...], str, bool, int]:
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise RuntimeError(f"unsafe npy_path: {relative_path}")
    url = f"{BASE}/{relative_path}?download=true"
    prefix, status, content_range = fetch(url, range_header="bytes=0-511")
    if status != 206 or not content_range:
        raise RuntimeError(f"server did not honor header-only range request for {relative_path}: status={status}")
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
        prefix, status, _ = fetch(url, range_header=f"bytes=0-{total_header - 1}")
        if status != 206:
            raise RuntimeError(f"server did not honor extended header-only range request for {relative_path}")
    header = ast.literal_eval(prefix[header_start:total_header].decode("latin1").strip())
    shape = tuple(int(x) for x in header["shape"])
    descr = str(header["descr"])
    fortran = bool(header["fortran_order"])
    return shape, descr, fortran, total_header


def ref_meta(value: object, label: str) -> tuple[tuple[int, ...], str, bool, int, str]:
    if not isinstance(value, dict) or value.get("__type__") != "ndarray_ref":
        raise RuntimeError(f"{label} is not an ndarray_ref; refusing numerical training payload access")
    relative = str(value.get("npy_path", ""))
    shape, descr, fortran, header_bytes = npy_header_only(relative)
    return shape, descr, fortran, header_bytes, relative


def main() -> None:
    tree_raw, _, _ = fetch(TREE_URL)
    entries = json.loads(tree_raw)
    files = [e for e in entries if e.get("type") == "file"]
    train = [e for e in files if str(e["path"]).endswith("_train.jsonl")]
    test = [e for e in files if str(e["path"]).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError(f"expected one train and one test manifest, received {len(train)} and {len(test)}")
    train_entry, test_entry = train[0], test[0]
    train_name = Path(str(train_entry["path"])).name
    test_name = Path(str(test_entry["path"])).name

    raw, _, _ = fetch(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")

    tensor_shapes: Counter[tuple[int, ...]] = Counter()
    tensor_dtypes: Counter[str] = Counter()
    mask_dtypes: Counter[str] = Counter()
    fortran_counts: Counter[str] = Counter()
    header_bytes_total = 0
    tensor_ref_paths: set[str] = set()
    mask_ref_paths: set[str] = set()

    for index, row in enumerate(rows, start=1):
        problem = row.get("problem")
        if not isinstance(problem, dict) or "tensor" not in problem or "mask" not in problem:
            raise RuntimeError(f"record {index} has unexpected problem schema")
        tshape, tdtype, tfortran, tbytes, tpath = ref_meta(problem["tensor"], f"record {index} tensor")
        mshape, mdtype, mfortran, mbytes, mpath = ref_meta(problem["mask"], f"record {index} mask")
        if tshape != mshape or len(tshape) != 3 or any(d <= 0 for d in tshape):
            raise RuntimeError(f"record {index} tensor/mask shape mismatch: {tshape} vs {mshape}")
        tensor_shapes[tshape] += 1
        tensor_dtypes[tdtype] += 1
        mask_dtypes[mdtype] += 1
        fortran_counts[f"tensor={tfortran},mask={mfortran}"] += 1
        header_bytes_total += tbytes + mbytes
        tensor_ref_paths.add(tpath)
        mask_ref_paths.add(mpath)

    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 5,
        "task": TASK,
        "stage": "metadata_revision1",
        "dataset_revision": REVISION,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_entry.get("oid"),
        "train_manifest_git_blob_sha1": git_blob(raw),
        "train_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "train_manifest_size": len(raw),
        "test_manifest_name": test_name,
        "test_manifest_tree_oid": test_entry.get("oid"),
        "training_records": len(rows),
        "tensor_shape_counts": {"x".join(map(str, k)): v for k, v in sorted(tensor_shapes.items())},
        "tensor_dtype_counts": dict(tensor_dtypes),
        "mask_dtype_counts": dict(mask_dtypes),
        "fortran_order_counts": dict(fortran_counts),
        "unique_tensor_sidecars": len(tensor_ref_paths),
        "unique_mask_sidecars": len(mask_ref_paths),
        "training_sidecar_access": "NumPy header byte ranges only; numerical payload not read",
        "training_sidecar_header_bytes_read_total": header_bytes_total,
        "training_numerical_payload_read": False,
        "test_manifest_downloaded": False,
        "test_sidecars_accessed": 0,
        "candidate_execution_count": 0,
        "reference_execution_count": 0,
        "cvxpy_solve_count": 0,
        "training_revision_consumed": False,
    }
    out = Path("metadata-evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
