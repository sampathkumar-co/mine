from __future__ import annotations

import ast
import hashlib
import json
import struct
import urllib.request
from collections import Counter
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "unit_simplex_projection"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=true&expand=false"
RAW_BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task6-metadata-r1b"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def npy_header_only(relative_path: str) -> tuple[tuple[int, ...], str, int]:
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise RuntimeError(f"unsafe npy_path: {relative_path}")
    url = f"{RAW_BASE}/{relative_path}?download=true"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LEXIGEN-v4-task6-metadata-r1b", "Range": "bytes=0-511"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        status = getattr(response, "status", None)
        if status != 206 or not response.headers.get("Content-Range"):
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
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LEXIGEN-v4-task6-metadata-r1b", "Range": f"bytes=0-{total_header - 1}"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            if getattr(response, "status", None) != 206:
                raise RuntimeError(f"server did not honor extended header-only range request for {relative_path}")
            prefix = response.read(total_header)
    header = ast.literal_eval(prefix[header_start:total_header].decode("latin1").strip())
    shape = tuple(int(v) for v in header["shape"])
    dtype_descr = str(header["descr"])
    return shape, dtype_descr, total_header


def vector_metadata(problem: object) -> tuple[int, str, str, int, bool]:
    if not isinstance(problem, dict) or "y" not in problem:
        raise RuntimeError("unexpected training problem schema")
    y = problem["y"]
    if isinstance(y, list):
        category = "json_number" if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in y) else "other"
        return len(y), category, "inline_json_list", 0, True
    if not isinstance(y, dict):
        raise RuntimeError(f"unsupported y metadata type {type(y).__name__}")
    kind = y.get("__type__")
    if kind != "ndarray_ref":
        raise RuntimeError(f"unsupported y ndarray wrapper: {kind!r}")
    shape, dtype_descr, header_bytes = npy_header_only(str(y["npy_path"]))
    if len(shape) != 1:
        raise RuntimeError(f"Task 6 y sidecar is not 1D: {shape}")
    return shape[0], dtype_descr, "ndarray_ref_header_only", header_bytes, False


def main() -> None:
    tree = json.loads(fetch(TREE_URL))
    files = [x for x in tree if x.get("type") == "file"]
    train = [x for x in files if str(x.get("path", "")).endswith("_train.jsonl")]
    test = [x for x in files if str(x.get("path", "")).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError(f"expected exactly one train/test manifest, got train={len(train)} test={len(test)}")
    train_item, test_item = train[0], test[0]
    train_name = Path(str(train_item["path"])).name
    test_name = Path(str(test_item["path"])).name

    raw = fetch(f"{RAW_BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, got {len(rows)}")

    lengths: Counter[int] = Counter()
    dtypes: Counter[str] = Counter()
    encodings: Counter[str] = Counter()
    header_bytes_total = 0
    inline_payload_seen = False
    for row in rows:
        length, dtype, encoding, header_bytes, inline = vector_metadata(row.get("problem"))
        if length <= 0:
            raise RuntimeError(f"nonpositive training vector length: {length}")
        lengths[length] += 1
        dtypes[dtype] += 1
        encodings[encoding] += 1
        header_bytes_total += header_bytes
        inline_payload_seen = inline_payload_seen or inline

    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 6,
        "task": TASK,
        "stage": "metadata_revision1b",
        "dataset_revision": REVISION,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_item.get("oid"),
        "train_manifest_git_blob_sha1": git_blob(raw),
        "train_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "train_manifest_size": len(raw),
        "test_manifest_name": test_name,
        "test_manifest_tree_oid": test_item.get("oid"),
        "training_records": len(rows),
        "vector_length_counts": {str(k): v for k, v in sorted(lengths.items())},
        "dtype_descriptor_counts": dict(dtypes),
        "encoding_counts": dict(encodings),
        "training_sidecar_access": "NumPy header byte ranges only; numerical sidecar payload not read",
        "training_sidecar_header_bytes_read_total": header_bytes_total,
        "training_manifest_contains_inline_payload": inline_payload_seen,
        "training_values_statistically_summarized": False,
        "training_values_used_for_candidate_design_or_adaptation": False,
        "candidate_family_changed_after_metadata_failure": False,
        "preserved_metadata_failure_run_id": 32649263787,
        "test_manifest_downloaded": False,
        "test_payloads_opened": 0,
        "candidate_execution_count": 0,
        "reference_execution_count": 0,
        "training_revision_consumed": False,
    }
    out = Path("metadata-evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
