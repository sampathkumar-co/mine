from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
SOURCE_COMMIT = "dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"
TASK = "spectral_clustering"
SOURCE_PATH = "AlgoTuneTasks/spectral_clustering/spectral_clustering.py"
SOURCE_GIT_BLOB_SHA1 = "7e8055db9c069388e4d3fe7468c4a6d4f33c02e8"
TRAIN_NAME = "spectral_clustering_T100ms_n8_size100_train.jsonl"
TRAIN_TREE_OID = "a437767fb704bb4bcccaee3240de0f1080426c90"
TRAIN_SIZE = 15658
TEST_NAME = "spectral_clustering_T100ms_n8_size100_test.jsonl"
TEST_TREE_OID = "dc9082ea69c205b712e98da4b62fb387998e54df"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SOURCE_URL = f"https://raw.githubusercontent.com/oripress/AlgoTune/{SOURCE_COMMIT}/{SOURCE_PATH}"
SHARDS = 10
EXPECTED_RECORDS = 100

ELIGIBLE = {arm: {name for name, _ in rows} for arm, rows in CANDIDATES_BY_ARM.items()}
if set(ELIGIBLE) != {"v5_full", "v5_no_transfer", "random_search", "static_template", "v4_compatible"}:
    raise RuntimeError("Task 3 training arm identity mismatch")
if any(len(names) != 6 for names in ELIGIBLE.values()) or sum(len(names) for names in ELIGIBLE.values()) != 30:
    raise RuntimeError("Task 3 training must contain exactly six frozen candidates per arm")


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v5-task3-train-r1"})
            with urllib.request.urlopen(req, timeout=240) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"training fetch exhausted retries: {url}") from last


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def load_authoritative_task() -> object:
    raw = fetch(SOURCE_URL)
    if git_blob_sha1(raw) != SOURCE_GIT_BLOB_SHA1:
        raise RuntimeError("authoritative Task 3 source Git blob mismatch")

    pkg = types.ModuleType("AlgoTuneTasks")
    pkg.__path__ = []  # type: ignore[attr-defined]
    base = types.ModuleType("AlgoTuneTasks.base")

    class Task:
        def __init__(self, **kwargs: object) -> None:
            pass

    def register_task(name: str):
        def decorator(cls):
            return cls
        return decorator

    base.Task = Task  # type: ignore[attr-defined]
    base.register_task = register_task  # type: ignore[attr-defined]
    sys.modules["AlgoTuneTasks"] = pkg
    sys.modules["AlgoTuneTasks.base"] = base

    module = types.ModuleType("lexigen_v5_task3_authoritative")
    module.__file__ = SOURCE_PATH
    exec(compile(raw, SOURCE_PATH, "exec"), module.__dict__)
    cls = module.__dict__.get("SpectralClusteringTask")
    if cls is None:
        raise RuntimeError("SpectralClusteringTask missing from authoritative source")
    task = cls()
    if not callable(getattr(task, "solve", None)) or not callable(getattr(task, "is_solution", None)):
        raise RuntimeError("authoritative Task 3 solve/is_solution unavailable")
    return task


def decode_array(value: object) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, dict):
        raise RuntimeError(f"unsupported array encoding: {type(value).__name__}")
    kind = value.get("__type__")
    if kind == "ndarray_ref":
        relative = str(value.get("npy_path", ""))
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe ndarray_ref path: {relative}")
        raw = fetch(f"{BASE}/{relative}?download=true")
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        expected_shape = value.get("shape")
        if expected_shape is not None and tuple(arr.shape) != tuple(int(x) for x in expected_shape):
            raise RuntimeError(f"ndarray_ref shape mismatch for {relative}")
        expected_dtype = value.get("dtype")
        if expected_dtype is not None and arr.dtype != np.dtype(str(expected_dtype)):
            raise RuntimeError(f"ndarray_ref dtype mismatch for {relative}")
        return arr
    if kind == "ndarray_b64":
        raw = base64.b64decode(str(value["data_b64"]).encode("ascii"), validate=True)
        return np.frombuffer(raw, dtype=np.dtype(str(value["dtype"]))).reshape(tuple(int(x) for x in value["shape"]))
    if kind == "ndarray":
        return np.asarray(value["data"], dtype=np.dtype(str(value.get("dtype", "float64"))))
    raise RuntimeError(f"unsupported ndarray wrapper: {kind!r}")


def decode_problem(raw: object) -> Problem:
    if not isinstance(raw, dict):
        raise RuntimeError("training problem is not a dictionary")
    if "similarity_matrix" not in raw or "n_clusters" not in raw:
        raise RuntimeError("training problem missing similarity_matrix or n_clusters")
    S = np.asarray(decode_array(raw["similarity_matrix"]), dtype=np.float64)
    k = int(raw["n_clusters"])
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not np.all(np.isfinite(S)):
        raise RuntimeError(f"invalid official similarity matrix shape={S.shape}")
    if k < 1:
        raise RuntimeError("invalid official n_clusters")
    return {"similarity_matrix": S, "n_clusters": k}


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        return solution, time.perf_counter() - start, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def flattened_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    result: list[tuple[str, str, Callable[[Problem], Solution]]] = []
    for arm, rows in CANDIDATES_BY_ARM.items():
        for name, fn in rows:
            if name in ELIGIBLE[arm]:
                result.append((arm, name, fn))
    if len(result) != 30:
        raise RuntimeError(f"expected 30 frozen candidates, got {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0,{SHARDS})")

    task = load_authoritative_task()
    raw = fetch(f"{BASE}/{TRAIN_NAME}?download=true")
    if len(raw) != TRAIN_SIZE:
        raise RuntimeError(f"training manifest size mismatch: {len(raw)} != {TRAIN_SIZE}")
    train_blob = git_blob_sha1(raw)
    if train_blob != TRAIN_TREE_OID:
        raise RuntimeError(f"training manifest Git blob mismatch: {train_blob} != {TRAIN_TREE_OID}")
    train_sha256 = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected {EXPECTED_RECORDS} training records, got {len(rows)}")

    candidates = flattened_candidates()
    evidence: list[dict[str, object]] = []
    for index, row in ((i, r) for i, r in enumerate(rows) if i % SHARDS == args.shard):
        problem = decode_problem(row.get("problem"))
        shift = index % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]

        def reference_fn(p: Problem) -> Solution:
            return task.solve(p)  # type: ignore[attr-defined]

        if index % 2 == 0:
            expected, reference_s, reference_error = timed(reference_fn, problem)
            candidate_results = [(arm, name, *timed(fn, problem)) for arm, name, fn in ordered]
            execution_order = "reference_first"
        else:
            candidate_results = [(arm, name, *timed(fn, problem)) for arm, name, fn in ordered]
            expected, reference_s, reference_error = timed(reference_fn, problem)
            execution_order = "candidates_first"
        if expected is None or reference_s is None or reference_error is not None:
            raise RuntimeError(f"reference failed on training record {index + 1}: {reference_error}")
        if not bool(task.is_solution(problem, expected)):  # type: ignore[attr-defined]
            raise RuntimeError(f"authoritative reference rejected itself on training record {index + 1}")

        S = np.asarray(problem["similarity_matrix"])
        k = int(problem["n_clusters"])
        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            if candidate_error is None and proposed is not None:
                try:
                    valid = bool(task.is_solution(problem, proposed))  # type: ignore[attr-defined]
                    failure_reason = None if valid else "authoritative_is_solution_false"
                except Exception as exc:
                    valid = False
                    failure_reason = f"verifier_exception:{type(exc).__name__}:{exc}"
            else:
                valid = False
                failure_reason = candidate_error or "candidate_exception"
            speedup = reference_s / candidate_s if candidate_s is not None and candidate_s > 0 else 0.0
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": bool(valid),
                "failure_reason": failure_reason,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "matrix_shape": list(S.shape),
                "n_clusters": k,
                "train_manifest_name": TRAIN_NAME,
                "train_manifest_tree_oid": TRAIN_TREE_OID,
                "train_manifest_git_blob_sha1": train_blob,
                "train_manifest_sha256": train_sha256,
                "expected_test_manifest_name": TEST_NAME,
                "expected_test_manifest_tree_oid": TEST_TREE_OID,
                "authoritative_source_git_blob_sha1": SOURCE_GIT_BLOB_SHA1,
                "verifier": "exact_frozen_source_is_solution",
                "execution_order": execution_order,
                "shard": args.shard,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
                "test_manifest_contents_opened": False,
                "test_payloads_opened": 0
            })
            print(f"[{index+1}/100] {arm}/{name} valid={valid} speedup={speedup:.3f} reason={failure_reason}", flush=True)
        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
