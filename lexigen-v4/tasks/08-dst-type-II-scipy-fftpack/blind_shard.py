from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.fftpack

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "dst_type_II_scipy_fftpack"
MANIFEST = "dst_type_II_scipy_fftpack_T100ms_n2054_size100_test.jsonl"
EXPECTED_GIT_BLOB_SHA1 = "92c05fdab7f07bda70c6462485de69c5eaa7d665"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
EXPECTED_SHAPE = (2054, 2054)
EXPECTED_DTYPE = np.dtype("float64")
TOL = 1e-6
SELECTED = {
    "v4_full": "v4_zero_dtype_vector",
    "v4_no_transfer": "no_transfer_contiguous_dtype_vector",
    "random_search": "random_zero_dtype_vector",
    "template_synthesis": "template_dtype",
    "v3_compatible": "v3_dtype_specialization",
}


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task8-blind-r1"})
            with urllib.request.urlopen(req, timeout=240) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"blind fetch exhausted retries: {url}") from last


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def decode_problem(problem_raw: object) -> Problem:
    value = problem_raw
    if isinstance(value, dict) and "__type__" not in value:
        if "x" in value:
            value = value["x"]
        elif len(value) == 1:
            value = next(iter(value.values()))
        else:
            typed = [v for v in value.values() if isinstance(v, dict) and str(v.get("__type__", "")).startswith("ndarray")]
            if len(typed) == 1:
                value = typed[0]
    if isinstance(value, list):
        arr = np.asarray(value)
    elif isinstance(value, dict) and value.get("__type__") == "ndarray_ref":
        relative = str(value.get("npy_path", ""))
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe ndarray_ref path: {relative}")
        arr = np.load(io.BytesIO(fetch(f"{BASE}/{relative}?download=true")), allow_pickle=False)
    elif isinstance(value, dict) and value.get("__type__") == "ndarray_b64":
        raw = base64.b64decode(str(value["data_b64"]).encode("ascii"), validate=True)
        arr = np.frombuffer(raw, dtype=np.dtype(str(value["dtype"]))).reshape(tuple(int(x) for x in value["shape"]))
    elif isinstance(value, dict) and value.get("__type__") == "ndarray":
        arr = np.asarray(value["data"], dtype=np.dtype(str(value.get("dtype", "float64"))))
    else:
        raise RuntimeError(f"unsupported blind problem encoding: {type(value).__name__}")
    arr = np.asarray(arr)
    if arr.shape != EXPECTED_SHAPE or arr.dtype != EXPECTED_DTYPE or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"unexpected blind matrix: shape={arr.shape} dtype={arr.dtype}")
    return arr


def reference(problem: Problem) -> Solution:
    return np.asarray(scipy.fftpack.dstn(problem, type=2))


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        result = fn(problem)
        elapsed = time.perf_counter() - start
        return result, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def validate(problem: Problem, proposed: Solution | None, expected: Solution) -> tuple[bool, float, str | None]:
    try:
        candidate = np.asarray(proposed)
    except Exception:
        return False, float("inf"), "solution_decode"
    if candidate.shape != problem.shape or not np.all(np.isfinite(candidate)):
        return False, float("inf"), "shape_or_nonfinite"
    ref = np.asarray(expected, dtype=np.float64)
    relerr = float(np.linalg.norm(candidate.astype(np.float64) - ref) / (np.linalg.norm(ref) + 1e-12))
    if relerr > TOL:
        return False, relerr, "official_relative_tolerance"
    return True, relerr, None


def selected_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    result: list[tuple[str, str, Callable[[Problem], Solution]]] = []
    for arm, candidate_name in SELECTED.items():
        matches = [(name, fn) for name, fn in CANDIDATES_BY_ARM[arm] if name == candidate_name]
        if len(matches) != 1:
            raise RuntimeError(f"selected candidate resolution failed for {arm}/{candidate_name}")
        result.append((arm, candidate_name, matches[0][1]))
    if len(result) != 5:
        raise RuntimeError("expected five selected blind candidates")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0,{SHARDS})")

    raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    if git_blob_sha1(raw) != EXPECTED_GIT_BLOB_SHA1:
        raise RuntimeError("blind test manifest Git blob SHA-1 mismatch")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind records, got {len(rows)}")

    candidates = selected_candidates()
    evidence: list[dict[str, object]] = []
    for index, row in ((i, r) for i, r in enumerate(rows) if i % SHARDS == args.shard):
        problem = decode_problem(row.get("problem"))
        shift = index % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]
        if index % 2 == 0:
            expected, reference_s, reference_error = timed(reference, problem)
            candidate_results = [(arm, name, *timed(fn, problem)) for arm, name, fn in ordered]
            execution_order = "reference_first"
        else:
            candidate_results = [(arm, name, *timed(fn, problem)) for arm, name, fn in ordered]
            expected, reference_s, reference_error = timed(reference, problem)
            execution_order = "candidates_first"
        if expected is None or reference_s is None or reference_error is not None:
            raise RuntimeError(f"reference failed on blind record {index + 1}: {reference_error}")
        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            if candidate_error is None:
                valid, relerr, validation_error = validate(problem, proposed, expected)
            else:
                valid, relerr, validation_error = False, float("inf"), None
            speedup = reference_s / candidate_s if candidate_s and candidate_s > 0.0 else 0.0
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": valid and candidate_error is None,
                "failure_reason": candidate_error or validation_error,
                "relative_error_to_reference": relerr,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "matrix_shape": list(problem.shape),
                "matrix_dtype": str(problem.dtype),
                "test_manifest_name": MANIFEST,
                "test_manifest_git_blob_sha1": EXPECTED_GIT_BLOB_SHA1,
                "test_manifest_sha256": manifest_sha256,
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
            })
            print(f"[{index+1}/100] {arm}/{name} valid={valid and candidate_error is None} speedup={speedup:.3f} relerr={relerr:.3e}", flush=True)
        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
