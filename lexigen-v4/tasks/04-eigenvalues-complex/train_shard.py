from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "eigenvalues_complex"
MANIFEST = "eigenvalues_complex_T100ms_n474_size100_train.jsonl"
EXPECTED_SHA256 = "4d1eb81b05e772d1238cd693dcb5c2463cac3521b3a4a79f5d9b3e2139c09270"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
TOL = 1e-6
EPSILON = 1e-12


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task4-train-r1"})
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


def decode_problem(problem_raw: object) -> Problem:
    value = problem_raw
    if isinstance(problem_raw, dict) and "__type__" not in problem_raw:
        typed = [v for v in problem_raw.values() if isinstance(v, dict) and str(v.get("__type__", "")).startswith("ndarray")]
        if len(typed) == 1:
            value = typed[0]
    if isinstance(value, list):
        arr = np.asarray(value, dtype=np.float64)
    elif isinstance(value, dict) and value.get("__type__") == "ndarray_ref":
        relative = str(value["npy_path"])
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe npy_path {relative}")
        raw = fetch(f"{BASE}/{relative}?download=true")
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    elif isinstance(value, dict) and value.get("__type__") == "ndarray_b64":
        import base64
        raw = base64.b64decode(str(value["data_b64"]).encode("ascii"), validate=True)
        arr = np.frombuffer(raw, dtype=np.dtype(str(value["dtype"]))).reshape(tuple(int(x) for x in value["shape"]))
    elif isinstance(value, dict) and value.get("__type__") == "ndarray":
        arr = np.asarray(value["data"], dtype=np.dtype(str(value.get("dtype", "float64"))))
    else:
        raise RuntimeError(f"unsupported training problem encoding: {type(value).__name__}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (474, 474) or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"unexpected training matrix shape/values: {arr.shape}")
    return arr


def reference(problem: Problem) -> Solution:
    eigenvalues = np.linalg.eig(problem)[0]
    return sorted((complex(x) for x in eigenvalues), key=lambda x: (-x.real, -x.imag))


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        elapsed = time.perf_counter() - start
        return solution, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def validate(problem: Problem, proposed: Solution | None, expected: Solution) -> tuple[bool, float, str | None]:
    n = problem.shape[0]
    if not isinstance(proposed, list) or len(proposed) != n:
        return False, float("inf"), "solution_shape_or_type"
    converted: list[complex] = []
    for value in proposed:
        try:
            z = complex(value)
        except Exception:
            return False, float("inf"), "non_complex_value"
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            return False, float("inf"), "nonfinite_value"
        converted.append(z)
    resorted = sorted(converted, key=lambda x: (-x.real, -x.imag))
    if any(abs(a - b) > 1e-12 for a, b in zip(converted, resorted)):
        return False, float("inf"), "not_sorted"
    maximum = 0.0
    for cand, exp in zip(converted, expected):
        rel = abs(cand - exp) / max(abs(exp), EPSILON)
        maximum = max(maximum, rel)
        if rel > TOL:
            return False, maximum, "relative_error"
    return True, maximum, None


def flattened_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    return [(arm, name, fn) for arm, items in CANDIDATES_BY_ARM.items() for name, fn in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("training manifest SHA-256 mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, received {len(rows)}")

    candidates = flattened_candidates()
    evidence: list[dict[str, object]] = []
    selected_rows = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    for index, row in selected_rows:
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
            raise RuntimeError(f"reference failed on record {index + 1}: {reference_error}")
        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            valid, max_rel, validation_error = validate(problem, proposed, expected)
            speedup = reference_s / candidate_s if candidate_s and candidate_s > 0.0 else 0.0
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": valid,
                "failure_reason": candidate_error or validation_error,
                "maximum_relative_error": max_rel,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "matrix_n": 474,
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
            })
            print(f"[{index + 1}/100] {arm}/{name} valid={valid} speedup={speedup:.3f} max_rel={max_rel:.3e}", flush=True)
        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
