from __future__ import annotations

import argparse
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

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "unit_simplex_projection"
MANIFEST = "unit_simplex_projection_T100ms_n982958_size100_train.jsonl"
EXPECTED_SHA256 = "fbb47d089bbe5ffeab8f930947e14aa3afc968f5159eeca427677dc0a9e39419"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
EXPECTED_N = 982958


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task6-train-r1"})
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


def decode_problem(raw: object) -> Problem:
    if not isinstance(raw, dict) or "y" not in raw:
        raise RuntimeError("unexpected training problem schema")
    ref = raw["y"]
    if not isinstance(ref, dict) or ref.get("__type__") != "ndarray_ref":
        raise RuntimeError("expected ndarray_ref y in official Task 6 training manifest")
    relative = str(ref.get("npy_path", ""))
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise RuntimeError(f"unsafe or missing training npy_path: {relative}")
    payload = fetch(f"{BASE}/{relative}?download=true")
    y = np.load(io.BytesIO(payload), allow_pickle=False)
    y = np.asarray(y)
    if y.shape != (EXPECTED_N,) or y.dtype != np.float64 or not np.all(np.isfinite(y)):
        raise RuntimeError(f"unexpected training y sidecar: shape={y.shape} dtype={y.dtype}")
    return {"y": y}


def reference(problem: Problem) -> Solution:
    y = np.array(problem.get("y"))
    y = y.flatten()
    n = len(y)
    sorted_y = np.sort(y)[::-1]
    cumsum_y = np.cumsum(sorted_y) - 1
    rho = np.where(sorted_y > cumsum_y / np.arange(1, n + 1))[0][-1]
    theta = cumsum_y[rho] / (rho + 1)
    x = np.maximum(y - theta, 0)
    return {"solution": x}


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        result = fn(problem)
        elapsed = time.perf_counter() - start
        return result, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def validate(proposed: Solution | None, expected: Solution) -> tuple[bool, str | None, float, float]:
    if not isinstance(proposed, dict) or "solution" not in proposed:
        return False, "missing_solution", float("inf"), float("inf")
    raw = proposed["solution"]
    try:
        candidate = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception:
        return False, "solution_decode", float("inf"), float("inf")
    ref = np.asarray(expected["solution"], dtype=np.float64).reshape(-1)
    if candidate.shape != ref.shape or candidate.shape != (EXPECTED_N,) or not np.all(np.isfinite(candidate)):
        return False, "shape_or_nonfinite", float("inf"), float("inf")
    max_error = float(np.max(np.abs(candidate - ref)))
    simplex_error = abs(float(np.sum(candidate, dtype=np.float64)) - 1.0)
    if not np.allclose(candidate, ref, atol=1e-6):
        return False, "reference_mismatch", max_error, simplex_error
    return True, None, max_error, simplex_error


def flattened_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    result = [(arm, name, fn) for arm, items in CANDIDATES_BY_ARM.items() for name, fn in items]
    if len(result) != 28:
        raise RuntimeError(f"expected 28 frozen candidates, got {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0,{SHARDS})")

    raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("training manifest SHA-256 mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, got {len(rows)}")

    candidates = flattened_candidates()
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
            raise RuntimeError(f"reference execution exception on record {index+1}: {reference_error}")

        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            if candidate_error is None:
                valid, validation_error, max_error, simplex_error = validate(proposed, expected)
            else:
                valid, validation_error, max_error, simplex_error = False, None, float("inf"), float("inf")
            speedup = reference_s / candidate_s if candidate_s and candidate_s > 0.0 else 0.0
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": valid and candidate_error is None,
                "failure_reason": candidate_error or validation_error,
                "max_abs_error_to_reference": max_error,
                "simplex_sum_error": simplex_error,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "vector_length": EXPECTED_N,
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
            })
            print(f"[{index+1}/100] {arm}/{name} valid={valid and candidate_error is None} speedup={speedup:.3f}", flush=True)

        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
