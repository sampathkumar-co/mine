from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from candidate import Problem, Solution, solve

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "outer_product"
MANIFEST = "outer_product_T100ms_n10630_size100_test.jsonl"
EXPECTED_LFS_SHA256 = "7c96c6cb4b391f0268625869217c50a5948eddd15a44d0d5f650e8f2adf04538"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
EXPECTED_LENGTH = 10630
ALLOWED_SHARDS = {2, 7}


def request_bytes(url: str) -> tuple[bytes, int]:
    delays = (0, 5, 15, 30, 60, 120)
    last_error: Exception | None = None
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LEXIGEN-v2-task5-infrastructure-completion"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read(), attempt
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last_error = exc
    raise RuntimeError(f"manifest download exhausted infrastructure retries: {last_error}")


def tuple_items(problem: object) -> list[dict[str, object]]:
    if not isinstance(problem, dict) or problem.get("__type__") != "tuple":
        raise TypeError("expected tagged tuple problem")
    for key in ("items", "values", "data", "value"):
        value = problem.get(key)
        if isinstance(value, list):
            return value
    raise TypeError(f"unsupported tuple keys: {sorted(problem)}")


def decode_array(value: object) -> np.ndarray:
    if not isinstance(value, dict) or value.get("__type__") != "ndarray_b64":
        raise TypeError("expected ndarray_b64")
    dtype = np.dtype(str(value["dtype"]))
    shape = tuple(int(dimension) for dimension in value["shape"])
    raw = base64.b64decode(str(value["data_b64"]), validate=True)
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError("embedded array byte length mismatch")
    return np.ascontiguousarray(np.frombuffer(raw, dtype=dtype).reshape(shape))


def decode_problem(row: dict[str, object]) -> Problem:
    items = tuple_items(row["problem"])
    if len(items) != 2:
        raise ValueError("expected exactly two vectors")
    left = decode_array(items[0])
    right = decode_array(items[1])
    if left.shape != (EXPECTED_LENGTH,) or right.shape != (EXPECTED_LENGTH,):
        raise ValueError(f"unexpected vector shapes: {left.shape}, {right.shape}")
    if left.dtype != np.float64 or right.dtype != np.float64:
        raise TypeError(f"unexpected dtypes: {left.dtype}, {right.dtype}")
    return left, right


def reference(problem: Problem) -> Solution:
    return np.outer(problem[0], problem[1])


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        return solution, time.perf_counter() - start, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def validate_exact(candidate: Solution | None, expected: Solution) -> tuple[bool, float, str | None]:
    if candidate is None:
        return False, float("inf"), "missing_candidate"
    array = np.asarray(candidate)
    if array.shape != expected.shape:
        return False, float("inf"), f"shape:{array.shape}"
    if array.dtype != np.float64:
        return False, float("inf"), f"dtype:{array.dtype}"
    for start in range(0, expected.shape[0], 128):
        stop = min(start + 128, expected.shape[0])
        candidate_block = array[start:stop]
        reference_block = expected[start:stop]
        if not np.array_equal(candidate_block, reference_block):
            maximum = float(np.max(np.abs(candidate_block - reference_block)))
            return False, maximum, "value_mismatch"
    return True, 0.0, None


def warm_up() -> None:
    problem = (
        np.linspace(0.0, 1.0, 257, dtype=np.float64),
        np.linspace(1.0, 0.0, 193, dtype=np.float64),
    )
    expected = reference(problem)
    candidate = solve(problem)
    valid, _, reason = validate_exact(candidate, expected)
    if not valid:
        raise RuntimeError(f"selected candidate synthetic exactness failure: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shard not in ALLOWED_SHARDS:
        raise ValueError(f"completion run is restricted to shards {sorted(ALLOWED_SHARDS)}")

    raw, download_attempts = request_bytes(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_LFS_SHA256:
        raise RuntimeError("blind manifest LFS content hash mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind rows, received {len(rows)}")

    warm_up()
    records: list[dict[str, object]] = []
    selected = [(index, row) for index, row in enumerate(rows) if index % 10 == args.shard]
    for index, row in selected:
        problem = decode_problem(row)
        if index % 2 == 0:
            reference_solution, reference_s, reference_error = timed(reference, problem)
            candidate_solution, candidate_s, candidate_error = timed(solve, problem)
            execution_order = "reference_first"
        else:
            candidate_solution, candidate_s, candidate_error = timed(solve, problem)
            reference_solution, reference_s, reference_error = timed(reference, problem)
            execution_order = "candidate_first"

        if reference_error is not None or reference_solution is None or reference_s is None:
            raise RuntimeError(f"reference failed on record {index + 1}: {reference_error}")
        valid, maximum_absolute_error, validation_reason = validate_exact(candidate_solution, reference_solution)
        failure_reason = candidate_error or validation_reason
        speedup = reference_s / candidate_s if candidate_s else 0.0
        records.append({
            "index": index + 1,
            "seed": int(row["seed"]),
            "candidate": "native_parallel8",
            "valid": valid,
            "failure_reason": failure_reason,
            "maximum_absolute_error": maximum_absolute_error,
            "candidate_s": candidate_s,
            "reference_s": reference_s,
            "speedup": speedup,
            "vector_length": EXPECTED_LENGTH,
            "output_bytes": EXPECTED_LENGTH * EXPECTED_LENGTH * 8,
            "shard": args.shard,
            "execution_order": execution_order,
            "candidate_executions": 1,
            "reference_executions": 1,
            "candidate_retries": 0,
            "manifest_download_attempts": download_attempts,
            "provenance": "infrastructure_completion_after_original_zero-byte HTTP 429",
        })
        print(
            f"[{index + 1}/100] native_parallel8 valid={valid} "
            f"candidate={candidate_s!s}s reference={reference_s:.6f}s speedup={speedup:.3f}",
            flush=True,
        )
        del candidate_solution, reference_solution, problem
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
