from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from candidates import CANDIDATES, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "outer_product"
MANIFEST = "outer_product_T100ms_n10630_size100_train.jsonl"
EXPECTED_LFS_SHA256 = "a910e76b4058137a8d69213d72541a6ea55410a494c69f52058b63b22b020372"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
EXPECTED_LENGTH = 10630


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v2-task5-train"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


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
    array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return np.ascontiguousarray(array)


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
        elapsed = time.perf_counter() - start
        return solution, elapsed, None
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
    maximum_absolute_error = 0.0
    block_rows = 128
    for start in range(0, expected.shape[0], block_rows):
        stop = min(start + block_rows, expected.shape[0])
        candidate_block = array[start:stop]
        reference_block = expected[start:stop]
        if not np.array_equal(candidate_block, reference_block):
            difference = np.abs(candidate_block - reference_block)
            block_maximum = float(np.max(difference))
            if block_maximum > maximum_absolute_error:
                maximum_absolute_error = block_maximum
            return False, maximum_absolute_error, "value_mismatch"
    return True, maximum_absolute_error, None


def warm_up() -> None:
    left = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    right = np.linspace(1.0, 0.0, 193, dtype=np.float64)
    problem = (left, right)
    expected = reference(problem)
    for name, candidate in CANDIDATES.items():
        solution = candidate(problem)
        valid, _, reason = validate_exact(solution, expected)
        if not valid:
            raise RuntimeError(f"{name} synthetic exactness failure: {reason}")


def record_result(
    records: list[dict[str, object]],
    *,
    index: int,
    seed: int,
    shard: int,
    name: str,
    candidate_solution: Solution | None,
    candidate_s: float | None,
    candidate_error: str | None,
    reference_solution: Solution,
    reference_s: float,
    execution_order: str,
) -> None:
    valid, maximum_absolute_error, validation_reason = validate_exact(
        candidate_solution,
        reference_solution,
    )
    failure_reason = candidate_error or validation_reason
    speedup = reference_s / candidate_s if candidate_s else 0.0
    record = {
        "index": index + 1,
        "seed": seed,
        "candidate": name,
        "valid": valid,
        "failure_reason": failure_reason,
        "maximum_absolute_error": maximum_absolute_error,
        "candidate_s": candidate_s,
        "reference_s": reference_s,
        "speedup": speedup,
        "vector_length": EXPECTED_LENGTH,
        "output_bytes": EXPECTED_LENGTH * EXPECTED_LENGTH * 8,
        "shard": shard,
        "execution_order": execution_order,
    }
    records.append(record)
    print(
        f"[{index + 1}/100] {name} valid={valid} "
        f"candidate={candidate_s!s}s reference={reference_s:.6f}s "
        f"speedup={speedup:.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    raw = request_bytes(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_LFS_SHA256:
        raise RuntimeError("training manifest LFS content hash mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 rows, received {len(rows)}")

    warm_up()
    records: list[dict[str, object]] = []
    selected = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    for index, row in selected:
        problem = decode_problem(row)
        names = list(CANDIDATES)
        rotation = index % len(names)
        names = names[rotation:] + names[:rotation]
        if index % 2:
            names.reverse()

        if index % 2 == 0:
            reference_solution, reference_s, reference_error = timed(reference, problem)
            if reference_error is not None or reference_solution is None or reference_s is None:
                raise RuntimeError(f"reference failed on record {index + 1}: {reference_error}")
            for name in names:
                candidate_solution, candidate_s, candidate_error = timed(CANDIDATES[name], problem)
                record_result(
                    records,
                    index=index,
                    seed=int(row["seed"]),
                    shard=args.shard,
                    name=name,
                    candidate_solution=candidate_solution,
                    candidate_s=candidate_s,
                    candidate_error=candidate_error,
                    reference_solution=reference_solution,
                    reference_s=reference_s,
                    execution_order="reference_first",
                )
                del candidate_solution
                gc.collect()
        else:
            first_name = names[0]
            first_solution, first_s, first_error = timed(CANDIDATES[first_name], problem)
            reference_solution, reference_s, reference_error = timed(reference, problem)
            if reference_error is not None or reference_solution is None or reference_s is None:
                raise RuntimeError(f"reference failed on record {index + 1}: {reference_error}")
            record_result(
                records,
                index=index,
                seed=int(row["seed"]),
                shard=args.shard,
                name=first_name,
                candidate_solution=first_solution,
                candidate_s=first_s,
                candidate_error=first_error,
                reference_solution=reference_solution,
                reference_s=reference_s,
                execution_order="candidate_first",
            )
            del first_solution
            gc.collect()
            for name in names[1:]:
                candidate_solution, candidate_s, candidate_error = timed(CANDIDATES[name], problem)
                record_result(
                    records,
                    index=index,
                    seed=int(row["seed"]),
                    shard=args.shard,
                    name=name,
                    candidate_solution=candidate_solution,
                    candidate_s=candidate_s,
                    candidate_error=candidate_error,
                    reference_solution=reference_solution,
                    reference_s=reference_s,
                    execution_order="after_reference",
                )
                del candidate_solution
                gc.collect()

        del reference_solution
        del problem
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
