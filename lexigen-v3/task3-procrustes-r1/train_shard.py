from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from candidates import CANDIDATES, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "procrustes"
MANIFEST = "procrustes_T100ms_n585_size100_train.jsonl"
EXPECTED_SHA256 = "61a55f5731fa9ba6e3b83b5cce05bb91f5d82edd81c1ea273ab8b7681bbef59f"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10


def request_bytes(url: str) -> bytes:
    last_error: Exception | None = None
    for delay in (0, 5, 15, 30, 60):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-procrustes-r1-train"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last_error = exc
    raise RuntimeError(f"download exhausted infrastructure retries before execution: {last_error}")


def reference(problem: Problem) -> Solution:
    a = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["B"], dtype=np.float64)
    product = b @ a.T
    u, _, vh = np.linalg.svd(product, full_matrices=False)
    return {"solution": (u @ vh).tolist()}


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            solution = fn(problem)
            elapsed = time.perf_counter() - start
        return solution, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def validate(solution: Solution | None, reference_solution: Solution) -> tuple[bool, float, str | None]:
    try:
        candidate = np.asarray(solution["solution"] if solution is not None else [], dtype=np.float64)
        expected = np.asarray(reference_solution["solution"], dtype=np.float64)
        if candidate.shape != expected.shape or not np.all(np.isfinite(candidate)):
            return False, float("inf"), "shape_or_finite"
        max_abs = float(np.max(np.abs(candidate - expected), initial=0.0))
        valid = bool(np.allclose(candidate, expected, atol=1e-5))
        return valid, max_abs, None if valid else "matrix_mismatch"
    except Exception as exc:
        return False, float("inf"), f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    raw = request_bytes(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("training manifest content hash mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, received {len(rows)}")

    records: list[dict[str, object]] = []
    selected = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for index, row in selected:
            problem_data = row["problem"]
            a_path = temporary / f"A-{index}.npy"
            b_path = temporary / f"B-{index}.npy"
            a_path.write_bytes(request_bytes(f"{BASE}/{problem_data['A']['npy_path']}?download=true"))
            b_path.write_bytes(request_bytes(f"{BASE}/{problem_data['B']['npy_path']}?download=true"))
            a = np.load(a_path, allow_pickle=False)
            b = np.load(b_path, allow_pickle=False)
            if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
                raise RuntimeError(f"invalid matrix dimensions on record {index + 1}: {a.shape}, {b.shape}")
            problem: Problem = {"A": a, "B": b}

            order = list(CANDIDATES)
            shift = index % len(order)
            order = order[shift:] + order[:shift]
            if index % 2 == 0:
                reference_solution, reference_s, reference_error = timed(reference, problem)
                candidate_results = [(name, *timed(CANDIDATES[name], problem)) for name in order]
                execution_order = "reference_first"
            else:
                candidate_results = [(name, *timed(CANDIDATES[name], problem)) for name in order]
                reference_solution, reference_s, reference_error = timed(reference, problem)
                execution_order = "candidates_first"
            if reference_solution is None or reference_s is None or reference_error is not None:
                raise RuntimeError(f"reference failed on record {index + 1}: {reference_error}")

            for name, solution, candidate_s, candidate_error in candidate_results:
                valid, max_abs_error, validation_reason = validate(solution, reference_solution)
                speedup = reference_s / candidate_s if candidate_s else 0.0
                records.append({
                    "index": index + 1,
                    "seed": int(row["seed"]),
                    "candidate": name,
                    "valid": valid,
                    "failure_reason": candidate_error or validation_reason,
                    "maximum_absolute_error": max_abs_error,
                    "candidate_s": candidate_s,
                    "reference_s": reference_s,
                    "speedup": speedup,
                    "dimension": int(a.shape[0]),
                    "shard": args.shard,
                    "execution_order": execution_order,
                })
                print(
                    f"[{index + 1}/100] {name} valid={valid} speedup={speedup:.3f} max_abs={max_abs_error:.3e}",
                    flush=True,
                )
            del a, b, problem, reference_solution, candidate_results
            a_path.unlink(missing_ok=True)
            b_path.unlink(missing_ok=True)
            gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
