from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import cvxpy as cp
import numpy as np

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "tensor_completion_3d"
MANIFEST = "tensor_completion_3d_T100ms_n6_size100_train.jsonl"
EXPECTED_SHA256 = "9116ae1beb04139892ea8711f4f4bd7d58b66f555a23a5ba6fac4104e8ab1548"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
FIDELITY_EPS = 1e-5
OPTIMALITY_FACTOR = 1.01


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task5-train-r1"})
            with urllib.request.urlopen(request, timeout=240) as response:
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
    if not isinstance(raw, dict) or set(raw) < {"tensor", "mask"}:
        raise RuntimeError("unexpected training problem schema")
    tensor = np.asarray(raw["tensor"], dtype=np.float64)
    mask = np.asarray(raw["mask"], dtype=bool)
    if tensor.shape != (6, 7, 5) or mask.shape != tensor.shape or not np.all(np.isfinite(tensor)):
        raise RuntimeError(f"unexpected training tensor/mask: {tensor.shape}/{mask.shape}")
    return {"tensor": tensor.tolist(), "mask": mask.tolist()}


def reference(problem: Problem) -> Solution:
    observed_tensor = np.array(problem["tensor"])
    mask = np.array(problem["mask"])
    dim1, dim2, dim3 = observed_tensor.shape

    unfolding1 = observed_tensor.reshape(dim1, dim2 * dim3)
    mask1 = mask.reshape(dim1, dim2 * dim3)

    unfolding2 = np.zeros((dim2, dim1 * dim3))
    mask2 = np.zeros((dim2, dim1 * dim3), dtype=bool)
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                unfolding2[j, i * dim3 + k] = observed_tensor[i, j, k]
                mask2[j, i * dim3 + k] = mask[i, j, k]

    unfolding3 = np.zeros((dim3, dim1 * dim2))
    mask3 = np.zeros((dim3, dim1 * dim2), dtype=bool)
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                unfolding3[k, i * dim2 + j] = observed_tensor[i, j, k]
                mask3[k, i * dim2 + j] = mask[i, j, k]

    x1 = cp.Variable((dim1, dim2 * dim3))
    x2 = cp.Variable((dim2, dim1 * dim3))
    x3 = cp.Variable((dim3, dim1 * dim2))
    objective = cp.Minimize(cp.norm(x1, "nuc") + cp.norm(x2, "nuc") + cp.norm(x3, "nuc"))
    constraints = [
        cp.multiply(x1, mask1) == cp.multiply(unfolding1, mask1),
        cp.multiply(x2, mask2) == cp.multiply(unfolding2, mask2),
        cp.multiply(x3, mask3) == cp.multiply(unfolding3, mask3),
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except cp.SolverError:
        return {"completed_tensor": []}
    if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x1.value is None:
        return {"completed_tensor": []}
    return {"completed_tensor": np.asarray(x1.value, dtype=np.float64).reshape(observed_tensor.shape).tolist()}


def mode2(value: np.ndarray) -> np.ndarray:
    return np.transpose(value, (1, 0, 2)).reshape(value.shape[1], value.shape[0] * value.shape[2])


def mode3(value: np.ndarray) -> np.ndarray:
    return np.transpose(value, (2, 0, 1)).reshape(value.shape[2], value.shape[0] * value.shape[1])


def nuclear_sum(value: np.ndarray) -> float:
    d1, d2, d3 = value.shape
    return float(
        np.linalg.norm(value.reshape(d1, d2 * d3), ord="nuc")
        + np.linalg.norm(mode2(value), ord="nuc")
        + np.linalg.norm(mode3(value), ord="nuc")
    )


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        elapsed = time.perf_counter() - start
        return solution, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def validate(problem: Problem, proposed: Solution | None, expected: Solution) -> tuple[bool, str | None, float, float]:
    if not isinstance(proposed, dict):
        return False, "solution_not_dict", float("inf"), float("inf")
    raw = proposed.get("completed_tensor")
    if not isinstance(raw, list) or not raw:
        return False, "empty_or_missing", float("inf"), float("inf")
    observed = np.asarray(problem["tensor"], dtype=np.float64)
    mask = np.asarray(problem["mask"], dtype=bool)
    candidate = np.asarray(raw, dtype=np.float64)
    if candidate.shape != observed.shape or not np.all(np.isfinite(candidate)):
        return False, "shape_or_nonfinite", float("inf"), float("inf")
    fidelity = float(np.max(np.abs(candidate[mask] - observed[mask]))) if mask.any() else 0.0
    if fidelity > FIDELITY_EPS:
        return False, "fidelity", fidelity, float("inf")
    ref_raw = expected.get("completed_tensor")
    if not isinstance(ref_raw, list) or not ref_raw:
        return True, None, fidelity, 1.0
    ref = np.asarray(ref_raw, dtype=np.float64)
    candidate_nuc = nuclear_sum(candidate)
    ref_nuc = nuclear_sum(ref)
    ratio = candidate_nuc / max(ref_nuc, 1e-12)
    if not math.isfinite(ratio) or candidate_nuc > ref_nuc * OPTIMALITY_FACTOR:
        return False, "nuclear_norm", fidelity, ratio
    return True, None, fidelity, ratio


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
    selected_rows = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    evidence: list[dict[str, object]] = []

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
            raise RuntimeError(f"reference execution exception on record {index + 1}: {reference_error}")
        ref_raw = expected.get("completed_tensor") if isinstance(expected, dict) else None
        if not isinstance(ref_raw, list) or not ref_raw:
            raise RuntimeError(f"reference solver failed on record {index + 1}")

        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            if candidate_error is None:
                valid, validation_error, fidelity, nuc_ratio = validate(problem, proposed, expected)
            else:
                valid, validation_error, fidelity, nuc_ratio = False, None, float("inf"), float("inf")
            speedup = reference_s / candidate_s if candidate_s and candidate_s > 0.0 else 0.0
            record = {
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": valid and candidate_error is None,
                "failure_reason": candidate_error or validation_error,
                "fidelity_error": fidelity,
                "nuclear_ratio_to_reference": nuc_ratio,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "tensor_shape": [6, 7, 5],
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
            }
            evidence.append(record)
            print(
                f"[{index + 1}/100] {arm}/{name} valid={record['valid']} speedup={speedup:.3f} nuc_ratio={nuc_ratio:.6f}",
                flush=True,
            )

        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in evidence) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
