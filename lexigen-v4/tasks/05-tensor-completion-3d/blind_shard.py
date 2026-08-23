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
MANIFEST = "tensor_completion_3d_T100ms_n6_size100_test.jsonl"
EXPECTED_GIT_BLOB_SHA1 = "0bdbf8d4e6dd3897d50143dbf3778ca3e4e02f56"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
FIDELITY_EPS = 1e-5
OPTIMALITY_FACTOR = 1.01
SELECTED = {
    "v4_full": "v4_structure_refine_closed",
    "v4_no_transfer": "no_transfer_structure_refine_closed",
    "random_search": "random_zero_closed",
    "template_synthesis": "template_active_set",
    "v3_compatible": "v3_zero_copy_representation",
}


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task5-blind-r1"})
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


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def decode_problem(raw: object) -> Problem:
    if not isinstance(raw, dict) or set(raw) < {"tensor", "mask"}:
        raise RuntimeError("unexpected blind problem schema")
    tensor = np.asarray(raw["tensor"], dtype=np.float64)
    mask = np.asarray(raw["mask"], dtype=bool)
    if tensor.shape != (6, 7, 5) or mask.shape != tensor.shape or not np.all(np.isfinite(tensor)):
        raise RuntimeError(f"unexpected blind tensor/mask: {tensor.shape}/{mask.shape}")
    return {"tensor": tensor.tolist(), "mask": mask.tolist()}


def reference(problem: Problem) -> Solution:
    observed = np.asarray(problem["tensor"], dtype=np.float64)
    mask = np.asarray(problem["mask"], dtype=bool)
    d1, d2, d3 = observed.shape
    u1 = observed.reshape(d1, d2 * d3)
    m1 = mask.reshape(d1, d2 * d3)
    u2 = np.zeros((d2, d1 * d3))
    m2 = np.zeros((d2, d1 * d3), dtype=bool)
    u3 = np.zeros((d3, d1 * d2))
    m3 = np.zeros((d3, d1 * d2), dtype=bool)
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                u2[j, i * d3 + k] = observed[i, j, k]
                m2[j, i * d3 + k] = mask[i, j, k]
                u3[k, i * d2 + j] = observed[i, j, k]
                m3[k, i * d2 + j] = mask[i, j, k]
    x1 = cp.Variable(u1.shape)
    x2 = cp.Variable(u2.shape)
    x3 = cp.Variable(u3.shape)
    p = cp.Problem(
        cp.Minimize(cp.norm(x1, "nuc") + cp.norm(x2, "nuc") + cp.norm(x3, "nuc")),
        [
            cp.multiply(x1, m1) == cp.multiply(u1, m1),
            cp.multiply(x2, m2) == cp.multiply(u2, m2),
            cp.multiply(x3, m3) == cp.multiply(u3, m3),
        ],
    )
    try:
        p.solve()
    except cp.SolverError:
        return {"completed_tensor": []}
    if p.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x1.value is None:
        return {"completed_tensor": []}
    return {"completed_tensor": np.asarray(x1.value, dtype=np.float64).reshape(observed.shape).tolist()}


def mode2(v: np.ndarray) -> np.ndarray:
    return np.transpose(v, (1, 0, 2)).reshape(v.shape[1], v.shape[0] * v.shape[2])


def mode3(v: np.ndarray) -> np.ndarray:
    return np.transpose(v, (2, 0, 1)).reshape(v.shape[2], v.shape[0] * v.shape[1])


def nuclear_sum(v: np.ndarray) -> float:
    d1, d2, d3 = v.shape
    return float(np.linalg.norm(v.reshape(d1, d2 * d3), ord="nuc") + np.linalg.norm(mode2(v), ord="nuc") + np.linalg.norm(mode3(v), ord="nuc"))


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        out = fn(problem)
        return out, time.perf_counter() - start, None
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


def selected_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    result = []
    for arm, name in SELECTED.items():
        matches = [(n, fn) for n, fn in CANDIDATES_BY_ARM[arm] if n == name]
        if len(matches) != 1:
            raise RuntimeError(f"selected candidate not unique: {arm}/{name}")
        result.append((arm, name, matches[0][1]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0,{SHARDS})")

    raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    actual_oid = git_blob(raw)
    if actual_oid != EXPECTED_GIT_BLOB_SHA1:
        raise RuntimeError(f"blind manifest Git blob mismatch: {actual_oid} != {EXPECTED_GIT_BLOB_SHA1}")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind records, received {len(rows)}")

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
            raise RuntimeError(f"blind reference exception on record {index+1}: {reference_error}")
        ref_raw = expected.get("completed_tensor") if isinstance(expected, dict) else None
        if not isinstance(ref_raw, list) or not ref_raw:
            raise RuntimeError(f"blind reference solver failed on record {index+1}")
        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            if candidate_error is None:
                valid, validation_error, fidelity, nuc_ratio = validate(problem, proposed, expected)
            else:
                valid, validation_error, fidelity, nuc_ratio = False, None, float("inf"), float("inf")
            speedup = reference_s / candidate_s if candidate_s and candidate_s > 0 else 0.0
            evidence.append({
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
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
                "test_manifest_git_blob_sha1": actual_oid,
                "test_manifest_sha256": manifest_sha256,
            })
            print(f"[{index+1}/100] {arm}/{name} valid={valid and candidate_error is None} speedup={speedup:.3f}", flush=True)
        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
