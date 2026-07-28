from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from candidates import CANDIDATES, Problem, Solution


def reference(problem: Problem) -> Solution:
    a = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["B"], dtype=np.float64)
    product = b @ a.T
    u, _, vh = np.linalg.svd(product, full_matrices=False)
    return {"solution": (u @ vh).tolist()}


def make_case(name: str, n: int, seed: int, mode: str) -> tuple[str, Problem]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n), dtype=np.float64)
    b = rng.standard_normal((n, n), dtype=np.float64)
    if mode == "column_scaled":
        scales = np.logspace(0.0, -3.0, n, dtype=np.float64)
        a = a * scales[None, :]
        b = b * scales[::-1][None, :]
    elif mode == "near_dependent":
        a[:, -1] = a[:, 0] + 1e-3 * rng.standard_normal(n)
        b[:, -1] = b[:, 0] - 1e-3 * rng.standard_normal(n)
    elif mode != "normal":
        raise ValueError(mode)
    return name, {"A": a, "B": b}


def execute(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[np.ndarray, float]:
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        solution = fn(problem)
        elapsed = time.perf_counter() - start
    return np.asarray(solution["solution"], dtype=np.float64), elapsed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    warm = {"A": np.eye(16), "B": np.eye(16)}
    reference(warm)
    for candidate in CANDIDATES.values():
        candidate(warm)

    cases = [make_case(f"normal_585_seed_{seed}", 585, seed, "normal") for seed in range(4001, 4013)]
    cases.extend([
        make_case("column_scaled_256", 256, 4101, "column_scaled"),
        make_case("near_dependent_256", 256, 4102, "near_dependent"),
    ])

    rows: list[dict[str, object]] = []
    for case_index, (case_name, problem) in enumerate(cases):
        expected, reference_s = execute(reference, problem)
        identity = np.eye(expected.shape[0], dtype=np.float64)
        names = list(CANDIDATES)
        rotation = case_index % len(names)
        names = names[rotation:] + names[:rotation]
        for candidate_name in names:
            actual, candidate_s = execute(CANDIDATES[candidate_name], problem)
            shape_valid = actual.shape == expected.shape
            finite = bool(np.all(np.isfinite(actual)))
            maximum_absolute_error = (
                float(np.max(np.abs(actual - expected), initial=0.0))
                if shape_valid
                else float("inf")
            )
            valid = bool(shape_valid and finite and np.allclose(actual, expected, atol=1e-5))
            orthogonality_error = (
                float(np.max(np.abs(actual.T @ actual - identity), initial=0.0))
                if shape_valid and finite
                else float("inf")
            )
            speedup = reference_s / candidate_s if candidate_s > 0.0 else 0.0
            rows.append({
                "case": case_name,
                "candidate": candidate_name,
                "dimension": int(expected.shape[0]),
                "valid": valid,
                "maximum_absolute_error": maximum_absolute_error,
                "orthogonality_maximum_error": orthogonality_error,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
            })
            print(
                f"{case_name} {candidate_name} valid={valid} speedup={speedup:.3f} "
                f"max_abs={maximum_absolute_error:.3e}",
                flush=True,
            )

    summaries = []
    for candidate_name in sorted(CANDIDATES):
        selected = [row for row in rows if row["candidate"] == candidate_name]
        speedups = [float(row["speedup"]) for row in selected]
        summaries.append({
            "candidate": candidate_name,
            "cases": len(selected),
            "valid_cases": sum(bool(row["valid"]) for row in selected),
            "minimum_speedup": min(speedups),
            "median_speedup": statistics.median(speedups),
            "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in selected),
            "maximum_orthogonality_error": max(float(row["orthogonality_maximum_error"]) for row in selected),
        })

    report = {
        "task": "procrustes",
        "candidate_revision": 3,
        "architecture": "mixed_precision_low_singular_subspace_refinement",
        "scope": "synthetic_only_no_benchmark_access",
        "rows": rows,
        "summaries": summaries,
        "candidate_source_sha256": sha256(Path("candidates.py")),
        "passes_exactness_gate": all(bool(row["valid"]) for row in rows),
        "benchmark_data_accessed": False,
        "synthetic_speed_is_diagnostic_only": True,
    }
    Path("synthetic-result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passes_exactness_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
