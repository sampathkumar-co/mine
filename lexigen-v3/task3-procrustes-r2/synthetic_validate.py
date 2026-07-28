from __future__ import annotations

import hashlib
import json
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


def make_case(name: str, n: int, seed: int, conditioning: str) -> tuple[str, Problem]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n), dtype=np.float64)
    b = rng.standard_normal((n, n), dtype=np.float64)
    if conditioning == "column_scaled":
        scales = np.logspace(0.0, -3.0, n, dtype=np.float64)
        a = a * scales[None, :]
        b = b * scales[::-1][None, :]
    elif conditioning == "near_dependent":
        a[:, -1] = a[:, 0] + 1e-3 * rng.standard_normal(n)
        b[:, -1] = b[:, 0] - 1e-3 * rng.standard_normal(n)
    elif conditioning != "normal":
        raise ValueError(f"unknown conditioning {conditioning}")
    return name, {"A": a, "B": b}


def execute(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[np.ndarray, float]:
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        solution = fn(problem)
        elapsed = time.perf_counter() - start
    matrix = np.asarray(solution["solution"], dtype=np.float64)
    return matrix, elapsed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    warm = {"A": np.eye(16), "B": np.eye(16)}
    reference(warm)
    for candidate in CANDIDATES.values():
        candidate(warm)

    cases = [
        make_case("normal_128", 128, 3101, "normal"),
        make_case("normal_256", 256, 3102, "normal"),
        make_case("normal_585", 585, 3103, "normal"),
        make_case("column_scaled_256", 256, 3104, "column_scaled"),
        make_case("near_dependent_256", 256, 3105, "near_dependent"),
    ]
    rows: list[dict[str, object]] = []
    for case_name, problem in cases:
        expected, reference_s = execute(reference, problem)
        n = expected.shape[0]
        identity = np.eye(n, dtype=np.float64)
        for candidate_name, candidate in CANDIDATES.items():
            actual, candidate_s = execute(candidate, problem)
            shape_valid = actual.shape == expected.shape
            finite = bool(np.all(np.isfinite(actual)))
            max_abs = float(np.max(np.abs(actual - expected), initial=0.0)) if shape_valid else float("inf")
            allclose = bool(shape_valid and finite and np.allclose(actual, expected, atol=1e-5))
            orthogonality = (
                float(np.max(np.abs(actual.T @ actual - identity), initial=0.0))
                if shape_valid and finite
                else float("inf")
            )
            speedup = reference_s / candidate_s if candidate_s > 0.0 else 0.0
            row = {
                "case": case_name,
                "dimension": n,
                "candidate": candidate_name,
                "valid": allclose,
                "shape_valid": shape_valid,
                "finite": finite,
                "maximum_absolute_error": max_abs,
                "orthogonality_maximum_error": orthogonality,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
            }
            rows.append(row)
            print(
                f"{case_name} {candidate_name} valid={allclose} "
                f"speedup={speedup:.3f} max_abs={max_abs:.3e} ortho={orthogonality:.3e}",
                flush=True,
            )

    candidate_names = sorted(CANDIDATES)
    summaries: list[dict[str, object]] = []
    for candidate_name in candidate_names:
        selected = [row for row in rows if row["candidate"] == candidate_name]
        summaries.append({
            "candidate": candidate_name,
            "cases": len(selected),
            "valid_cases": sum(bool(row["valid"]) for row in selected),
            "minimum_speedup": min(float(row["speedup"]) for row in selected),
            "median_speedup": float(np.median([float(row["speedup"]) for row in selected])),
            "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in selected),
            "maximum_orthogonality_error": max(float(row["orthogonality_maximum_error"]) for row in selected),
        })
    report = {
        "task": "procrustes",
        "candidate_revision": 2,
        "scope": "synthetic_only_no_benchmark_access",
        "environment_contract": {
            "python": "3.12",
            "threadpool_limit": 1,
            "numpy": "1.26.4",
            "scipy": "1.15.3",
        },
        "acceptance": {
            "all_cases_valid": True,
            "absolute_tolerance": 1e-5,
            "synthetic_speed_is_diagnostic_only": True,
        },
        "rows": rows,
        "summaries": summaries,
        "candidate_source_sha256": sha256(Path("candidates.py")),
        "passes_exactness_gate": all(bool(row["valid"]) for row in rows),
        "benchmark_data_accessed": False,
    }
    Path("synthetic-result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passes_exactness_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
