from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from candidates import CANDIDATES, CONDITION_THRESHOLD, Problem, Solution, diagnostic_ratio


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
    return np.asarray(solution["solution"], dtype=np.float64), elapsed


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
    modes = {
        "adaptive_cast_gesdd": "cast",
        "adaptive_product_gesdd": "product",
    }
    for case_name, problem in cases:
        expected, reference_s = execute(reference, problem)
        identity = np.eye(expected.shape[0], dtype=np.float64)
        for candidate_name, candidate in CANDIDATES.items():
            ratio = diagnostic_ratio(problem, modes[candidate_name])
            fallback_expected = bool(ratio <= CONDITION_THRESHOLD)
            actual, candidate_s = execute(candidate, problem)
            shape_valid = actual.shape == expected.shape
            finite = bool(np.all(np.isfinite(actual)))
            max_abs = float(np.max(np.abs(actual - expected), initial=0.0)) if shape_valid else float("inf")
            valid = bool(shape_valid and finite and np.allclose(actual, expected, atol=1e-5))
            orthogonality = (
                float(np.max(np.abs(actual.T @ actual - identity), initial=0.0))
                if shape_valid and finite
                else float("inf")
            )
            speedup = reference_s / candidate_s if candidate_s > 0.0 else 0.0
            rows.append({
                "case": case_name,
                "dimension": int(expected.shape[0]),
                "candidate": candidate_name,
                "valid": valid,
                "maximum_absolute_error": max_abs,
                "orthogonality_maximum_error": orthogonality,
                "condition_ratio": ratio,
                "condition_threshold": CONDITION_THRESHOLD,
                "fallback_expected": fallback_expected,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
            })
            print(
                f"{case_name} {candidate_name} valid={valid} fallback={fallback_expected} "
                f"ratio={ratio:.3e} speedup={speedup:.3f} max_abs={max_abs:.3e}",
                flush=True,
            )

    summaries: list[dict[str, object]] = []
    for candidate_name in sorted(CANDIDATES):
        selected = [row for row in rows if row["candidate"] == candidate_name]
        regular = [row for row in selected if row["case"] != "near_dependent_256"]
        irregular = [row for row in selected if row["case"] == "near_dependent_256"]
        summaries.append({
            "candidate": candidate_name,
            "cases": len(selected),
            "valid_cases": sum(bool(row["valid"]) for row in selected),
            "fallback_cases": sum(bool(row["fallback_expected"]) for row in selected),
            "regular_minimum_speedup": min(float(row["speedup"]) for row in regular),
            "regular_median_speedup": float(np.median([float(row["speedup"]) for row in regular])),
            "irregular_valid": all(bool(row["valid"]) for row in irregular),
            "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in selected),
            "maximum_orthogonality_error": max(float(row["orthogonality_maximum_error"]) for row in selected),
        })

    report = {
        "task": "procrustes",
        "candidate_revision": 2,
        "architecture": "condition_aware_precision_escalation",
        "scope": "synthetic_only_no_benchmark_access",
        "condition_threshold": CONDITION_THRESHOLD,
        "threshold_derivation": "4 * numpy.finfo(float32).eps",
        "rows": rows,
        "summaries": summaries,
        "candidate_source_sha256": sha256(Path("candidates.py")),
        "passes_exactness_gate": all(bool(row["valid"]) for row in rows),
        "irregular_variant_triggered_fallback": all(
            bool(row["fallback_expected"])
            for row in rows
            if row["case"] == "near_dependent_256"
        ),
        "regular_variants_avoided_fallback": all(
            not bool(row["fallback_expected"])
            for row in rows
            if row["case"] != "near_dependent_256"
        ),
        "benchmark_data_accessed": False,
        "synthetic_speed_is_diagnostic_only": True,
    }
    Path("synthetic-result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not (
        report["passes_exactness_gate"]
        and report["irregular_variant_triggered_fallback"]
        and report["regular_variants_avoided_fallback"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
