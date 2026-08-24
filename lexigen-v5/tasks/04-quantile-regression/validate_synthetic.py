from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import QuantileRegressor

from candidates import CANDIDATES_BY_ARM, PROVENANCE


def make_problem(n: int, seed: int, quantile: float, fit_intercept: bool) -> dict[str, object]:
    rng = np.random.RandomState(seed)
    n_samples = max(4, n)
    n_features = max(1, n // 3)
    X = rng.randn(n_samples, n_features)
    w = rng.randn(n_features)
    intercept = rng.randn() * 0.5 if fit_intercept else 0.0
    y = X @ w + intercept + rng.randn(n_samples) * 0.3
    return {
        "X": X.tolist(),
        "y": y.tolist(),
        "quantile": float(quantile),
        "fit_intercept": bool(fit_intercept),
    }


def reference(problem: dict[str, object]) -> dict[str, object]:
    X = np.asarray(problem["X"], dtype=float)
    y = np.asarray(problem["y"], dtype=float)
    model = QuantileRegressor(
        quantile=float(problem["quantile"]),
        alpha=0.0,
        fit_intercept=bool(problem["fit_intercept"]),
        solver="highs",
    )
    model.fit(X, y)
    intercept = float(model.intercept_) if bool(problem["fit_intercept"]) else 0.0
    return {
        "coef": np.asarray(model.coef_, dtype=float),
        "intercept": np.asarray([intercept], dtype=float),
        "predictions": np.asarray(model.predict(X), dtype=float),
    }


def valid(proposed: dict[str, object], expected: dict[str, object]) -> tuple[bool, str | None, float]:
    try:
        coef = np.asarray(proposed["coef"], dtype=float)
        inter = np.asarray(proposed["intercept"], dtype=float)
        preds = np.asarray(proposed["predictions"], dtype=float)
    except Exception as exc:
        return False, f"shape_or_parse:{type(exc).__name__}:{exc}", float("inf")
    if coef.shape != expected["coef"].shape or inter.shape != expected["intercept"].shape or preds.shape != expected["predictions"].shape:
        return False, "shape_mismatch", float("inf")
    err = max(
        float(np.max(np.abs(coef - expected["coef"]))) if coef.size else 0.0,
        float(np.max(np.abs(inter - expected["intercept"]))) if inter.size else 0.0,
        float(np.max(np.abs(preds - expected["predictions"]))) if preds.size else 0.0,
    )
    ok = (
        np.allclose(coef, expected["coef"], atol=1e-5)
        and np.allclose(inter, expected["intercept"], atol=1e-5)
        and np.allclose(preds, expected["predictions"], atol=1e-5)
    )
    return bool(ok), None if ok else "authoritative_tolerance_mismatch", err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [
        (8, 11, 0.50, True),
        (12, 23, 0.25, True),
        (20, 37, 0.75, True),
        (30, 41, 0.50, False),
        (50, 53, 0.20, True),
        (80, 67, 0.80, False),
    ]
    rows: list[dict[str, object]] = []
    failed = 0
    for case_index, (n, seed, q, fit) in enumerate(cases, 1):
        problem = make_problem(n, seed, q, fit)
        start = time.perf_counter()
        expected = reference(problem)
        reference_s = time.perf_counter() - start
        for arm, candidates in CANDIDATES_BY_ARM.items():
            for candidate_name, fn in candidates:
                try:
                    cstart = time.perf_counter()
                    proposed = fn(problem)
                    candidate_s = time.perf_counter() - cstart
                    ok, reason, max_abs_error = valid(proposed, expected)
                except Exception as exc:
                    candidate_s = None
                    ok, reason, max_abs_error = False, f"exception:{type(exc).__name__}:{exc}", float("inf")
                if not ok:
                    failed += 1
                rows.append({
                    "case": case_index,
                    "n": n,
                    "seed": seed,
                    "quantile": q,
                    "fit_intercept": fit,
                    "arm": arm,
                    "candidate": candidate_name,
                    "valid": ok,
                    "failure_reason": reason,
                    "max_abs_error": max_abs_error,
                    "reference_s": reference_s,
                    "candidate_s": candidate_s,
                    "diagnostic_speedup": (reference_s / candidate_s) if candidate_s and candidate_s > 0 else None,
                    "diagnostic_only_not_used_for_selection": True,
                })
    total = len(rows)
    if total != 180:
        raise RuntimeError(f"expected 180 synthetic checks, got {total}")
    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 4,
        "task": "quantile_regression",
        "stage": "synthetic_r1",
        "status": "passed" if failed == 0 else "failed",
        "total_checks": total,
        "passed_checks": total - failed,
        "failed_checks": failed,
        "case_count": len(cases),
        "candidate_count": sum(len(v) for v in CANDIDATES_BY_ARM.values()),
        "provenance": PROVENANCE,
        "official_training_manifest_opened": False,
        "official_training_payloads_opened": 0,
        "official_test_manifest_opened": False,
        "official_test_payloads_opened": 0,
        "diagnostic_synthetic_timing_used_for_candidate_selection": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status","total_checks","passed_checks","failed_checks","case_count","candidate_count")}, indent=2), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
