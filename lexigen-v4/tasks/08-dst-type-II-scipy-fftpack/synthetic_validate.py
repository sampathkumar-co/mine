from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from candidates import ALL_CANDIDATES, Problem, Solution


def direct_dst2_type_ii(value: np.ndarray) -> np.ndarray:
    """Independent O(n^3) two-dimensional DST-II oracle for small synthetic matrices."""
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or x.size == 0:
        raise ValueError("oracle expects a nonempty square matrix")
    n = x.shape[0]
    rows = np.arange(n, dtype=np.float64)
    k = np.arange(1, n + 1, dtype=np.float64)[:, None]
    transform = 2.0 * np.sin(np.pi * k * (2.0 * rows[None, :] + 1.0) / (2.0 * n))
    return transform @ x @ transform.T


def cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20260823)
    return [
        ("single", np.array([[0.25]], dtype=np.float64)),
        ("signed_2", np.array([[-1.0, 0.5], [2.0, -0.25]], dtype=np.float64)),
        ("random_3", rng.normal(size=(3, 3))),
        ("ramp_4", np.linspace(-1.0, 1.0, 16, dtype=np.float64).reshape(4, 4)),
        ("random_5", rng.uniform(-2.0, 2.0, size=(5, 5))),
        ("random_6", rng.normal(scale=3.0, size=(6, 6))),
    ]


def validate(problem: Problem, raw: Solution, oracle: np.ndarray) -> tuple[bool, str | None, float]:
    proposed = np.asarray(raw)
    if proposed.shape != problem.shape or not np.all(np.isfinite(proposed)):
        return False, "shape_or_nonfinite", float("inf")
    relerr = float(np.linalg.norm(proposed.astype(np.float64) - oracle) / (np.linalg.norm(oracle) + 1e-12))
    if relerr > 1e-6:
        return False, "official_relative_tolerance", relerr
    return True, None, relerr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence: list[dict[str, object]] = []
    for case_name, problem in cases():
        oracle = direct_dst2_type_ii(problem)
        for arm, candidate_name, fn in ALL_CANDIDATES:
            try:
                proposed = fn(problem.copy())
                valid, reason, relerr = validate(problem, proposed, oracle)
                exception = None
            except Exception as exc:
                valid, reason, relerr = False, "exception", float("inf")
                exception = f"{type(exc).__name__}: {exc}"
            evidence.append({
                "case": case_name,
                "arm": arm,
                "candidate": candidate_name,
                "valid": valid,
                "failure_reason": exception or reason,
                "relative_error_to_direct_oracle": relerr,
                "candidate_executions": 1,
                "official_training_access": False,
                "official_test_access": False,
            })
            print(f"{case_name}: {arm}/{candidate_name} valid={valid} relerr={relerr:.3e}", flush=True)

    passed = sum(bool(row["valid"]) for row in evidence)
    finite_errors = [float(row["relative_error_to_direct_oracle"]) for row in evidence if bool(row["valid"])]
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 8,
        "task": "dst_type_II_scipy_fftpack",
        "stage": "synthetic_revision1",
        "oracle": "direct_separable_DST_II_formula_independent_of_SciPy_FFT_implementation",
        "candidate_count": len(ALL_CANDIDATES),
        "case_count": len(cases()),
        "checks": len(evidence),
        "passed": passed,
        "failed": len(evidence) - passed,
        "maximum_valid_relative_error": max(finite_errors) if finite_errors else None,
        "synthetic_gate_passed": passed == len(evidence),
        "training_manifest_opened": False,
        "training_payloads_opened": False,
        "test_manifest_opened": False,
        "test_payloads_opened": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "synthetic-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "synthetic-results.jsonl").write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in evidence) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["synthetic_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
