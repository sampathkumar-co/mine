from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from candidates import ALL_CANDIDATES, Problem

VERIFIER_RTOL = 1.0e-5
VERIFIER_ATOL = 1.0e-8


def _rhs(params: dict[str, float]):
    a, b, c, current = (float(params[k]) for k in ("a", "b", "c", "I"))

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        v, w = y[0], y[1]
        return np.array([v - (v**3) / 3.0 - w + current, a * (b * v - c * w)], dtype=np.float64)

    return rhs


def _integrate(problem: Problem, *, method: str, rtol: float, atol: float) -> np.ndarray:
    sol = solve_ivp(
        _rhs(problem["params"]),
        (float(problem["t0"]), float(problem["t1"])),
        np.asarray(problem["y0"], dtype=np.float64),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.asarray(sol.y[:, -1], dtype=np.float64)


def oracle(problem: Problem) -> np.ndarray:
    dop = _integrate(problem, method="DOP853", rtol=1.0e-12, atol=1.0e-12)
    rk = _integrate(problem, method="RK45", rtol=1.0e-11, atol=1.0e-12)
    if not np.allclose(dop, rk, rtol=2.0e-9, atol=2.0e-10):
        raise RuntimeError(f"independent high-accuracy oracles disagree: DOP853={dop} RK45={rk}")
    return dop


def cases() -> list[tuple[str, Problem]]:
    return [
        ("center_n1", {"t0": 0.0, "t1": 100.0, "y0": [-1.0, -0.5], "params": {"a": 0.08, "b": 0.8, "c": 0.7, "I": 0.5}}),
        ("low_corner_n2", {"t0": 0.0, "t1": 200.0, "y0": [-0.9, -0.45], "params": {"a": 0.064, "b": 0.72, "c": 0.63, "I": 0.45}}),
        ("high_corner_n3", {"t0": 0.0, "t1": 300.0, "y0": [-1.1, -0.55], "params": {"a": 0.096, "b": 0.88, "c": 0.77, "I": 0.55}}),
        ("mixed_aI_n1", {"t0": 0.0, "t1": 100.0, "y0": [-1.08, -0.46], "params": {"a": 0.095, "b": 0.75, "c": 0.74, "I": 0.452}}),
        ("mixed_bc_n2", {"t0": 0.0, "t1": 200.0, "y0": [-0.92, -0.54], "params": {"a": 0.067, "b": 0.875, "c": 0.635, "I": 0.542}}),
        ("interior_n3", {"t0": 0.0, "t1": 300.0, "y0": [-1.035, -0.487], "params": {"a": 0.083, "b": 0.817, "c": 0.716, "I": 0.513}}),
    ]


def validate(raw: object, expected: np.ndarray) -> tuple[bool, str | None, float, float]:
    try:
        proposed = np.asarray(raw, dtype=np.float64)
    except Exception as exc:
        return False, f"conversion:{type(exc).__name__}", float("inf"), float("inf")
    if proposed.shape != (2,) or not np.all(np.isfinite(proposed)):
        return False, "shape_or_nonfinite", float("inf"), float("inf")
    abs_error = float(np.max(np.abs(proposed - expected)))
    scaled = np.abs(proposed - expected) / (VERIFIER_ATOL + VERIFIER_RTOL * np.abs(expected))
    max_scaled_error = float(np.max(scaled))
    if not np.allclose(proposed, expected, rtol=VERIFIER_RTOL, atol=VERIFIER_ATOL):
        return False, "verifier_tolerance_mismatch", abs_error, max_scaled_error
    return True, None, abs_error, max_scaled_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence: list[dict[str, object]] = []
    for case_name, problem in cases():
        expected = oracle(problem)
        for arm, candidate_name, fn in ALL_CANDIDATES:
            try:
                raw = fn(problem)
                valid, reason, abs_error, scaled_error = validate(raw, expected)
                exception = None
            except Exception as exc:
                valid, reason, abs_error, scaled_error = False, "exception", float("inf"), float("inf")
                exception = f"{type(exc).__name__}: {exc}"
            evidence.append({
                "case": case_name,
                "arm": arm,
                "candidate": candidate_name,
                "valid": valid,
                "failure_reason": exception or reason,
                "max_abs_error_to_oracle": abs_error,
                "max_verifier_scaled_error": scaled_error,
                "candidate_executions": 1,
                "official_training_access": False,
                "official_test_access": False,
            })
            print(f"{case_name}: {arm}/{candidate_name} valid={valid} scaled={scaled_error:.4g}", flush=True)

    passed = sum(bool(row["valid"]) for row in evidence)
    valid_rows = [row for row in evidence if bool(row["valid"])]
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 7,
        "task": "ode_fitzhughnagumo",
        "stage": "synthetic_revision1",
        "oracle": "DOP853_1e-12_crosschecked_RK45_1e-11",
        "candidate_count": len(ALL_CANDIDATES),
        "case_count": len(cases()),
        "checks": len(evidence),
        "passed": passed,
        "failed": len(evidence) - passed,
        "maximum_valid_max_abs_error": max(float(r["max_abs_error_to_oracle"]) for r in valid_rows) if valid_rows else None,
        "maximum_valid_verifier_scaled_error": max(float(r["max_verifier_scaled_error"]) for r in valid_rows) if valid_rows else None,
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
