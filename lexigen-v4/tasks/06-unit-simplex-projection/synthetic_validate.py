from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from candidates import ALL_CANDIDATES, Problem, Solution


def oracle_projection(y: np.ndarray) -> np.ndarray:
    """Independent small-n oracle by exhaustive KKT support enumeration."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    if n == 0 or n > 12:
        raise ValueError("enumeration oracle supports 1..12 dimensions")
    found: list[np.ndarray] = []
    for bits in range(1, 1 << n):
        support = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool)
        theta = float((np.sum(y[support], dtype=np.float64) - 1.0) / np.count_nonzero(support))
        if np.all(y[support] > theta) and np.all(y[~support] <= theta):
            x = np.maximum(y - theta, 0.0)
            found.append(x)
    if not found:
        raise RuntimeError(f"no KKT support found for {y}")
    first = found[0]
    for other in found[1:]:
        if not np.allclose(first, other, rtol=0.0, atol=1e-12):
            raise RuntimeError("KKT enumeration produced non-unique projection")
    return first


def cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20260823)
    items = [
        ("single_negative", np.array([-3.25], dtype=np.float64)),
        ("equal_values", np.full(5, 0.2, dtype=np.float64)),
        ("already_simplex", np.array([0.1, 0.2, 0.7, 0.0, 0.0], dtype=np.float64)),
        ("dominant_coordinate", np.array([10.0, -2.0, -3.0, 0.1], dtype=np.float64)),
        ("all_negative", np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64)),
        ("threshold_ties", np.array([0.5, 0.5, -1.0, -1.0, -1.0], dtype=np.float64)),
        ("mixed_signs", np.array([1.2, 0.8, -0.1, 0.3, -2.0, 0.05], dtype=np.float64)),
        ("near_threshold", np.array([0.5000004, 0.4999996, 2e-8, -2e-8, -0.2], dtype=np.float64)),
        ("random_8", rng.standard_normal(8)),
        ("random_10", rng.standard_normal(10)),
    ]
    assert len(items) == 10
    return items


def validate(y: np.ndarray, raw: Solution, oracle: np.ndarray) -> tuple[bool, str | None, float, float]:
    if not isinstance(raw, dict) or "solution" not in raw:
        return False, "missing_solution", float("inf"), float("inf")
    proposed = np.asarray(raw["solution"], dtype=np.float64).reshape(-1)
    if proposed.shape != y.shape or not np.all(np.isfinite(proposed)):
        return False, "shape_or_nonfinite", float("inf"), float("inf")
    max_error = float(np.max(np.abs(proposed - oracle)))
    simplex_error = abs(float(np.sum(proposed, dtype=np.float64)) - 1.0)
    if np.any(proposed < -1e-12):
        return False, "negative_entry", max_error, simplex_error
    if simplex_error > 2e-10:
        return False, "simplex_sum", max_error, simplex_error
    if not np.allclose(proposed, oracle, rtol=2e-10, atol=2e-10):
        return False, "oracle_mismatch", max_error, simplex_error
    return True, None, max_error, simplex_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence: list[dict[str, object]] = []
    for case_name, y in cases():
        oracle = oracle_projection(y)
        for arm, candidate_name, fn in ALL_CANDIDATES:
            try:
                proposed = fn({"y": y.tolist()})
                valid, reason, max_error, simplex_error = validate(y, proposed, oracle)
                exception = None
            except Exception as exc:
                valid, reason, max_error, simplex_error = False, "exception", float("inf"), float("inf")
                exception = f"{type(exc).__name__}: {exc}"
            evidence.append({
                "case": case_name,
                "arm": arm,
                "candidate": candidate_name,
                "valid": valid,
                "failure_reason": exception or reason,
                "max_abs_error_to_oracle": max_error,
                "simplex_sum_error": simplex_error,
                "candidate_executions": 1,
                "official_training_access": False,
                "official_test_access": False,
            })
            print(f"{case_name}: {arm}/{candidate_name} valid={valid} maxerr={max_error:.3e}", flush=True)

    passed = sum(bool(row["valid"]) for row in evidence)
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 6,
        "task": "unit_simplex_projection",
        "stage": "synthetic_revision1",
        "oracle": "exhaustive_KKT_support_enumeration_independent_of_reference_sort",
        "candidate_count": len(ALL_CANDIDATES),
        "case_count": len(cases()),
        "checks": len(evidence),
        "passed": passed,
        "failed": len(evidence) - passed,
        "maximum_valid_max_abs_error": max(float(r["max_abs_error_to_oracle"]) for r in evidence if bool(r["valid"])),
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
