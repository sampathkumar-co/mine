from __future__ import annotations

import argparse
import json
from pathlib import Path

import cvxpy as cp
import numpy as np

from candidates import ALL_CANDIDATES, Problem, Solution

FIDELITY_EPS = 1e-5
OPTIMALITY_FACTOR = 1.01


def mode2(value: np.ndarray) -> np.ndarray:
    return np.transpose(value, (1, 0, 2)).reshape(value.shape[1], value.shape[0] * value.shape[2])


def mode3(value: np.ndarray) -> np.ndarray:
    return np.transpose(value, (2, 0, 1)).reshape(value.shape[2], value.shape[0] * value.shape[1])


def official_reference(problem: Problem) -> Solution:
    observed = np.asarray(problem["tensor"], dtype=np.float64)
    mask = np.asarray(problem["mask"], dtype=bool)
    d1, d2, d3 = observed.shape
    u1 = observed.reshape(d1, d2 * d3)
    m1 = mask.reshape(d1, d2 * d3)
    u2 = mode2(observed)
    m2 = mode2(mask)
    u3 = mode3(observed)
    m3 = mode3(mask)
    x1 = cp.Variable(u1.shape)
    x2 = cp.Variable(u2.shape)
    x3 = cp.Variable(u3.shape)
    objective = cp.Minimize(cp.norm(x1, "nuc") + cp.norm(x2, "nuc") + cp.norm(x3, "nuc"))
    constraints = [
        cp.multiply(x1, m1) == cp.multiply(u1, m1),
        cp.multiply(x2, m2) == cp.multiply(u2, m2),
        cp.multiply(x3, m3) == cp.multiply(u3, m3),
    ]
    p = cp.Problem(objective, constraints)
    p.solve()
    if p.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x1.value is None:
        raise RuntimeError(f"synthetic reference failed: {p.status}")
    return {"completed_tensor": np.asarray(x1.value, dtype=np.float64).reshape(observed.shape).tolist()}


def nuclear_sum(tensor: np.ndarray) -> float:
    d1, d2, d3 = tensor.shape
    return float(
        np.linalg.norm(tensor.reshape(d1, d2 * d3), ord="nuc")
        + np.linalg.norm(mode2(tensor), ord="nuc")
        + np.linalg.norm(mode3(tensor), ord="nuc")
    )


def validate(problem: Problem, solution: Solution, reference: Solution) -> tuple[bool, str | None, float, float]:
    observed = np.asarray(problem["tensor"], dtype=np.float64)
    mask = np.asarray(problem["mask"], dtype=bool)
    raw = solution.get("completed_tensor")
    if not isinstance(raw, list) or not raw:
        return False, "empty_or_missing", float("inf"), float("inf")
    candidate = np.asarray(raw, dtype=np.float64)
    if candidate.shape != observed.shape or not np.all(np.isfinite(candidate)):
        return False, "shape_or_nonfinite", float("inf"), float("inf")
    fidelity = float(np.max(np.abs(candidate[mask] - observed[mask]))) if mask.any() else 0.0
    if fidelity > FIDELITY_EPS:
        return False, "fidelity", fidelity, float("inf")
    ref = np.asarray(reference["completed_tensor"], dtype=np.float64)
    candidate_nuc = nuclear_sum(candidate)
    ref_nuc = nuclear_sum(ref)
    if candidate_nuc > ref_nuc * OPTIMALITY_FACTOR + 1e-7:
        return False, "nuclear_norm", fidelity, candidate_nuc / max(ref_nuc, 1e-12)
    return True, None, fidelity, candidate_nuc / max(ref_nuc, 1e-12)


def low_rank(shape: tuple[int, int, int], rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = [rng.standard_normal((d, rank)) for d in shape]
    result = np.zeros(shape, dtype=np.float64)
    for r in range(rank):
        result += factors[0][:, r, None, None] * factors[1][None, :, r, None] * factors[2][None, None, :, r]
    return result


def problem_from(full: np.ndarray, mask: np.ndarray) -> Problem:
    observed = np.zeros_like(full)
    observed[mask] = full[mask]
    return {"tensor": observed.tolist(), "mask": mask.tolist()}


def cases() -> list[tuple[str, Problem]]:
    out: list[tuple[str, Problem]] = []
    a = low_rank((2, 3, 2), 1, 1)
    out.append(("fully_observed", problem_from(a, np.ones_like(a, dtype=bool))))
    out.append(("unobserved", problem_from(a, np.zeros_like(a, dtype=bool))))

    rng = np.random.default_rng(2)
    b = low_rank((2, 3, 2), 1, 3) + 0.01 * rng.standard_normal((2, 3, 2))
    checker = np.indices(b.shape).sum(axis=0) % 2 == 0
    out.append(("checker", problem_from(b, checker)))

    for i, (shape, rank, rate) in enumerate([
        ((3, 4, 2), 1, 0.35),
        ((3, 4, 2), 2, 0.45),
        ((4, 5, 3), 2, 0.35),
        ((4, 5, 3), 3, 0.50),
    ], start=10):
        full = low_rank(shape, rank, i)
        rr = np.random.default_rng(i + 100)
        mask = rr.random(shape) < rate
        if not mask.any():
            mask.flat[0] = True
        out.append((f"random_{shape[0]}_{rank}_{int(rate*100)}", problem_from(full, mask)))

    c = low_rank((3, 4, 2), 1, 31)
    slice_mask = np.zeros_like(c, dtype=bool)
    slice_mask[0, :, :] = True
    slice_mask[:, 0, :] = True
    out.append(("observed_slices", problem_from(c, slice_mask)))

    d = -low_rank((3, 4, 2), 2, 41)
    sparse = np.zeros_like(d, dtype=bool)
    sparse[0, 0, 0] = sparse[1, 2, 1] = sparse[2, 3, 0] = True
    out.append(("sparse_negative", problem_from(d, sparse)))

    e = low_rank((3, 4, 2), 2, 51)
    noisy = e + 0.01 * np.random.default_rng(52).standard_normal(e.shape)
    mask = np.random.default_rng(53).random(e.shape) < 0.4
    out.append(("noisy_low_rank", problem_from(noisy, mask)))
    assert len(out) == 10
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence: list[dict[str, object]] = []
    for case_name, problem in cases():
        reference = official_reference(problem)
        for arm, candidate_name, fn in ALL_CANDIDATES:
            try:
                proposed = fn(problem)
                valid, reason, fidelity, nuc_ratio = validate(problem, proposed, reference)
                error = None
            except Exception as exc:
                valid, reason, fidelity, nuc_ratio = False, "exception", float("inf"), float("inf")
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "case": case_name,
                "arm": arm,
                "candidate": candidate_name,
                "valid": valid,
                "failure_reason": error or reason,
                "fidelity_error": fidelity,
                "nuclear_ratio_to_reference": nuc_ratio,
                "candidate_executions": 1,
                "reference_execution_for_case": 1,
                "official_training_access": False,
            }
            evidence.append(row)
            print(f"{case_name}: {arm}/{candidate_name} valid={valid} nuc_ratio={nuc_ratio:.6f}", flush=True)
    passed = sum(bool(r["valid"]) for r in evidence)
    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 5,
        "task": "tensor_completion_3d",
        "stage": "synthetic_revision1",
        "candidate_count": len(ALL_CANDIDATES),
        "case_count": 10,
        "checks": len(evidence),
        "passed": passed,
        "failed": len(evidence) - passed,
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
