from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

Problem = dict[str, object]
Solution = list[float]
Candidate = Callable[[Problem], Solution]

REFERENCE_RTOL = 1.0e-8
REFERENCE_ATOL = 1.0e-8
VERIFIER_RTOL = 1.0e-5
VERIFIER_ATOL = 1.0e-8
# Frozen source-only risk policy: keep a 20x relative-tolerance safety margin
# inside the official verifier and never loosen the absolute tolerance.
RISK_RTOL = VERIFIER_RTOL / 20.0
RISK_ATOL = VERIFIER_ATOL


def _inputs(problem: Problem, *, contiguous: bool = False) -> tuple[np.ndarray, float, float, dict[str, float]]:
    y0 = np.asarray(problem["y0"], dtype=np.float64)
    if y0.shape != (2,) or not np.all(np.isfinite(y0)):
        raise ValueError("y0 must be a finite two-state vector")
    if contiguous:
        y0 = np.ascontiguousarray(y0)
    t0 = float(problem["t0"])
    t1 = float(problem["t1"])
    raw = problem["params"]
    if not isinstance(raw, dict):
        raise ValueError("params must be a mapping")
    params = {k: float(raw[k]) for k in ("a", "b", "c", "I")}
    if not all(np.isfinite(v) for v in params.values()) or not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("problem contains invalid numeric values")
    return y0, t0, t1, params


def _rhs(params: dict[str, float], *, dtype_specialized: bool) -> Callable[[float, np.ndarray], np.ndarray]:
    if dtype_specialized:
        a, b, c, current = (np.float32(params[k]) for k in ("a", "b", "c", "I"))

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            yy = np.asarray(y, dtype=np.float32)
            v, w = yy[0], yy[1]
            dv = v - (v * v * v) / np.float32(3.0) - w + current
            dw = a * (b * v - c * w)
            return np.stack((dv, dw), axis=0)

        return rhs

    a, b, c, current = (params[k] for k in ("a", "b", "c", "I"))

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        v, w = y[0], y[1]
        return np.stack((v - (v * v * v) / 3.0 - w + current, a * (b * v - c * w)), axis=0)

    return rhs


def _integrate(
    problem: Problem,
    *,
    contiguous: bool = False,
    dtype_specialized: bool = False,
    vectorized: bool = False,
    risk_aware: bool = False,
) -> Solution:
    y0, t0, t1, params = _inputs(problem, contiguous=contiguous)
    if dtype_specialized:
        y0 = y0.astype(np.float32, copy=False)
    sol = solve_ivp(
        _rhs(params, dtype_specialized=dtype_specialized),
        (t0, t1),
        y0,
        method="RK45",
        rtol=RISK_RTOL if risk_aware else REFERENCE_RTOL,
        atol=RISK_ATOL if risk_aware else REFERENCE_ATOL,
        vectorized=vectorized,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    final = np.asarray(sol.y[:, -1], dtype=np.float64)
    if final.shape != (2,) or not np.all(np.isfinite(final)):
        raise RuntimeError("integration produced invalid final state")
    return final.tolist()


def _zero_vector_risk(p: Problem) -> Solution:
    return _integrate(p, vectorized=True, risk_aware=True)


def _dtype_vector_risk(p: Problem) -> Solution:
    return _integrate(p, dtype_specialized=True, vectorized=True, risk_aware=True)


def _contiguous_vector_risk(p: Problem) -> Solution:
    return _integrate(p, contiguous=True, vectorized=True, risk_aware=True)


def _zero_dtype_vector(p: Problem) -> Solution:
    return _integrate(p, dtype_specialized=True, vectorized=True)


def _zero_dtype_risk(p: Problem) -> Solution:
    return _integrate(p, dtype_specialized=True, risk_aware=True)


def _zero_contiguous_vector(p: Problem) -> Solution:
    return _integrate(p, contiguous=True, vectorized=True)


def _vector_risk(p: Problem) -> Solution:
    return _integrate(p, vectorized=True, risk_aware=True)


def _zero_contiguous(p: Problem) -> Solution:
    return _integrate(p, contiguous=True)


def _risk(p: Problem) -> Solution:
    return _integrate(p, risk_aware=True)


def _dtype(p: Problem) -> Solution:
    return _integrate(p, dtype_specialized=True)


def _vector(p: Problem) -> Solution:
    return _integrate(p, vectorized=True)


def _zero(p: Problem) -> Solution:
    return _integrate(p)


def _contiguous(p: Problem) -> Solution:
    return _integrate(p, contiguous=True)


CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {
    "v4_full": [
        ("v4_zero_vector_risk", _zero_vector_risk),
        ("v4_dtype_vector_risk", _dtype_vector_risk),
        ("v4_contiguous_vector_risk", _contiguous_vector_risk),
        ("v4_zero_dtype_vector", _zero_dtype_vector),
        ("v4_zero_dtype_risk", _zero_dtype_risk),
        ("v4_zero_contiguous_vector", _zero_contiguous_vector),
    ],
    "v4_no_transfer": [
        ("no_transfer_zero_vector_risk", _zero_vector_risk),
        ("no_transfer_dtype_vector_risk", _dtype_vector_risk),
        ("no_transfer_contiguous_vector_risk", _contiguous_vector_risk),
        ("no_transfer_zero_dtype_vector", _zero_dtype_vector),
        ("no_transfer_zero_dtype_risk", _zero_dtype_risk),
        ("no_transfer_vector_risk", _vector_risk),
    ],
    "random_search": [
        ("random_zero_contiguous", _zero_contiguous),
        ("random_risk", _risk),
        ("random_dtype", _dtype),
        ("random_dtype_vector_risk", _dtype_vector_risk),
        ("random_contiguous_vector_risk", _contiguous_vector_risk),
        ("random_dtype_risk", _zero_dtype_risk),
    ],
    "template_synthesis": [
        ("template_vectorized_batch", _vector),
        ("template_risk_stage", _risk),
        ("template_zero_copy", _zero),
        ("template_dtype", _dtype),
        ("template_contiguous", _contiguous),
    ],
    "v3_compatible": [
        ("v3_vectorized_batch", _vector),
        ("v3_zero_copy_representation", _zero),
        ("v3_dtype_specialization", _dtype),
        ("v3_contiguous_layout", _contiguous),
    ],
}

ALL_CANDIDATES = [(arm, name, fn) for arm, items in CANDIDATES_BY_ARM.items() for name, fn in items]
assert len(ALL_CANDIDATES) == 27
