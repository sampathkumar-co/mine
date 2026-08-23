from __future__ import annotations

from typing import Callable

import numpy as np

Problem = dict[str, object]
Solution = dict[str, object]
Candidate = Callable[[Problem], Solution]


def _y(problem: Problem, *, dtype: np.dtype | type = np.float64, contiguous: bool = False) -> np.ndarray:
    value = np.asarray(problem.get("y"), dtype=dtype).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("y must be a nonempty finite vector")
    if contiguous:
        value = np.ascontiguousarray(value)
    return value


def _solution(x: np.ndarray) -> Solution:
    return {"solution": np.asarray(x, dtype=np.float64)}


def _sort_projection_from(y: np.ndarray) -> np.ndarray:
    ordered = np.sort(y)[::-1]
    cssv = np.cumsum(ordered, dtype=np.float64) - 1.0
    rho_idx = np.nonzero(ordered > cssv / np.arange(1, y.size + 1, dtype=np.float64))[0]
    if rho_idx.size == 0:
        raise RuntimeError("simplex threshold support unexpectedly empty")
    rho = int(rho_idx[-1])
    theta = float(cssv[rho] / (rho + 1))
    return np.maximum(y - theta, 0.0)


def _sort_projection(problem: Problem) -> Solution:
    return _solution(_sort_projection_from(_y(problem)))


def _sort_zero_copy(problem: Problem) -> Solution:
    return _solution(_sort_projection_from(_y(problem, contiguous=False)))


def _sort_contiguous(problem: Problem) -> Solution:
    return _solution(_sort_projection_from(_y(problem, contiguous=True)))


def _active_projection_from(y: np.ndarray, *, max_iterations: int | None = None) -> tuple[np.ndarray, float, bool]:
    active = y
    iterations = 0
    while True:
        theta = float((np.sum(active, dtype=np.float64) - 1.0) / active.size)
        reduced = active[active > theta]
        iterations += 1
        if reduced.size == active.size:
            x = np.maximum(y - theta, 0.0)
            return x, theta, True
        if reduced.size == 0:
            return _sort_projection_from(y), 0.0, False
        if max_iterations is not None and iterations >= max_iterations:
            return _sort_projection_from(y), 0.0, False
        active = reduced


def _kkt_certificate(y: np.ndarray, x: np.ndarray) -> bool:
    if x.shape != y.shape or not np.all(np.isfinite(x)) or np.any(x < -1e-12):
        return False
    if abs(float(np.sum(x, dtype=np.float64)) - 1.0) > 2e-9:
        return False
    support = x > 1e-12
    if not np.any(support):
        return False
    theta = float(np.mean(y[support] - x[support], dtype=np.float64))
    if np.max(np.abs((y[support] - x[support]) - theta)) > 2e-9:
        return False
    if np.any(y[~support] > theta + 2e-9):
        return False
    return True


def _active_exact(problem: Problem) -> Solution:
    y = _y(problem)
    x, _, _ = _active_projection_from(y)
    return _solution(x)


def _active_guarded(problem: Problem) -> Solution:
    y = _y(problem)
    x, _, converged = _active_projection_from(y, max_iterations=64)
    if not converged or not _kkt_certificate(y, x):
        x = _sort_projection_from(y)
    return _solution(x)


def _active_zero_copy(problem: Problem) -> Solution:
    y = _y(problem, contiguous=False)
    x, _, _ = _active_projection_from(y)
    return _solution(x)


def _active_contiguous(problem: Problem) -> Solution:
    y = _y(problem, contiguous=True)
    x, _, _ = _active_projection_from(y)
    return _solution(x)


def _dtype_active_guarded(problem: Problem) -> Solution:
    y64 = _y(problem)
    y32 = np.asarray(y64, dtype=np.float32)
    active = y32
    converged = False
    for _ in range(64):
        theta32 = np.float32((np.sum(active, dtype=np.float32) - np.float32(1.0)) / np.float32(active.size))
        reduced = active[active > theta32]
        if reduced.size == active.size:
            converged = True
            break
        if reduced.size == 0:
            break
        active = reduced
    if converged:
        # Float32 is used only to identify the support. Recompute the threshold
        # in float64, then require an exact KKT certificate before accepting it.
        theta64 = float((np.sum(y64[y32 > theta32], dtype=np.float64) - 1.0) / np.count_nonzero(y32 > theta32))
        x = np.maximum(y64 - theta64, 0.0)
        if _kkt_certificate(y64, x):
            return _solution(x)
    return _active_guarded(problem)


def _dtype_sort_refined(problem: Problem) -> Solution:
    y64 = _y(problem)
    y32 = np.asarray(y64, dtype=np.float32)
    ordered = np.sort(y32)[::-1]
    cssv = np.cumsum(ordered, dtype=np.float32) - np.float32(1.0)
    idx = np.nonzero(ordered > cssv / np.arange(1, y32.size + 1, dtype=np.float32))[0]
    if idx.size:
        rho = int(idx[-1])
        threshold32 = ordered[rho]
        support = y32 >= threshold32
        # Refine support in float64 and certify; ties/rounding fall back safely.
        if np.any(support):
            theta64 = float((np.sum(y64[support], dtype=np.float64) - 1.0) / np.count_nonzero(support))
            x = np.maximum(y64 - theta64, 0.0)
            if _kkt_certificate(y64, x):
                return _solution(x)
    return _sort_projection(problem)


CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {
    "v4_full": [
        ("v4_active_vector_risk", _active_guarded),
        ("v4_active_partition_vector", _active_exact),
        ("v4_active_partition_risk", _active_guarded),
        ("v4_zero_active_vector", _active_zero_copy),
        ("v4_zero_active_risk", _active_guarded),
        ("v4_dtype_active_risk", _dtype_active_guarded),
    ],
    "v4_no_transfer": [
        ("no_transfer_active_vector_risk", _active_guarded),
        ("no_transfer_active_partition_vector", _active_exact),
        ("no_transfer_active_partition_risk", _active_guarded),
        ("no_transfer_partition_vector_risk", _active_guarded),
        ("no_transfer_zero_active_vector", _active_zero_copy),
        ("no_transfer_dtype_active_risk", _dtype_active_guarded),
    ],
    "random_search": [
        ("random_zero_active", _active_zero_copy),
        ("random_dtype_risk", _dtype_sort_refined),
        ("random_sort_partition", _sort_projection),
        ("random_zero_dtype_risk", _dtype_active_guarded),
        ("random_zero_dtype_vector", _dtype_sort_refined),
        ("random_contiguous_dtype_active", _dtype_active_guarded),
    ],
    "template_synthesis": [
        ("template_active_set", _active_exact),
        ("template_vectorized_batch", _sort_projection),
        ("template_risk_stage", _active_guarded),
        ("template_sort_partition", _sort_projection),
        ("template_zero_copy", _sort_zero_copy),
        ("template_dtype", _dtype_sort_refined),
    ],
    "v3_compatible": [
        ("v3_vectorized_batch", _sort_projection),
        ("v3_zero_copy_representation", _sort_zero_copy),
        ("v3_dtype_specialization", _dtype_sort_refined),
        ("v3_contiguous_layout", _sort_contiguous),
    ],
}

ALL_CANDIDATES = [(arm, name, fn) for arm, items in CANDIDATES_BY_ARM.items() for name, fn in items]
assert len(ALL_CANDIDATES) == 28
