from __future__ import annotations

from typing import Callable

import cvxpy as cp
import numpy as np

Problem = dict[str, object]
Solution = dict[str, object]
Candidate = Callable[[Problem], Solution]


def _arrays(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(problem["tensor"], dtype=np.float64)
    mask = np.asarray(problem["mask"], dtype=bool)
    if observed.ndim != 3 or mask.shape != observed.shape:
        raise ValueError("tensor/mask must be matching 3D arrays")
    if not np.all(np.isfinite(observed)):
        raise ValueError("tensor contains non-finite entries")
    return observed, mask


def _solution(value: np.ndarray | None) -> Solution:
    if value is None:
        return {"completed_tensor": []}
    return {"completed_tensor": np.asarray(value, dtype=np.float64).tolist()}


def _mode2(value: np.ndarray) -> np.ndarray:
    return np.transpose(value, (1, 0, 2)).reshape(value.shape[1], value.shape[0] * value.shape[2])


def _mode3(value: np.ndarray) -> np.ndarray:
    return np.transpose(value, (2, 0, 1)).reshape(value.shape[2], value.shape[0] * value.shape[1])


def _svd_warm_start(unfolding: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = np.where(mask, unfolding, 0.0)
    rank = max(1, min(3, min(filled.shape)))
    for _ in range(2):
        try:
            u, s, vh = np.linalg.svd(filled, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        low_rank = (u[:, :rank] * s[:rank]) @ vh[:rank]
        filled = np.where(mask, unfolding, low_rank)
    return filled


def _solve_single_mode(
    problem: Problem,
    *,
    indexed_constraints: bool,
    warm_start: bool,
    fast_paths: bool,
    contiguous: bool = False,
) -> Solution:
    observed, mask = _arrays(problem)
    if fast_paths and mask.all():
        return _solution(observed.copy())
    if not mask.any():
        return _solution(np.zeros_like(observed))

    dim1, dim2, dim3 = observed.shape
    unfolding = observed.reshape(dim1, dim2 * dim3)
    mask1 = mask.reshape(dim1, dim2 * dim3)
    if contiguous:
        unfolding = np.ascontiguousarray(unfolding)
        mask1 = np.ascontiguousarray(mask1)

    x = cp.Variable(unfolding.shape)
    if indexed_constraints:
        observed_idx = np.flatnonzero(mask1.ravel(order="C"))
        x_vec = cp.vec(x, order="C")
        rhs = unfolding.ravel(order="C")[observed_idx]
        constraints = [x_vec[observed_idx] == rhs]
    else:
        constraints = [cp.multiply(x, mask1) == cp.multiply(unfolding, mask1)]

    objective = cp.Minimize(cp.norm(x, "nuc"))
    problem_cp = cp.Problem(objective, constraints)
    if warm_start:
        x.value = _svd_warm_start(unfolding, mask1)
    try:
        problem_cp.solve(warm_start=warm_start)
    except cp.SolverError:
        return {"completed_tensor": []}
    if problem_cp.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
        return {"completed_tensor": []}
    return _solution(np.asarray(x.value, dtype=np.float64).reshape(observed.shape))


def _solve_reference_shape(problem: Problem, *, contiguous: bool = False, fast_paths: bool = False) -> Solution:
    observed, mask = _arrays(problem)
    if fast_paths and mask.all():
        return _solution(observed.copy())
    if not mask.any():
        return _solution(np.zeros_like(observed))

    dim1, dim2, dim3 = observed.shape
    unfolding1 = observed.reshape(dim1, dim2 * dim3)
    mask1 = mask.reshape(dim1, dim2 * dim3)
    unfolding2 = _mode2(observed)
    mask2 = _mode2(mask)
    unfolding3 = _mode3(observed)
    mask3 = _mode3(mask)
    if contiguous:
        unfolding1 = np.ascontiguousarray(unfolding1)
        mask1 = np.ascontiguousarray(mask1)
        unfolding2 = np.ascontiguousarray(unfolding2)
        mask2 = np.ascontiguousarray(mask2)
        unfolding3 = np.ascontiguousarray(unfolding3)
        mask3 = np.ascontiguousarray(mask3)

    x1 = cp.Variable(unfolding1.shape)
    x2 = cp.Variable(unfolding2.shape)
    x3 = cp.Variable(unfolding3.shape)
    objective = cp.Minimize(cp.norm(x1, "nuc") + cp.norm(x2, "nuc") + cp.norm(x3, "nuc"))
    constraints = [
        cp.multiply(x1, mask1) == cp.multiply(unfolding1, mask1),
        cp.multiply(x2, mask2) == cp.multiply(unfolding2, mask2),
        cp.multiply(x3, mask3) == cp.multiply(unfolding3, mask3),
    ]
    problem_cp = cp.Problem(objective, constraints)
    try:
        problem_cp.solve()
    except cp.SolverError:
        return {"completed_tensor": []}
    if problem_cp.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x1.value is None:
        return {"completed_tensor": []}
    return _solution(np.asarray(x1.value, dtype=np.float64).reshape(observed.shape))


def _single_active_warm(problem: Problem) -> Solution:
    return _solve_single_mode(problem, indexed_constraints=True, warm_start=True, fast_paths=True)


def _single_active(problem: Problem) -> Solution:
    return _solve_single_mode(problem, indexed_constraints=True, warm_start=False, fast_paths=True)


def _single_multiply_warm(problem: Problem) -> Solution:
    return _solve_single_mode(problem, indexed_constraints=False, warm_start=True, fast_paths=True)


def _single_multiply(problem: Problem) -> Solution:
    return _solve_single_mode(problem, indexed_constraints=False, warm_start=False, fast_paths=True)


def _single_contiguous(problem: Problem) -> Solution:
    return _solve_single_mode(problem, indexed_constraints=True, warm_start=False, fast_paths=True, contiguous=True)


def _full_vectorized(problem: Problem) -> Solution:
    return _solve_reference_shape(problem, contiguous=False, fast_paths=False)


def _full_contiguous(problem: Problem) -> Solution:
    return _solve_reference_shape(problem, contiguous=True, fast_paths=False)


def _full_fast_paths(problem: Problem) -> Solution:
    return _solve_reference_shape(problem, contiguous=False, fast_paths=True)


CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {
    "v4_full": [
        ("v4_structure_refine_active", _single_active_warm),
        ("v4_structure_refine_bitmask", _single_contiguous),
        ("v4_structure_active_bitmask", _single_active),
        ("v4_structure_refine_risk", _single_multiply_warm),
        ("v4_structure_active_risk", _single_active),
        ("v4_structure_refine_closed", _single_multiply),
    ],
    "v4_no_transfer": [
        ("no_transfer_structure_active_risk", _single_active),
        ("no_transfer_structure_refine_active", _single_active_warm),
        ("no_transfer_structure_active_bitmask", _single_contiguous),
        ("no_transfer_structure_refine_risk", _single_multiply_warm),
        ("no_transfer_structure_refine_closed", _single_multiply),
        ("no_transfer_structure_bitmask", _single_contiguous),
    ],
    "random_search": [
        ("random_dtype_bit_vector", _full_contiguous),
        ("random_structure_early", _single_multiply),
        ("random_zero_contiguous_active", _single_contiguous),
        ("random_dtype_closed_early", _single_multiply),
        ("random_zero_closed", _single_multiply),
        ("random_dtype_structure_early", _single_active),
    ],
    "template_synthesis": [
        ("template_structure_initialization", _single_multiply_warm),
        ("template_active_set", _single_active),
        ("template_bit_parallel", _full_contiguous),
        ("template_risk_stage", _full_fast_paths),
        ("template_closed_form", _single_multiply),
        ("template_vectorized_batch", _full_vectorized),
    ],
    "v3_compatible": [
        ("v3_vectorized_batch", _full_vectorized),
        ("v3_zero_copy_representation", _full_vectorized),
        ("v3_dtype_specialization", _full_contiguous),
        ("v3_contiguous_layout", _full_contiguous),
    ],
}

ALL_CANDIDATES = [(arm, name, fn) for arm, items in CANDIDATES_BY_ARM.items() for name, fn in items]
assert len(ALL_CANDIDATES) == 28
