from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import linalg

Problem = dict[str, object]
Solution = dict[str, list[list[float]]]


def _product(problem: Problem) -> np.ndarray:
    a = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["B"], dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("invalid Procrustes matrix dimensions")
    return np.ascontiguousarray(b @ a.T)


def _refined(problem: Problem, rank: int) -> Solution:
    product = _product(problem)
    product32 = np.asarray(product, dtype=np.float32, order="C")
    u32, _, vh32 = linalg.svd(
        product32,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    u = np.asarray(u32, dtype=np.float64)
    vh = np.asarray(vh32, dtype=np.float64)
    orthogonal = u @ vh

    active_rank = min(rank, product.shape[0])
    weak_u = u[:, -active_rank:]
    weak_v = vh[-active_rank:, :].T
    compressed = weak_u.T @ (product @ weak_v)
    correction_u, _, correction_vh = np.linalg.svd(compressed, full_matrices=False)
    rotation = correction_u @ correction_vh
    orthogonal += (weak_u @ (rotation - np.eye(active_rank))) @ weak_v.T
    return {"solution": orthogonal.tolist()}


def subspace_refine4(problem: Problem) -> Solution:
    return _refined(problem, 4)


def subspace_refine8(problem: Problem) -> Solution:
    return _refined(problem, 8)


def subspace_refine16(problem: Problem) -> Solution:
    return _refined(problem, 16)


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "subspace_refine4": subspace_refine4,
    "subspace_refine8": subspace_refine8,
    "subspace_refine16": subspace_refine16,
}
