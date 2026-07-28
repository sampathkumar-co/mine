from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import linalg

Problem = dict[str, object]
Solution = dict[str, list[list[float]]]


def _matrix(problem: Problem) -> np.ndarray:
    a = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["B"], dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("invalid Procrustes matrix dimensions")
    return np.ascontiguousarray(b @ a.T)


def scipy_gesdd(problem: Problem) -> Solution:
    product = _matrix(problem)
    u, _, vh = linalg.svd(
        product,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    return {"solution": (u @ vh).tolist()}


def scipy_gesvd(problem: Problem) -> Solution:
    product = _matrix(problem)
    u, _, vh = linalg.svd(
        product,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesvd",
    )
    return {"solution": (u @ vh).tolist()}


def scipy_polar(problem: Problem) -> Solution:
    product = _matrix(problem)
    unitary, _ = linalg.polar(product, side="right")
    return {"solution": unitary.tolist()}


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "scipy_gesdd": scipy_gesdd,
    "scipy_gesvd": scipy_gesvd,
    "scipy_polar": scipy_polar,
}
