from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import linalg

Problem = dict[str, object]
Solution = dict[str, list[list[float]]]

CONDITION_THRESHOLD = 4.0 * float(np.finfo(np.float32).eps)


def _inputs(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["B"], dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("invalid Procrustes matrix dimensions")
    return a, b


def _exact_from_product(product: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(product, full_matrices=False)
    return u @ vh


def _float32_svd(product32: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return linalg.svd(
        product32,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )


def adaptive_cast_gesdd(problem: Problem) -> Solution:
    a, b = _inputs(problem)
    product64 = np.ascontiguousarray(b @ a.T)
    product32 = np.asarray(product64, dtype=np.float32, order="C")
    u, singular_values, vh = _float32_svd(product32)
    ratio = float(singular_values[-1] / singular_values[0])
    if ratio <= CONDITION_THRESHOLD:
        orthogonal = _exact_from_product(product64)
    else:
        orthogonal = np.asarray(u @ vh, dtype=np.float64)
    return {"solution": orthogonal.tolist()}


def adaptive_product_gesdd(problem: Problem) -> Solution:
    a, b = _inputs(problem)
    a32 = np.asarray(a, dtype=np.float32, order="C")
    b32 = np.asarray(b, dtype=np.float32, order="C")
    product32 = np.ascontiguousarray(b32 @ a32.T)
    u, singular_values, vh = _float32_svd(product32)
    ratio = float(singular_values[-1] / singular_values[0])
    if ratio <= CONDITION_THRESHOLD:
        product64 = np.ascontiguousarray(b @ a.T)
        orthogonal = _exact_from_product(product64)
    else:
        orthogonal = np.asarray(u @ vh, dtype=np.float64)
    return {"solution": orthogonal.tolist()}


def diagnostic_ratio(problem: Problem, product_mode: str) -> float:
    a, b = _inputs(problem)
    if product_mode == "cast":
        product32 = np.asarray(b @ a.T, dtype=np.float32, order="C")
    elif product_mode == "product":
        product32 = np.ascontiguousarray(
            np.asarray(b, dtype=np.float32, order="C")
            @ np.asarray(a, dtype=np.float32, order="C").T
        )
    else:
        raise ValueError(f"unknown product mode {product_mode}")
    singular_values = linalg.svdvals(product32, overwrite_a=True, check_finite=False)
    return float(singular_values[-1] / singular_values[0])


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "adaptive_cast_gesdd": adaptive_cast_gesdd,
    "adaptive_product_gesdd": adaptive_product_gesdd,
}
