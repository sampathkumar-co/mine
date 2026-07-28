from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import linalg

Problem = dict[str, object]
Solution = dict[str, list[list[float]]]


def _inputs(problem: Problem) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["B"], dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("invalid Procrustes matrix dimensions")
    return a, b


def float32_cast_gesdd(problem: Problem) -> Solution:
    a, b = _inputs(problem)
    product = np.asarray(b @ a.T, dtype=np.float32, order="C")
    u, _, vh = linalg.svd(
        product,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    return {"solution": np.asarray(u @ vh, dtype=np.float64).tolist()}


def float32_product_gesdd(problem: Problem) -> Solution:
    a, b = _inputs(problem)
    a32 = np.asarray(a, dtype=np.float32, order="C")
    b32 = np.asarray(b, dtype=np.float32, order="C")
    product = np.ascontiguousarray(b32 @ a32.T)
    u, _, vh = linalg.svd(
        product,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    return {"solution": np.asarray(u @ vh, dtype=np.float64).tolist()}


def gram_eigh_evd(problem: Problem) -> Solution:
    a, b = _inputs(problem)
    product = np.ascontiguousarray(b @ a.T)
    gram = np.ascontiguousarray(product.T @ product)
    eigenvalues, eigenvectors = linalg.eigh(
        gram,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    maximum = max(float(eigenvalues[-1]), 1.0)
    floor = np.finfo(np.float64).eps * maximum
    inverse_roots = 1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    scaled_vectors = eigenvectors * inverse_roots[None, :]
    orthogonal = (product @ scaled_vectors) @ eigenvectors.T
    return {"solution": orthogonal.tolist()}


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "float32_cast_gesdd": float32_cast_gesdd,
    "float32_product_gesdd": float32_product_gesdd,
    "gram_eigh_evd": gram_eigh_evd,
}
