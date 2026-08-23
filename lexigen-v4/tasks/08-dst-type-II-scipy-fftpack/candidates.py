from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.fft
import scipy.fftpack

Problem = np.ndarray
Solution = np.ndarray
Candidate = Callable[[Problem], Solution]


def _array(problem: Problem, *, dtype=None, contiguous: bool = False) -> np.ndarray:
    value = np.asarray(problem, dtype=dtype)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.size == 0:
        raise ValueError("DST input must be a nonempty square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError("DST input contains non-finite values")
    if contiguous:
        value = np.ascontiguousarray(value)
    return value


def _fftpack(problem: Problem) -> Solution:
    return np.asarray(scipy.fftpack.dstn(_array(problem), type=2))


def _fftpack_contiguous(problem: Problem) -> Solution:
    return np.asarray(scipy.fftpack.dstn(_array(problem, contiguous=True), type=2))


def _fftpack_float32(problem: Problem) -> Solution:
    return np.asarray(scipy.fftpack.dstn(_array(problem, dtype=np.float32), type=2))


def _fftpack_guarded(problem: Problem) -> Solution:
    value = _fftpack(problem)
    if np.all(np.isfinite(value)):
        return value
    return np.asarray(scipy.fft.dstn(_array(problem), type=2))


def _fftpack_float32_guarded(problem: Problem) -> Solution:
    value = _fftpack_float32(problem)
    if np.all(np.isfinite(value)):
        return value
    return _fftpack(problem)


def _modern(problem: Problem) -> Solution:
    return np.asarray(scipy.fft.dstn(_array(problem), type=2))


def _modern_contiguous(problem: Problem) -> Solution:
    return np.asarray(scipy.fft.dstn(_array(problem, contiguous=True), type=2))


def _modern_float32(problem: Problem) -> Solution:
    return np.asarray(scipy.fft.dstn(_array(problem, dtype=np.float32), type=2))


def _modern_guarded(problem: Problem) -> Solution:
    value = _modern(problem)
    if np.all(np.isfinite(value)):
        return value
    return _fftpack(problem)


def _modern_contiguous_guarded(problem: Problem) -> Solution:
    value = _modern_contiguous(problem)
    if np.all(np.isfinite(value)):
        return value
    return _fftpack(problem)


def _modern_float32_guarded(problem: Problem) -> Solution:
    value = _modern_float32(problem)
    if np.all(np.isfinite(value)):
        return value
    return _modern(problem)


CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {
    "v4_full": [
        ("v4_zero_vector_risk", _modern_guarded),
        ("v4_dtype_vector_risk", _modern_float32_guarded),
        ("v4_contiguous_vector_risk", _modern_contiguous_guarded),
        ("v4_zero_dtype_vector", _modern_float32),
        ("v4_zero_dtype_risk", _fftpack_float32_guarded),
        ("v4_zero_contiguous_vector", _modern_contiguous),
    ],
    "v4_no_transfer": [
        ("no_transfer_zero_vector_risk", _modern_guarded),
        ("no_transfer_dtype_vector_risk", _modern_float32_guarded),
        ("no_transfer_contiguous_vector_risk", _modern_contiguous_guarded),
        ("no_transfer_zero_dtype_vector", _modern_float32),
        ("no_transfer_zero_dtype_risk", _fftpack_float32_guarded),
        ("no_transfer_contiguous_dtype_vector", _modern_float32),
    ],
    "random_search": [
        ("random_zero_dtype_risk", _fftpack_float32_guarded),
        ("random_dtype", _fftpack_float32),
        ("random_dtype_vector_risk", _modern_float32_guarded),
        ("random_zero_dtype_vector", _modern_float32),
        ("random_risk", _fftpack_guarded),
        ("random_zero_risk", _fftpack_guarded),
    ],
    "template_synthesis": [
        ("template_vectorized_batch", _modern),
        ("template_risk_stage", _fftpack_guarded),
        ("template_zero_copy", _fftpack),
        ("template_dtype", _fftpack_float32),
        ("template_contiguous", _fftpack_contiguous),
    ],
    "v3_compatible": [
        ("v3_vectorized_batch", _modern),
        ("v3_zero_copy_representation", _fftpack),
        ("v3_dtype_specialization", _fftpack_float32),
        ("v3_contiguous_layout", _fftpack_contiguous),
    ],
}

ALL_CANDIDATES = [(arm, name, fn) for arm, items in CANDIDATES_BY_ARM.items() for name, fn in items]
assert len(ALL_CANDIDATES) == 27
