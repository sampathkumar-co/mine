from __future__ import annotations

from collections.abc import Callable
import numpy as np

Problem = np.ndarray
Solution = list[complex]
Candidate = Callable[[Problem], Solution]


def _sorted_python(values) -> Solution:
    return sorted((complex(x) for x in values), key=lambda z: (-z.real, -z.imag))


def _sorted_numpy(values) -> Solution:
    arr = np.asarray(values)
    if arr.size == 0:
        return []
    order = np.lexsort((-arr.imag, -arr.real))
    return [complex(arr[i]) for i in order]


def _eig_reference(problem: Problem, numpy_sort: bool = False) -> Solution:
    vals = np.linalg.eig(np.asarray(problem, dtype=np.float64))[0]
    return _sorted_numpy(vals) if numpy_sort else _sorted_python(vals)


def _eigvals64(problem: Problem, numpy_sort: bool = False, contiguous: bool = False) -> Solution:
    a = np.asarray(problem, dtype=np.float64)
    if contiguous:
        a = np.ascontiguousarray(a)
    vals = np.linalg.eigvals(a)
    return _sorted_numpy(vals) if numpy_sort else _sorted_python(vals)


def _float32_is_low_risk(a64: np.ndarray, vals32: np.ndarray) -> bool:
    n = int(a64.shape[0])
    if n < 4:
        return False
    scale = max(float(np.linalg.norm(a64, ord=np.inf)), 1.0)
    vals = np.asarray(vals32, dtype=np.complex128)
    if vals.size != n or not np.all(np.isfinite(vals)):
        return False
    if float(np.min(np.abs(vals))) < 0.05 * scale:
        return False
    if n > 1:
        d = np.abs(vals[:, None] - vals[None, :])
        d += np.eye(n) * (scale * 10.0)
        if float(np.min(d)) < 1.0e-3 * scale:
            return False
    trace_expected = complex(np.trace(a64))
    trace_observed = complex(np.sum(vals))
    trace_err = abs(trace_observed - trace_expected) / max(abs(trace_expected), scale, 1.0e-12)
    second_expected = complex(np.sum(a64 * a64.T))
    second_observed = complex(np.sum(vals * vals))
    second_err = abs(second_observed - second_expected) / max(abs(second_expected), scale * scale, 1.0e-12)
    return trace_err <= 1.0e-7 and second_err <= 1.0e-7


def _guarded_mixed(problem: Problem, numpy_sort: bool = True, contiguous: bool = False) -> Solution:
    a64 = np.asarray(problem, dtype=np.float64)
    if contiguous:
        a64 = np.ascontiguousarray(a64)
    vals32 = np.linalg.eigvals(a64.astype(np.float32, copy=False)).astype(np.complex128)
    if _float32_is_low_risk(a64, vals32):
        return _sorted_numpy(vals32) if numpy_sort else _sorted_python(vals32)
    return _eigvals64(a64, numpy_sort=numpy_sort)


def _symmetric_if_exact(problem: Problem, numpy_sort: bool = True) -> Solution:
    a = np.asarray(problem, dtype=np.float64)
    if np.array_equal(a, a.T):
        vals = np.linalg.eigvalsh(a)
        return _sorted_numpy(vals) if numpy_sort else _sorted_python(vals)
    return _eigvals64(a, numpy_sort=numpy_sort)


def _early_small_exact(problem: Problem, numpy_sort: bool = True) -> Solution:
    a = np.asarray(problem, dtype=np.float64)
    n = int(a.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [complex(a[0, 0])]
    if n == 2:
        tr = a[0, 0] + a[1, 1]
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        disc = complex(tr * tr - 4.0 * det) ** 0.5
        vals = np.array([(tr + disc) / 2.0, (tr - disc) / 2.0], dtype=np.complex128)
        return _sorted_numpy(vals) if numpy_sort else _sorted_python(vals)
    return _eigvals64(a, numpy_sort=numpy_sort)


CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {
    "v4_full": [
        ("v4_mixed_sort_risk", lambda p: _guarded_mixed(p, True)),
        ("v4_mixed_closed_risk", lambda p: _guarded_mixed(p, False)),
        ("v4_mixed_vector_risk", lambda p: _guarded_mixed(p, True, True)),
        ("v4_mixed_risk_early", lambda p: _guarded_mixed(p, True)),
        ("v4_mixed_symmetric_risk", lambda p: _symmetric_if_exact(p, True)),
        ("v4_zero_mixed_risk", lambda p: _guarded_mixed(np.asarray(p), True)),
    ],
    "v4_no_transfer": [
        ("no_transfer_mixed_sort_risk", lambda p: _guarded_mixed(p, True)),
        ("no_transfer_mixed_closed_risk", lambda p: _guarded_mixed(p, False)),
        ("no_transfer_mixed_vector_risk", lambda p: _guarded_mixed(p, True, True)),
        ("no_transfer_mixed_risk_early", lambda p: _guarded_mixed(p, True)),
        ("no_transfer_mixed_symmetric_risk", lambda p: _symmetric_if_exact(p, True)),
        ("no_transfer_mixed_sort_vector", lambda p: _guarded_mixed(p, True)),
    ],
    "random_search": [
        ("random_mixed_risk_early", lambda p: _guarded_mixed(p, True)),
        ("random_contiguous_mixed_early", lambda p: _guarded_mixed(p, False, True)),
        ("random_contiguous_symmetric_early", lambda p: _symmetric_if_exact(np.ascontiguousarray(p), True)),
        ("random_zero_symmetric_sort", lambda p: _symmetric_if_exact(np.asarray(p), False)),
        ("random_zero_contiguous_closed", lambda p: _eigvals64(p, True, True)),
        ("random_zero_dtype_risk", lambda p: _eigvals64(np.asarray(p, dtype=np.float64), True)),
    ],
    "template_synthesis": [
        ("template_mixed_precision", lambda p: _guarded_mixed(p, True)),
        ("template_risk_stage", lambda p: _eig_reference(p, False)),
        ("template_sort_partition", lambda p: _eig_reference(p, True)),
        ("template_closed_form", lambda p: _eigvals64(p, False)),
        ("template_vectorized_batch", lambda p: _eigvals64(p, True)),
        ("template_early_certificate", lambda p: _early_small_exact(p, True)),
    ],
    "v3_compatible": [
        ("v3_vectorized_batch", lambda p: _eig_reference(p, True)),
        ("v3_zero_copy_representation", lambda p: _eig_reference(np.asarray(p), False)),
        ("v3_dtype_specialization", lambda p: _eig_reference(np.asarray(p, dtype=np.float64), False)),
        ("v3_contiguous_layout", lambda p: _eig_reference(np.ascontiguousarray(p, dtype=np.float64), False)),
    ],
}

ALL_CANDIDATES = [(arm, name, fn) for arm, items in CANDIDATES_BY_ARM.items() for name, fn in items]
assert len(ALL_CANDIDATES) == 28
