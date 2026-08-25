from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.fft
import scipy.fftpack


@dataclass(frozen=True)
class Candidate:
    name: str
    arm: str
    implementation_class: str
    operators: tuple[str, ...]
    transfer_ids: tuple[str, ...]
    learned_template: str | None
    baseline_id: str | None
    solve: Callable[[np.ndarray], np.ndarray]


def _array(problem) -> np.ndarray:
    a = np.asarray(problem)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or min(a.shape) < 2:
        raise ValueError('DCT-I problem must be a nontrivial square matrix')
    if not np.all(np.isfinite(a)):
        raise ValueError('DCT-I problem contains nonfinite values')
    return a


def reference_exact(problem) -> np.ndarray:
    a = np.asarray(_array(problem), dtype=np.float64)
    return scipy.fftpack.dctn(a, type=1)


def pocketfft_exact(problem) -> np.ndarray:
    a = np.asarray(_array(problem), dtype=np.float64)
    return scipy.fft.dctn(a, type=1)


def pocketfft_contiguous_exact(problem) -> np.ndarray:
    a = np.ascontiguousarray(_array(problem), dtype=np.float64)
    return scipy.fft.dctn(a, type=1)


def pocketfft_float32(problem) -> np.ndarray:
    a = np.ascontiguousarray(_array(problem), dtype=np.float32)
    return scipy.fft.dctn(a, type=1)


def even_extension_rfftn_exact(problem) -> np.ndarray:
    a = np.asarray(_array(problem), dtype=np.float64)
    # A type-I DCT is the real FFT of an even extension.  Apply the extension
    # independently to both axes, then crop the real spectrum back to the
    # original square shape.  No fitted constants or task-instance tuning.
    ext0 = np.concatenate((a, a[-2:0:-1, :]), axis=0)
    ext = np.concatenate((ext0, ext0[:, -2:0:-1]), axis=1)
    spectrum = scipy.fft.rfftn(ext, axes=(0, 1))
    return np.asarray(spectrum.real[: a.shape[0], : a.shape[1]], dtype=np.float64)


def explicit_dct1_certificate(problem) -> np.ndarray:
    a = np.asarray(_array(problem), dtype=np.float64)
    n0, n1 = a.shape

    def matrix(n: int) -> np.ndarray:
        k = np.arange(n, dtype=np.float64)[:, None]
        j = np.arange(n, dtype=np.float64)[None, :]
        c = 2.0 * np.cos(np.pi * k * j / float(n - 1))
        c[:, 0] = 1.0
        c[:, -1] = (-1.0) ** np.arange(n, dtype=np.float64)
        return c

    c0 = matrix(n0)
    c1 = matrix(n1)
    return c0 @ a @ c1.T


def relative_l2(observed, expected) -> float:
    obs = np.asarray(observed, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    if obs.shape != exp.shape or not np.all(np.isfinite(obs)):
        return float('inf')
    return float(np.linalg.norm(obs - exp) / (np.linalg.norm(exp) + 1e-12))


def independent_semantic_certificate(problem, solution) -> bool:
    try:
        expected = explicit_dct1_certificate(problem)
        return relative_l2(solution, expected) <= 1e-6
    except Exception:
        return False


def official_verifier_accepts(problem, solution) -> bool:
    try:
        return relative_l2(solution, reference_exact(problem)) <= 1e-6
    except Exception:
        return False


def _map_engine_candidate(arm: str, proposal: dict) -> Candidate:
    ops = tuple(str(x) for x in proposal['operators'])
    tids = tuple(str(x) for x in proposal['transfer_ids'])
    s = set(ops)

    # BFR/CAC are deliberately preserved as source-fingerprint stress cases:
    # the task source contains lexical boolean/certificate signals, but no lawful
    # graph frontier or active-constraint reduction exists in DCT-I semantics.
    if tids in (("TM-BFR-01",), ("TM-CAC-01",)):
        impl, fn = 'source_equivalent_fallback', reference_exact
    elif tids == ("TM-RRR-01",):
        impl, fn = 'even_extension_rfftn_exact', even_extension_rfftn_exact
    elif tids == ("TM-PBEB-01",):
        impl, fn = 'pocketfft_float32_tolerance_equivalent', pocketfft_float32
    elif 'dtype_specialization' in s:
        impl, fn = 'pocketfft_float32_tolerance_equivalent', pocketfft_float32
    elif 'native_backend_substitution' in s:
        impl, fn = 'pocketfft_float64', pocketfft_exact
    elif 'reduced_representation' in s:
        impl, fn = 'even_extension_rfftn_exact', even_extension_rfftn_exact
    elif 'contiguous_layout' in s and 'vectorized_batch_kernel' in s:
        impl, fn = 'pocketfft_contiguous_float64', pocketfft_contiguous_exact
    elif s.intersection({'vectorized_batch_kernel', 'zero_copy_representation'}):
        impl, fn = 'pocketfft_float64', pocketfft_exact
    else:
        impl, fn = 'scipy_fftpack_reference', reference_exact

    public_arm = {
        'v5_full': 'v6_full',
        'v5_no_transfer': 'v6_no_transfer',
        'v4_compatible': 'v5_compatible',
    }.get(arm, arm)
    return Candidate(
        f"{public_arm}_r{proposal['rank']}_{proposal['proposal_id']}",
        public_arm,
        impl,
        ops,
        tids,
        proposal.get('learned_template'),
        None,
        fn,
    )


def build_candidates(task_source_text: str) -> dict[str, list[Candidate]]:
    from engine import generate_proposals

    generated = generate_proposals(task_source_text)
    arms = {k: [] for k in (
        'v6_full', 'v6_no_transfer', 'random_search',
        'static_template', 'v5_compatible', 'strong_baseline',
    )}
    for engine_arm, proposals in generated['arms'].items():
        for proposal in proposals:
            candidate = _map_engine_candidate(engine_arm, proposal)
            arms[candidate.arm].append(candidate)

    arms['strong_baseline'].append(Candidate(
        'strong_baseline_sb_native_numeric_01_pocketfft_float32',
        'strong_baseline',
        'pocketfft_float32_tolerance_equivalent',
        ('direct_general_purpose_numeric_backend', 'dtype_specialization'),
        (),
        None,
        'SB-NATIVE-NUMERIC-01',
        pocketfft_float32,
    ))
    return arms
