from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import linalg as sla
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering

Problem = dict[str, object]
Solution = dict[str, object]
Candidate = Callable[[Problem], Solution]


def _inputs(problem: Problem) -> tuple[np.ndarray, int]:
    S = problem.get("similarity_matrix")
    k = problem.get("n_clusters")
    if not isinstance(S, np.ndarray) or S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("similarity_matrix must be square numpy array")
    if not isinstance(k, int) or k < 1:
        raise ValueError("n_clusters must be positive int")
    if not np.all(np.isfinite(S)):
        raise ValueError("similarity_matrix contains non-finite values")
    return S, k


def _trivial(n: int, k: int) -> np.ndarray | None:
    if n == 0:
        return np.array([], dtype=int)
    if k == 1:
        return np.zeros(n, dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)
    return None


def _reference_exact(problem: Problem) -> Solution:
    S, k = _inputs(problem)
    n = S.shape[0]
    trivial = _trivial(n, k)
    if trivial is not None:
        return {"labels": trivial}
    labels = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    ).fit_predict(S)
    return {"labels": np.asarray(labels, dtype=int)}


def _farthest_first_lloyd(X: np.ndarray, k: int, iters: int = 15) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    x2 = np.einsum("ij,ij->i", X, X)
    seeds = [int(np.argmax(x2))]
    for _ in range(1, k):
        C = X[seeds]
        c2 = np.einsum("ij,ij->i", C, C)
        d2 = x2[:, None] + c2[None, :] - 2.0 * (X @ C.T)
        mind2 = d2.min(axis=1)
        mind2[seeds] = -np.inf
        seeds.append(int(np.argmax(mind2)))
    C = X[seeds].copy()
    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(iters):
        c2 = np.einsum("ij,ij->i", C, C)
        d2 = x2[:, None] + c2[None, :] - 2.0 * (X @ C.T)
        labels = np.argmin(d2, axis=1).astype(int)
        newC = C.copy()
        for j in range(k):
            idx = np.flatnonzero(labels == j)
            if idx.size:
                newC[j] = X[idx].mean(axis=0)
        if np.allclose(newC, C, rtol=1e-7, atol=1e-9):
            break
        C = newC
    return labels


def _normalized_similarity(S: np.ndarray, dtype: np.dtype = np.float64) -> np.ndarray:
    A = np.asarray(S, dtype=dtype)
    A = np.clip(A, 0.0, 1.0).copy()
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    inv = 1.0 / np.sqrt(np.maximum(deg, np.asarray(1e-12, dtype=A.dtype)))
    return (inv[:, None] * A) * inv[None, :]


def _dense_reduced(problem: Problem, *, dtype: np.dtype = np.float64) -> Solution:
    S, k = _inputs(problem)
    n = S.shape[0]
    trivial = _trivial(n, k)
    if trivial is not None:
        return {"labels": trivial}
    B = _normalized_similarity(S, dtype=dtype)
    L = np.eye(n, dtype=B.dtype) - B
    _, U = sla.eigh(
        (L + L.T) * 0.5,
        subset_by_index=[0, k - 1],
        driver="evr",
        check_finite=False,
    )
    U = np.asarray(U, dtype=np.float64)
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    labels = _farthest_first_lloyd(U / norms, k, iters=15)
    return {"labels": labels}


def _sparse_reduced(problem: Problem) -> Solution:
    S, k = _inputs(problem)
    n = S.shape[0]
    trivial = _trivial(n, k)
    if trivial is not None:
        return {"labels": trivial}
    B = _normalized_similarity(S, dtype=np.float64)
    v0 = np.linspace(1.0, 2.0, n, dtype=np.float64)
    vals, U = eigsh(
        csr_matrix(B),
        k=k,
        which="LA",
        tol=1e-5,
        maxiter=max(300, 8 * n),
        v0=v0,
    )
    U = U[:, np.argsort(vals)[::-1]]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    labels = _farthest_first_lloyd(U / norms, k, iters=15)
    return {"labels": labels}


def _certificate(problem: Problem, labels: np.ndarray) -> bool:
    S, k = _inputs(problem)
    n = S.shape[0]
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (n,):
        return False
    if n == 0:
        return True
    if k == 1:
        return np.unique(labels).size == 1
    if k >= n:
        return np.unique(labels).size == n
    if np.unique(labels).size != k or np.any(labels < 0):
        return False
    A = np.clip(np.asarray(S, dtype=np.float64), 0.0, 1.0).copy()
    np.fill_diagonal(A, 0.0)
    iu = np.triu_indices(n, 1)
    same = (labels[:, None] == labels[None, :])[iu]
    sims = A[iu]
    if not np.any(same) or not np.any(~same):
        return False
    return float(sims[same].mean()) >= float(sims[~same].mean())


def _frontier_components(problem: Problem) -> Solution | None:
    S, k = _inputs(problem)
    n = S.shape[0]
    trivial = _trivial(n, k)
    if trivial is not None:
        return {"labels": trivial}
    A = np.clip(np.asarray(S, dtype=np.float64), 0.0, 1.0).copy()
    np.fill_diagonal(A, 0.0)
    q = min(n - 1, max(2, int(np.ceil(np.log2(max(n, 2))))))
    idx = np.argpartition(A, -q, axis=1)[:, -q:]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, idx] = True
    graph = mask & mask.T
    count, labels = connected_components(csr_matrix(graph), directed=False, return_labels=True)
    labels = labels.astype(int)
    if count == k and _certificate(problem, labels):
        return {"labels": labels}
    return None


def _active_core(problem: Problem) -> Solution | None:
    S, k = _inputs(problem)
    n = S.shape[0]
    trivial = _trivial(n, k)
    if trivial is not None:
        return {"labels": trivial}
    A = np.clip(np.asarray(S, dtype=np.float64), 0.0, 1.0).copy()
    np.fill_diagonal(A, 0.0)
    off = A[np.triu_indices(n, 1)]
    if off.size == 0:
        return None
    threshold = float(np.quantile(off, 0.85))
    graph = A >= threshold
    np.fill_diagonal(graph, False)
    count, labels = connected_components(csr_matrix(graph), directed=False, return_labels=True)
    labels = labels.astype(int)
    if count == k and _certificate(problem, labels):
        return {"labels": labels}
    return None


def _safe(call: Callable[[Problem], Solution], problem: Problem, *, fallback: Callable[[Problem], Solution]) -> Solution:
    try:
        result = call(problem)
        labels = np.asarray(result["labels"], dtype=int)
        if _certificate(problem, labels):
            return {"labels": labels}
    except Exception:
        pass
    return fallback(problem)


def _learned_pipeline(problem: Problem, learned_template: str) -> Solution:
    if learned_template == "bit_frontier_restriction":
        early = _frontier_components(problem)
        return early if early is not None else _reference_exact(problem)
    if learned_template == "certified_active_core":
        early = _active_core(problem)
        return early if early is not None else _reference_exact(problem)
    if learned_template == "reduced_representation_refinement":
        return _safe(_sparse_reduced, problem, fallback=_reference_exact)
    raise ValueError(f"unknown learned template: {learned_template}")


def _generic_pipeline(problem: Problem, operators: tuple[str, ...]) -> Solution:
    ops = set(operators)
    if "sparse_frontier_search" in ops or "bit_parallel_representation" in ops:
        early = _frontier_components(problem)
        if early is not None:
            return early

    if "dtype_specialization" in ops:
        if "risk_aware_staging" in ops:
            return _safe(lambda p: _dense_reduced(p, dtype=np.float32), problem, fallback=_reference_exact)
        try:
            return _dense_reduced(problem, dtype=np.float32)
        except Exception:
            return _reference_exact(problem)

    if (
        "reduced_representation" in ops
        or "bounded_exact_refinement" in ops
        or "native_backend_substitution" in ops
        or "vectorized_batch_kernel" in ops
        or "sort_partition_reduction" in ops
    ):
        if "risk_aware_staging" in ops:
            return _safe(_dense_reduced, problem, fallback=_reference_exact)
        try:
            return _dense_reduced(problem)
        except Exception:
            return _reference_exact(problem)

    return _reference_exact(problem)


@lru_cache(maxsize=None)
def _implementation(operators: tuple[str, ...], learned_template: str | None) -> Candidate:
    if learned_template is None:
        def candidate(problem: Problem) -> Solution:
            return _generic_pipeline(problem, operators)
    else:
        def candidate(problem: Problem) -> Solution:
            return _learned_pipeline(problem, learned_template)
    return candidate


PROPOSALS: dict[str, list[tuple[str, tuple[str, ...], tuple[str, ...], str | None]]] = {
    "v5_full": [
        ("3304c859d463a501bd86", ("bit_parallel_representation", "sparse_frontier_search", "early_certificate_exit"), ("TM-BFR-01",), "bit_frontier_restriction"),
        ("41510e43e8fafb598496", ("reduced_representation", "bounded_exact_refinement", "risk_aware_staging"), ("TM-RRR-01",), "reduced_representation_refinement"),
        ("a6102573c9f355414229", ("active_set_decomposition", "early_certificate_exit", "risk_aware_staging"), ("TM-CAC-01",), "certified_active_core"),
        ("e909b567bac8aa01b86e", ("bit_parallel_representation", "reduced_representation", "bounded_exact_refinement"), (), None),
        ("2c6961f9ca6711ce3a3f", ("bit_parallel_representation", "sparse_frontier_search", "reduced_representation"), (), None),
        ("deab8c4f8228e58310f0", ("bit_parallel_representation", "bounded_exact_refinement", "sort_partition_reduction"), (), None),
    ],
    "v5_no_transfer": [
        ("66c5848a3c8a4f51b562", ("bit_parallel_representation", "reduced_representation", "bounded_exact_refinement"), (), None),
        ("b93eda021fe3bc5d89cb", ("bit_parallel_representation", "sparse_frontier_search", "reduced_representation"), (), None),
        ("acd3613bac98cfcfce94", ("bit_parallel_representation", "bounded_exact_refinement", "sort_partition_reduction"), (), None),
        ("d14c06bd6ae45a8dd009", ("bit_parallel_representation", "sparse_frontier_search", "bounded_exact_refinement"), (), None),
        ("0d67be77188699dfb9ee", ("bit_parallel_representation", "sparse_frontier_search", "sort_partition_reduction"), (), None),
        ("5fb64f97237473444d3c", ("bit_parallel_representation", "reduced_representation", "sort_partition_reduction"), (), None),
    ],
    "random_search": [
        ("53ff79efe576ec7bfcac", ("early_certificate_exit", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("614f3090e8375b79582f", ("vectorized_batch_kernel", "early_certificate_exit", "sparse_frontier_search"), (), None),
        ("53f2d6b2e3dc583d92fe", ("native_backend_substitution", "early_certificate_exit", "bounded_exact_refinement"), (), None),
        ("f38f42c50dd9ebd33968", ("zero_copy_representation", "early_certificate_exit", "reduced_representation"), (), None),
        ("5793361bb742983fa1ea", ("sparse_frontier_search",), (), None),
        ("0b295b955536d6d4b680", ("zero_copy_representation",), (), None),
    ],
    "static_template": [
        ("dbfcd2af539b0b2636e7", ("bit_parallel_representation", "sparse_frontier_search"), (), None),
        ("8fd871e046faa7e4d37c", ("reduced_representation", "bounded_exact_refinement"), (), None),
        ("820b1c309b6117eb268d", ("active_set_decomposition", "early_certificate_exit"), (), None),
        ("8f1dafda0d3fbc099aa9", ("zero_copy_representation", "vectorized_batch_kernel"), (), None),
        ("357e80313b8b9dc3cf36", ("contiguous_layout", "vectorized_batch_kernel"), (), None),
        ("d044a19fd4551034dc11", ("dtype_specialization", "risk_aware_staging"), (), None),
    ],
    "v4_compatible": [
        ("0dde88a4a159a3ad0e40", ("bit_parallel_representation", "bounded_exact_refinement", "sort_partition_reduction"), (), None),
        ("bd9a928b0a959b433de2", ("bit_parallel_representation", "sparse_frontier_search", "bounded_exact_refinement"), (), None),
        ("a55401472772ccc050ac", ("bit_parallel_representation", "sparse_frontier_search", "sort_partition_reduction"), (), None),
        ("885bf4f21e819b330732", ("vectorized_batch_kernel", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("7172672e952d57a46910", ("vectorized_batch_kernel", "bit_parallel_representation", "sort_partition_reduction"), (), None),
        ("695b907772d8a69a1186", ("vectorized_batch_kernel", "bit_parallel_representation", "sparse_frontier_search"), (), None),
    ],
}


def _implementation_class(operators: tuple[str, ...], learned_template: str | None) -> str:
    if learned_template == "reduced_representation_refinement":
        return "learned_sparse_reduced_staged"
    if learned_template == "bit_frontier_restriction":
        return "learned_frontier_then_reference"
    if learned_template == "certified_active_core":
        return "learned_active_core_then_reference"
    ops = set(operators)
    if "dtype_specialization" in ops:
        return "generic_dense_reduced_float32_staged" if "risk_aware_staging" in ops else "generic_dense_reduced_float32"
    if "sparse_frontier_search" in ops or "bit_parallel_representation" in ops:
        if ops.intersection({"reduced_representation", "bounded_exact_refinement", "native_backend_substitution", "vectorized_batch_kernel", "sort_partition_reduction"}):
            return "generic_frontier_then_dense_reduced"
        return "generic_frontier_then_reference"
    if ops.intersection({"reduced_representation", "bounded_exact_refinement", "native_backend_substitution", "vectorized_batch_kernel", "sort_partition_reduction"}):
        return "generic_dense_reduced_staged" if "risk_aware_staging" in ops else "generic_dense_reduced"
    return "reference_exact"


CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {}
PROVENANCE: dict[str, list[dict[str, object]]] = {}
for arm, rows in PROPOSALS.items():
    CANDIDATES_BY_ARM[arm] = []
    PROVENANCE[arm] = []
    for rank, (proposal_id, operators, transfer_ids, learned_template) in enumerate(rows, 1):
        name = f"{arm}_r{rank}_{proposal_id}"
        fn = _implementation(operators, learned_template)
        impl = _implementation_class(operators, learned_template)
        CANDIDATES_BY_ARM[arm].append((name, fn))
        PROVENANCE[arm].append({
            "candidate": name,
            "proposal_id": proposal_id,
            "rank": rank,
            "operators": list(operators),
            "transfer_ids": list(transfer_ids),
            "learned_template": learned_template,
            "implementation_class": impl,
            "semantic_signature": ["learned" if learned_template else "generic", learned_template or "none", list(operators), impl],
        })

if sum(len(rows) for rows in CANDIDATES_BY_ARM.values()) != 30:
    raise RuntimeError("expected exactly 30 frozen Task 3 candidates")
