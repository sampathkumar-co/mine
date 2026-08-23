from __future__ import annotations

from functools import lru_cache
from typing import Callable

import hdbscan
import numpy as np
from sklearn.neighbors import NearestNeighbors

Problem = dict[str, object]
Solution = dict[str, object]
Candidate = Callable[[Problem], Solution]


def _problem(problem: Problem, dtype: np.dtype | type = np.float64, contiguous: bool = False) -> tuple[np.ndarray, int, int]:
    x = np.asarray(problem["dataset"], dtype=dtype)
    if contiguous:
        x = np.ascontiguousarray(x)
    if x.ndim != 2 or x.shape[0] < 2 or not np.all(np.isfinite(x)):
        raise ValueError("invalid clustering dataset")
    min_cluster_size = max(2, int(problem.get("min_cluster_size", 5)))
    min_samples = max(1, int(problem.get("min_samples", 3)))
    return x, min_cluster_size, min_samples


def _solution(labels: np.ndarray, probabilities: np.ndarray | None = None, persistence: np.ndarray | None = None) -> Solution:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probabilities is None:
        probabilities = np.where(labels >= 0, 1.0, 0.0)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0, 1.0)
    clusters = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    if persistence is None:
        persistence = np.ones(len(clusters), dtype=np.float64)
    return {
        "labels": labels.tolist(),
        "probabilities": np.asarray(persistence * 0 + probabilities[: len(labels)] if False else probabilities, dtype=np.float64).tolist(),
        "cluster_persistence": np.clip(np.asarray(persistence, dtype=np.float64).reshape(-1), 0.0, 1.0).tolist(),
        "num_clusters": len(clusters),
        "num_noise_points": int(np.sum(labels == -1)),
    }


def _exact_hdbscan(problem: Problem, *, dtype: np.dtype | type = np.float64, contiguous: bool = False) -> Solution:
    x, min_cluster_size, min_samples = _problem(problem, dtype=dtype, contiguous=contiguous)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(x)
    return _solution(clusterer.labels_, clusterer.probabilities_, clusterer.cluster_persistence_)


def _neighbor_data(x: np.ndarray, min_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = min(x.shape[0], max(2, min_samples + 1))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1)
    nn.fit(x)
    distances, indices = nn.kneighbors(x, return_distance=True)
    return distances, indices, distances[:, -1]


def _graph_frontier(problem: Problem, *, certified: bool) -> Solution:
    x, min_cluster_size, min_samples = _problem(problem, dtype=np.float64, contiguous=True)
    n = x.shape[0]
    distances, indices, kth = _neighbor_data(x, min_samples)
    eps = float(np.quantile(kth, 0.80))
    if not np.isfinite(eps) or eps <= 0.0:
        return _exact_hdbscan(problem)
    core = kth <= eps
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for i in range(n):
        if not core[i]:
            continue
        for distance, j_raw in zip(distances[i, 1:], indices[i, 1:]):
            j = int(j_raw)
            if core[j] and float(distance) <= eps:
                union(i, j)

    components: dict[int, list[int]] = {}
    for i in np.flatnonzero(core):
        components.setdefault(find(int(i)), []).append(int(i))
    kept = [members for members in components.values() if len(members) >= min_cluster_size]
    kept.sort(key=lambda members: min(members))
    labels = np.full(n, -1, dtype=np.int64)
    for label, members in enumerate(kept):
        labels[np.asarray(members, dtype=np.int64)] = label

    probabilities = np.zeros(n, dtype=np.float64)
    probabilities[labels >= 0] = 1.0
    for i in np.flatnonzero(labels < 0):
        best_label = -1
        best_distance = float("inf")
        for distance, j_raw in zip(distances[i, 1:], indices[i, 1:]):
            label = int(labels[int(j_raw)])
            if label >= 0 and float(distance) < best_distance:
                best_label = label
                best_distance = float(distance)
        if best_label >= 0 and best_distance <= eps:
            labels[i] = best_label
            probabilities[i] = max(0.0, 1.0 - best_distance / (eps + 1e-12))

    result = _solution(labels, probabilities)
    if certified and not _structural_certificate(x, labels, min_cluster_size):
        return _exact_hdbscan(problem)
    return result


def _cluster_subset_and_lift(problem: Problem, subset: np.ndarray, *, certified: bool) -> Solution:
    x, min_cluster_size, min_samples = _problem(problem, dtype=np.float64, contiguous=True)
    n = x.shape[0]
    subset = np.unique(np.asarray(subset, dtype=np.int64))
    subset = subset[(subset >= 0) & (subset < n)]
    if len(subset) < max(8, min_cluster_size * 2):
        return _exact_hdbscan(problem)
    scale = len(subset) / n
    sub_mcs = max(2, min(len(subset) - 1, int(round(min_cluster_size * scale))))
    sub_ms = max(1, min(sub_mcs, int(round(min_samples * scale))))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=sub_mcs, min_samples=sub_ms)
    clusterer.fit(x[subset])
    sub_labels = np.asarray(clusterer.labels_, dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    labels[subset] = sub_labels
    probabilities = np.zeros(n, dtype=np.float64)
    probabilities[subset] = np.asarray(clusterer.probabilities_, dtype=np.float64)

    clusters = sorted(int(v) for v in np.unique(sub_labels) if int(v) >= 0)
    if not clusters:
        return _exact_hdbscan(problem) if certified else _solution(labels, probabilities)
    centroids: list[np.ndarray] = []
    radii: list[float] = []
    cluster_ids: list[int] = []
    for label in clusters:
        points = x[subset[sub_labels == label]]
        if len(points) == 0:
            continue
        centroid = np.mean(points, axis=0)
        d = np.linalg.norm(points - centroid, axis=1)
        radius = float(np.quantile(d, 0.95)) if len(d) > 1 else float(d[0] + 1e-12)
        centroids.append(centroid)
        radii.append(max(radius, 1e-12))
        cluster_ids.append(label)
    if not centroids:
        return _exact_hdbscan(problem) if certified else _solution(labels, probabilities)
    centroid_matrix = np.vstack(centroids)
    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), subset, assume_unique=True)
    for i in remaining:
        d = np.linalg.norm(centroid_matrix - x[i], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= 1.25 * radii[j]:
            labels[i] = cluster_ids[j]
            probabilities[i] = max(0.0, 1.0 - float(d[j]) / (1.25 * radii[j] + 1e-12))

    result = _solution(labels, probabilities, np.asarray(clusterer.cluster_persistence_, dtype=np.float64))
    if certified and not _structural_certificate(x, labels, min_cluster_size):
        return _exact_hdbscan(problem)
    return result


def _active_core(problem: Problem, *, certified: bool) -> Solution:
    x, min_cluster_size, min_samples = _problem(problem, dtype=np.float64, contiguous=True)
    _, _, kth = _neighbor_data(x, min_samples)
    cutoff = float(np.quantile(kth, 0.85 if certified else 0.90))
    subset = np.flatnonzero(kth <= cutoff)
    return _cluster_subset_and_lift(problem, subset, certified=certified)


def _reduced_representation(problem: Problem, *, certified: bool) -> Solution:
    x, min_cluster_size, _ = _problem(problem, dtype=np.float64, contiguous=True)
    n = x.shape[0]
    stride = 2
    subset = np.arange(0, n, stride, dtype=np.int64)
    if len(subset) < max(8, min_cluster_size * 2):
        subset = np.arange(n, dtype=np.int64)
    return _cluster_subset_and_lift(problem, subset, certified=certified)


def _structural_certificate(x: np.ndarray, labels: np.ndarray, min_cluster_size: int) -> bool:
    clusters = [int(v) for v in np.unique(labels) if int(v) >= 0]
    if not clusters:
        return False
    noise_ratio = float(np.mean(labels == -1))
    if noise_ratio > 0.45:
        return False
    centroids: list[np.ndarray] = []
    within: list[float] = []
    for label in clusters:
        points = x[labels == label]
        if len(points) < max(2, min_cluster_size // 2):
            return False
        centroid = np.mean(points, axis=0)
        centroids.append(centroid)
        within.extend(np.linalg.norm(points - centroid, axis=1).tolist())
    if len(centroids) <= 1:
        return True
    c = np.vstack(centroids)
    inter = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    inter[inter == 0] = np.inf
    separation = float(np.min(inter))
    spread = float(np.median(within)) if within else 0.0
    return separation >= 1.5 * max(spread, 1e-12)


def _generic_pipeline(problem: Problem, operators: tuple[str, ...]) -> Solution:
    ops = set(operators)
    if "reduced_representation" in ops:
        return _reduced_representation(problem, certified=False)
    if "active_set_decomposition" in ops or "sort_partition_reduction" in ops:
        return _active_core(problem, certified=False)
    if "sparse_frontier_search" in ops:
        return _graph_frontier(problem, certified=False)
    dtype = np.float32 if "dtype_specialization" in ops else np.float64
    contiguous = "contiguous_layout" in ops or "vectorized_batch_kernel" in ops
    return _exact_hdbscan(problem, dtype=dtype, contiguous=contiguous)


def _learned_pipeline(problem: Problem, learned_template: str) -> Solution:
    if learned_template == "bit_frontier_restriction":
        return _graph_frontier(problem, certified=True)
    if learned_template == "certified_active_core":
        return _active_core(problem, certified=True)
    if learned_template == "reduced_representation_refinement":
        return _reduced_representation(problem, certified=True)
    raise ValueError(f"unknown learned template: {learned_template}")


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
        ("3304c859d463a501bd86", ("bit_parallel_representation","sparse_frontier_search","early_certificate_exit"), ("TM-BFR-01",), "bit_frontier_restriction"),
        ("41510e43e8fafb598496", ("reduced_representation","bounded_exact_refinement","risk_aware_staging"), ("TM-RRR-01",), "reduced_representation_refinement"),
        ("a6102573c9f355414229", ("active_set_decomposition","early_certificate_exit","risk_aware_staging"), ("TM-CAC-01",), "certified_active_core"),
        ("4abf2b51384c560522e8", ("active_set_decomposition","bit_parallel_representation","bounded_exact_refinement"), (), None),
        ("c50e493c5549a408f3e5", ("active_set_decomposition","bit_parallel_representation","sparse_frontier_search"), (), None),
        ("514b3e8a41ba1f8b73a1", ("active_set_decomposition","bit_parallel_representation","reduced_representation"), (), None),
    ],
    "v5_no_transfer": [
        ("91e027e622f2d9a98240", ("active_set_decomposition","bit_parallel_representation","bounded_exact_refinement"), (), None),
        ("b2614109e1a5ccc10c14", ("active_set_decomposition","bit_parallel_representation","sparse_frontier_search"), (), None),
        ("20375ceceffce4d406a4", ("active_set_decomposition","bit_parallel_representation","reduced_representation"), (), None),
        ("4a4e1871b7f7b48b9485", ("active_set_decomposition","bit_parallel_representation","sort_partition_reduction"), (), None),
        ("d69e86803f54c5a83d06", ("vectorized_batch_kernel","active_set_decomposition","bit_parallel_representation"), (), None),
        ("66c5848a3c8a4f51b562", ("bit_parallel_representation","reduced_representation","bounded_exact_refinement"), (), None),
    ],
    "random_search": [
        ("281d4a03f9bc5812f7af", ("dtype_specialization","vectorized_batch_kernel","bit_parallel_representation"), (), None),
        ("0d6f85272e04650e490b", ("zero_copy_representation","vectorized_batch_kernel","active_set_decomposition"), (), None),
        ("38776670db84b717ed92", ("zero_copy_representation","risk_aware_staging","reduced_representation"), (), None),
        ("c7951713adcdd3e83e67", ("zero_copy_representation","dtype_specialization","sort_partition_reduction"), (), None),
        ("57b17f2971c60d7d437b", ("bounded_exact_refinement","sort_partition_reduction"), (), None),
        ("5fef7c7ae40af62d92e4", ("dtype_specialization","sparse_frontier_search"), (), None),
    ],
    "static_template": [
        ("dbfcd2af539b0b2636e7", ("bit_parallel_representation","sparse_frontier_search"), (), None),
        ("8fd871e046faa7e4d37c", ("reduced_representation","bounded_exact_refinement"), (), None),
        ("820b1c309b6117eb268d", ("active_set_decomposition","early_certificate_exit"), (), None),
        ("8f1dafda0d3fbc099aa9", ("zero_copy_representation","vectorized_batch_kernel"), (), None),
        ("357e80313b8b9dc3cf36", ("contiguous_layout","vectorized_batch_kernel"), (), None),
        ("d044a19fd4551034dc11", ("dtype_specialization","risk_aware_staging"), (), None),
    ],
    "v4_compatible": [
        ("f9f3239b6866512e4f68", ("active_set_decomposition","bit_parallel_representation","bounded_exact_refinement"), (), None),
        ("9f5f55df04a5ad23f542", ("active_set_decomposition","bit_parallel_representation","sparse_frontier_search"), (), None),
        ("ec4b9c17aaa3767d4f6", ("active_set_decomposition","bit_parallel_representation","sort_partition_reduction"), (), None),
        ("7c30efb65d2c20ff8cc9", ("vectorized_batch_kernel","active_set_decomposition","bit_parallel_representation"), (), None),
        ("3df5ed91505aea4ed6cb", ("active_set_decomposition","sparse_frontier_search","bounded_exact_refinement"), (), None),
        ("0dde88a4a159a3ad0e40", ("bit_parallel_representation","bounded_exact_refinement","sort_partition_reduction"), (), None),
    ],
}

CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {}
PROVENANCE: dict[str, list[dict[str, object]]] = {}
for arm, rows in PROPOSALS.items():
    CANDIDATES_BY_ARM[arm] = []
    PROVENANCE[arm] = []
    for rank, (proposal_id, operators, transfer_ids, learned_template) in enumerate(rows, 1):
        name = f"{arm}_r{rank}_{proposal_id}"
        fn = _implementation(operators, learned_template)
        CANDIDATES_BY_ARM[arm].append((name, fn))
        PROVENANCE[arm].append({
            "candidate": name,
            "proposal_id": proposal_id,
            "rank": rank,
            "operators": list(operators),
            "transfer_ids": list(transfer_ids),
            "learned_template": learned_template,
            "semantic_signature": ["learned" if learned_template else "generic", learned_template or "none", list(operators)],
        })

if sum(len(rows) for rows in CANDIDATES_BY_ARM.values()) != 30:
    raise RuntimeError("expected exactly 30 frozen Task 1 candidates")
