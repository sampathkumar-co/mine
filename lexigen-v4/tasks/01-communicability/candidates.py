from __future__ import annotations

import math
from collections.abc import Callable

import networkx as nx
import numpy as np

Problem = dict[str, object]
Solution = dict[str, dict[int, dict[int, float]]]


def _adjacency_list(problem: Problem) -> list[list[int]]:
    value = problem.get("adjacency_list")
    if not isinstance(value, list):
        raise ValueError("problem adjacency_list must be a list")
    result: list[list[int]] = []
    for neighbors in value:
        if not isinstance(neighbors, list):
            raise ValueError("each adjacency row must be a list")
        result.append([int(vertex) for vertex in neighbors])
    return result


def _adjacency(problem: Problem, dtype: np.dtype[np.floating] | type[np.floating] = np.float64) -> np.ndarray:
    adjacency_list = _adjacency_list(problem)
    n = len(adjacency_list)
    matrix = np.zeros((n, n), dtype=dtype)
    for u, neighbors in enumerate(adjacency_list):
        if neighbors:
            matrix[u, np.asarray(neighbors, dtype=np.intp)] = 1.0
    if n:
        matrix = np.maximum(matrix, matrix.T)
        np.fill_diagonal(matrix, 0.0)
    return matrix


def _to_solution(matrix: np.ndarray) -> Solution:
    n = int(matrix.shape[0])
    rows = np.asarray(matrix, dtype=np.float64).tolist()
    return {
        "communicability": {
            u: {v: float(rows[u][v]) for v in range(n)}
            for u in range(n)
        }
    }


def _spectral_vectorized(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    result = (eigenvectors * np.exp(eigenvalues)[None, :]) @ eigenvectors.T
    return np.asarray((result + result.T) * 0.5, dtype=np.float64)


def _spectral_float32(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    reduced = np.asarray(matrix, dtype=np.float32)
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    result = (eigenvectors * np.exp(eigenvalues)[None, :]) @ eigenvectors.T
    return np.asarray((result + result.T) * np.float32(0.5), dtype=np.float64)


def _spectral_loop(matrix: np.ndarray) -> np.ndarray:
    n = int(matrix.shape[0])
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    weights = np.exp(eigenvalues)
    output = np.empty((n, n), dtype=np.float64)
    for u in range(n):
        for v in range(n):
            total = 0.0
            for j in range(n):
                total += float(eigenvectors[u, j] * eigenvectors[v, j] * weights[j])
            output[u, v] = total
    return output


def _networkx_reference(problem: Problem) -> Solution:
    adjacency_list = _adjacency_list(problem)
    n = len(adjacency_list)
    if n == 0:
        return {"communicability": {}}
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for u, neighbors in enumerate(adjacency_list):
        for v in neighbors:
            if u < v:
                graph.add_edge(u, v)
    raw = nx.communicability(graph)
    return {
        "communicability": {
            u: {v: float(raw[u][v]) for v in range(n)}
            for u in range(n)
        }
    }


def _sparse_components(adjacency_list: list[list[int]]) -> list[list[int]]:
    n = len(adjacency_list)
    unseen = set(range(n))
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adjacency_list[u]:
                if v in unseen:
                    unseen.remove(v)
                    stack.append(v)
        components.append(sorted(component))
    return components


def _bit_components(adjacency_list: list[list[int]]) -> list[list[int]]:
    n = len(adjacency_list)
    if n == 0:
        return []
    neighborhood_masks: list[int] = []
    for u, neighbors in enumerate(adjacency_list):
        mask = 1 << u
        for v in neighbors:
            mask |= 1 << int(v)
        neighborhood_masks.append(mask)

    remaining = (1 << n) - 1
    components: list[list[int]] = []
    while remaining:
        seed = remaining & -remaining
        frontier = seed
        component_mask = 0
        while frontier:
            component_mask |= frontier
            remaining &= ~frontier
            neighbors_mask = 0
            work = frontier
            while work:
                bit = work & -work
                work -= bit
                vertex = bit.bit_length() - 1
                neighbors_mask |= neighborhood_masks[vertex]
            frontier = neighbors_mask & remaining
        component: list[int] = []
        work = component_mask
        while work:
            bit = work & -work
            work -= bit
            component.append(bit.bit_length() - 1)
        components.append(component)
    return components


def _component_matrix(
    problem: Problem,
    *,
    component_mode: str,
    vectorized: bool,
    degree_order: bool = False,
    early_exact: bool = False,
    float32: bool = False,
    risk_stage: bool = False,
) -> np.ndarray:
    adjacency_list = _adjacency_list(problem)
    n = len(adjacency_list)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    if component_mode == "bit":
        components = _bit_components(adjacency_list)
    elif component_mode == "sparse":
        components = _sparse_components(adjacency_list)
    else:
        raise ValueError(f"unknown component mode: {component_mode}")

    matrix = np.ascontiguousarray(_adjacency(problem, np.float64))
    if risk_stage and len(components) == 1:
        return _spectral_float32(matrix) if float32 else _spectral_vectorized(matrix)

    output = np.zeros((n, n), dtype=np.float64)
    for component in components:
        order = list(component)
        if degree_order:
            order.sort(key=lambda vertex: (-len(adjacency_list[vertex]), vertex))
        size = len(order)
        if early_exact and size == 1:
            output[order[0], order[0]] = 1.0
            continue
        if early_exact and size == 2:
            u, v = order
            output[u, u] = math.cosh(1.0)
            output[v, v] = math.cosh(1.0)
            output[u, v] = math.sinh(1.0)
            output[v, u] = math.sinh(1.0)
            continue
        submatrix = np.ascontiguousarray(matrix[np.ix_(order, order)])
        if float32:
            block = _spectral_float32(submatrix)
        elif vectorized:
            block = _spectral_vectorized(submatrix)
        else:
            block = _spectral_loop(submatrix)
        output[np.ix_(order, order)] = block
    return output


def _degree_order_full(problem: Problem, vectorized: bool) -> np.ndarray:
    adjacency_list = _adjacency_list(problem)
    n = len(adjacency_list)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    order = sorted(range(n), key=lambda vertex: (-len(adjacency_list[vertex]), vertex))
    matrix = _adjacency(problem, np.float64)
    reordered = np.ascontiguousarray(matrix[np.ix_(order, order)])
    result = _spectral_vectorized(reordered) if vectorized else _spectral_loop(reordered)
    inverse = np.empty(n, dtype=np.intp)
    inverse[np.asarray(order, dtype=np.intp)] = np.arange(n, dtype=np.intp)
    return result[np.ix_(inverse, inverse)]


# v4_full: exact mappings from the six frozen proposal compositions.
def v4_bit_sparse_risk(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="bit", vectorized=True, risk_stage=True))


def v4_sort_bit_sparse(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="bit", vectorized=True, degree_order=True))


def v4_bit_sparse_vector(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="sparse", vectorized=True))


def v4_sort_bit_risk(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="bit", vectorized=True, degree_order=True, risk_stage=True))


def v4_bit_sparse_early(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="sparse", vectorized=False, early_exact=True))


def v4_bit_vector_risk(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="bit", vectorized=True, early_exact=True, risk_stage=True))


# v4_no_transfer generated the same mechanism ordering on Task 1; aliases preserve that negative ablation result.
def no_transfer_bit_sparse_risk(problem: Problem) -> Solution:
    return v4_bit_sparse_risk(problem)


def no_transfer_sort_bit_sparse(problem: Problem) -> Solution:
    return v4_sort_bit_sparse(problem)


def no_transfer_bit_sparse_vector(problem: Problem) -> Solution:
    return v4_bit_sparse_vector(problem)


def no_transfer_sort_bit_risk(problem: Problem) -> Solution:
    return v4_sort_bit_risk(problem)


def no_transfer_bit_sparse_early(problem: Problem) -> Solution:
    return v4_bit_sparse_early(problem)


def no_transfer_bit_vector_risk(problem: Problem) -> Solution:
    return v4_bit_vector_risk(problem)


# Random-search arm, in its frozen proposal order.
def random_bit_sparse_risk(problem: Problem) -> Solution:
    return v4_bit_sparse_risk(problem)


def random_contiguous_vector_risk(problem: Problem) -> Solution:
    matrix = np.ascontiguousarray(_adjacency(problem, np.float64))
    return _to_solution(_spectral_vectorized(matrix))


def random_float32_sparse(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="sparse", vectorized=True, float32=True))


def random_bit_sparse(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="bit", vectorized=False))


def random_contiguous_bit(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="bit", vectorized=False, degree_order=True))


def random_contiguous(problem: Problem) -> Solution:
    return _to_solution(_spectral_loop(np.ascontiguousarray(_adjacency(problem, np.float64))))


# Single-operator deterministic template arm. The unchanged reference kernel completes semantics when the operator alone is insufficient.
def template_bit_parallel(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="bit", vectorized=False))


def template_sparse_frontier(problem: Problem) -> Solution:
    return _to_solution(_component_matrix(problem, component_mode="sparse", vectorized=False))


def template_risk_stage(problem: Problem) -> Solution:
    return _networkx_reference(problem)


def template_sort_partition(problem: Problem) -> Solution:
    return _to_solution(_degree_order_full(problem, vectorized=False))


def template_vectorized_batch(problem: Problem) -> Solution:
    return _to_solution(_spectral_vectorized(_adjacency(problem, np.float64)))


def template_early_certificate(problem: Problem) -> Solution:
    adjacency_list = _adjacency_list(problem)
    if not adjacency_list:
        return {"communicability": {}}
    if all(not neighbors for neighbors in adjacency_list):
        return _to_solution(np.eye(len(adjacency_list), dtype=np.float64))
    return _networkx_reference(problem)


# Reproducible v3-compatible arm.
def v3_vectorized_batch(problem: Problem) -> Solution:
    return _to_solution(_spectral_vectorized(_adjacency(problem, np.float64)))


def v3_zero_copy_representation(problem: Problem) -> Solution:
    return _to_solution(_spectral_loop(_adjacency(problem, np.float64)))


def v3_dtype_specialization(problem: Problem) -> Solution:
    return _to_solution(_spectral_float32(_adjacency(problem, np.float32)))


def v3_contiguous_layout(problem: Problem) -> Solution:
    return _to_solution(_spectral_loop(np.ascontiguousarray(_adjacency(problem, np.float64))))


CANDIDATES_BY_ARM: dict[str, dict[str, Callable[[Problem], Solution]]] = {
    "v4_full": {
        "v4_bit_sparse_risk": v4_bit_sparse_risk,
        "v4_sort_bit_sparse": v4_sort_bit_sparse,
        "v4_bit_sparse_vector": v4_bit_sparse_vector,
        "v4_sort_bit_risk": v4_sort_bit_risk,
        "v4_bit_sparse_early": v4_bit_sparse_early,
        "v4_bit_vector_risk": v4_bit_vector_risk,
    },
    "v4_no_transfer": {
        "no_transfer_bit_sparse_risk": no_transfer_bit_sparse_risk,
        "no_transfer_sort_bit_sparse": no_transfer_sort_bit_sparse,
        "no_transfer_bit_sparse_vector": no_transfer_bit_sparse_vector,
        "no_transfer_sort_bit_risk": no_transfer_sort_bit_risk,
        "no_transfer_bit_sparse_early": no_transfer_bit_sparse_early,
        "no_transfer_bit_vector_risk": no_transfer_bit_vector_risk,
    },
    "random_search": {
        "random_bit_sparse_risk": random_bit_sparse_risk,
        "random_contiguous_vector_risk": random_contiguous_vector_risk,
        "random_float32_sparse": random_float32_sparse,
        "random_bit_sparse": random_bit_sparse,
        "random_contiguous_bit": random_contiguous_bit,
        "random_contiguous": random_contiguous,
    },
    "template_synthesis": {
        "template_bit_parallel": template_bit_parallel,
        "template_sparse_frontier": template_sparse_frontier,
        "template_risk_stage": template_risk_stage,
        "template_sort_partition": template_sort_partition,
        "template_vectorized_batch": template_vectorized_batch,
        "template_early_certificate": template_early_certificate,
    },
    "v3_compatible": {
        "v3_vectorized_batch": v3_vectorized_batch,
        "v3_zero_copy_representation": v3_zero_copy_representation,
        "v3_dtype_specialization": v3_dtype_specialization,
        "v3_contiguous_layout": v3_contiguous_layout,
    },
}
