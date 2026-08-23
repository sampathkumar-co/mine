from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ortools.sat.python import cp_model

Problem = dict[str, list[list[int]]]
Solution = list[tuple[int, int]]


class SearchBudgetExceeded(RuntimeError):
    pass


def _normalise(problem: Problem) -> tuple[list[list[int]], list[list[int]]]:
    if not isinstance(problem, dict):
        raise ValueError("problem must be a dictionary")
    matrices: list[list[list[int]]] = []
    for key in ("A", "B"):
        raw = problem.get(key)
        if not isinstance(raw, list):
            raise ValueError(f"{key} must be a list")
        n = len(raw)
        matrix: list[list[int]] = []
        for row in raw:
            if not isinstance(row, list) or len(row) != n:
                raise ValueError(f"{key} must be square")
            matrix.append([1 if int(value) else 0 for value in row])
        for i in range(n):
            matrix[i][i] = 0
            for j in range(i + 1, n):
                if matrix[i][j] != matrix[j][i]:
                    raise ValueError(f"{key} must be symmetric")
        matrices.append(matrix)
    return matrices[0], matrices[1]


def _association_masks(A: list[list[int]], B: list[list[int]]) -> tuple[list[tuple[int, int]], list[int]]:
    n, m = len(A), len(B)
    pairs = [(i, p) for i in range(n) for p in range(m)]
    if not pairs:
        return pairs, []
    b_zero: list[int] = [0] * m
    b_one: list[int] = [0] * m
    for p in range(m):
        z = 0
        o = 0
        for q in range(m):
            if q == p:
                continue
            if B[p][q]:
                o |= 1 << q
            else:
                z |= 1 << q
        b_zero[p] = z
        b_one[p] = o
    adjacency = [0] * (n * m)
    for i in range(n):
        for p in range(m):
            mask = 0
            for j in range(n):
                if j == i:
                    continue
                compatible_q = b_one[p] if A[i][j] else b_zero[p]
                mask |= compatible_q << (j * m)
            adjacency[i * m + p] = mask
    return pairs, adjacency


def _greedy_clique(adjacency: list[int], vertices: int, prefer_degree: bool = True) -> list[int]:
    clique: list[int] = []
    remaining = vertices
    while remaining:
        best_v = -1
        best_score = -1
        scan = remaining
        while scan:
            bit = scan & -scan
            v = bit.bit_length() - 1
            score = (adjacency[v] & remaining).bit_count() if prefer_degree else -v
            if score > best_score:
                best_score = score
                best_v = v
            scan ^= bit
        if best_v < 0:
            break
        clique.append(best_v)
        remaining &= adjacency[best_v]
    return clique


def _color_sort(adjacency: list[int], vertices: int) -> tuple[list[int], list[int]]:
    order: list[int] = []
    bounds: list[int] = []
    remaining = vertices
    color = 0
    while remaining:
        color += 1
        available = remaining
        while available:
            bit = available & -available
            v = bit.bit_length() - 1
            order.append(v)
            bounds.append(color)
            remaining &= ~bit
            available &= ~bit
            available &= ~adjacency[v]
    return order, bounds


def _maximum_clique(
    adjacency: list[int],
    *,
    use_coloring: bool,
    degree_prune: bool,
    early_certificate: bool,
    node_budget: int | None,
) -> list[int]:
    count = len(adjacency)
    if count == 0:
        return []
    all_vertices = (1 << count) - 1
    best = _greedy_clique(adjacency, all_vertices, prefer_degree=True)

    if degree_prune and best:
        threshold = len(best)
        keep = 0
        for v in range(count):
            if (adjacency[v] & all_vertices).bit_count() + 1 >= threshold:
                keep |= 1 << v
        all_vertices = keep

    if early_certificate:
        _, root_colors = _color_sort(adjacency, all_vertices)
        upper = root_colors[-1] if root_colors else 0
        if len(best) >= upper:
            return best

    expanded = 0

    def search(vertices: int, clique: list[int]) -> None:
        nonlocal best, expanded
        expanded += 1
        if node_budget is not None and expanded > node_budget:
            raise SearchBudgetExceeded(f"maximum-clique node budget {node_budget} exceeded")
        if not vertices:
            if len(clique) > len(best):
                best = clique.copy()
            return

        if use_coloring:
            order, bounds = _color_sort(adjacency, vertices)
            for pos in range(len(order) - 1, -1, -1):
                if len(clique) + bounds[pos] <= len(best):
                    return
                v = order[pos]
                bit = 1 << v
                if not (vertices & bit):
                    continue
                clique.append(v)
                child = vertices & adjacency[v]
                if child:
                    search(child, clique)
                elif len(clique) > len(best):
                    best = clique.copy()
                clique.pop()
                vertices &= ~bit
        else:
            while vertices:
                if len(clique) + vertices.bit_count() <= len(best):
                    return
                # Sparse-frontier heuristic: branch on the vertex with most compatible
                # remaining neighbors. This changes order only, never feasibility.
                scan = vertices
                v = -1
                score = -1
                while scan:
                    bit = scan & -scan
                    candidate = bit.bit_length() - 1
                    degree = (adjacency[candidate] & vertices).bit_count()
                    if degree > score:
                        score = degree
                        v = candidate
                    scan ^= bit
                bit = 1 << v
                clique.append(v)
                child = vertices & adjacency[v]
                if child:
                    search(child, clique)
                elif len(clique) > len(best):
                    best = clique.copy()
                clique.pop()
                vertices &= ~bit

    search(all_vertices, [])
    return best


def _decode_clique(clique: list[int], pairs: list[tuple[int, int]]) -> Solution:
    return sorted((pairs[index] for index in clique), key=lambda pair: pair[0])


def _solve_association(
    problem: Problem,
    *,
    use_coloring: bool = True,
    degree_prune: bool = False,
    early_certificate: bool = False,
    node_budget: int | None = None,
    risk_fallback: bool = False,
) -> Solution:
    A, B = _normalise(problem)
    pairs, adjacency = _association_masks(A, B)
    try:
        clique = _maximum_clique(
            adjacency,
            use_coloring=use_coloring,
            degree_prune=degree_prune,
            early_certificate=early_certificate,
            node_budget=node_budget,
        )
        return _decode_clique(clique, pairs)
    except SearchBudgetExceeded:
        if not risk_fallback:
            raise
        return _solve_cp_sat_matrices(A, B, vectorized_constraints=False, hint=None)


def _greedy_mapping(A: list[list[int]], B: list[list[int]]) -> Solution:
    n, m = len(A), len(B)
    mapping: Solution = []
    used: set[int] = set()
    for i in range(n):
        candidates = []
        for p in range(m):
            if p in used:
                continue
            if all(A[i][j] == B[p][q] for j, q in mapping):
                score = sum(A[i]) - sum(B[p])
                candidates.append((abs(score), p))
        if candidates:
            _, p = min(candidates)
            mapping.append((i, p))
            used.add(p)
    return mapping


def _solve_cp_sat_matrices(
    A: list[list[int]],
    B: list[list[int]],
    *,
    vectorized_constraints: bool,
    hint: Solution | None,
) -> Solution:
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return []
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}_{p}") for i in range(n) for p in range(m)]

    for i in range(n):
        model.Add(sum(x[i * m + p] for p in range(m)) <= 1)
    for p in range(m):
        model.Add(sum(x[i * m + p] for i in range(n)) <= 1)

    if vectorized_constraints:
        arr_a = np.asarray(A, dtype=np.int8)
        arr_b = np.asarray(B, dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                mismatch = np.argwhere(arr_b != arr_a[i, j])
                for p_raw, q_raw in mismatch:
                    p, q = int(p_raw), int(q_raw)
                    if p != q:
                        model.Add(x[i * m + p] + x[j * m + q] <= 1)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                aij = A[i][j]
                for p in range(m):
                    for q in range(m):
                        if p != q and aij != B[p][q]:
                            model.Add(x[i * m + p] + x[j * m + q] <= 1)

    objective = sum(x)
    model.Maximize(objective)
    if hint:
        hinted = set(hint)
        for i in range(n):
            for p in range(m):
                model.AddHint(x[i * m + p], 1 if (i, p) in hinted else 0)
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"CP-SAT status {int(status)} is not OPTIMAL")
    return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i * m + p]) == 1]


def _solve_cp_sat(problem: Problem, *, vectorized_constraints: bool = False, use_hint: bool = False) -> Solution:
    A, B = _normalise(problem)
    hint = _greedy_mapping(A, B) if use_hint else None
    return _solve_cp_sat_matrices(A, B, vectorized_constraints=vectorized_constraints, hint=hint)


def _solve_early_then_cp_sat(problem: Problem) -> Solution:
    A, B = _normalise(problem)
    if not A or not B:
        return []
    greedy = _greedy_mapping(A, B)
    if len(greedy) == min(len(A), len(B)):
        return greedy
    return _solve_cp_sat_matrices(A, B, vectorized_constraints=False, hint=greedy)


# v4_full: exact implementations of the six frozen compositions.
def v4_bit_frontier_risk(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=True, degree_prune=False, early_certificate=False, node_budget=250_000, risk_fallback=True)


def v4_bit_closed_risk(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=True, degree_prune=True, early_certificate=False, node_budget=250_000, risk_fallback=True)


def v4_bit_frontier_closed(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=True, degree_prune=True, early_certificate=False, node_budget=None, risk_fallback=False)


def v4_bit_risk_early(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=True, degree_prune=False, early_certificate=True, node_budget=250_000, risk_fallback=True)


def v4_bit_frontier_early(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=True, degree_prune=False, early_certificate=True, node_budget=None, risk_fallback=False)


def v4_bit_closed_early(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=True, degree_prune=True, early_certificate=True, node_budget=None, risk_fallback=False)


# No-transfer proposals are byte-for-byte the same operator sequences; keep separate
# callables so the arm remains independently timed and selected.
def no_transfer_bit_frontier_risk(problem: Problem) -> Solution:
    return v4_bit_frontier_risk(problem)


def no_transfer_bit_closed_risk(problem: Problem) -> Solution:
    return v4_bit_closed_risk(problem)


def no_transfer_bit_frontier_closed(problem: Problem) -> Solution:
    return v4_bit_frontier_closed(problem)


def no_transfer_bit_risk_early(problem: Problem) -> Solution:
    return v4_bit_risk_early(problem)


def no_transfer_bit_frontier_early(problem: Problem) -> Solution:
    return v4_bit_frontier_early(problem)


def no_transfer_bit_closed_early(problem: Problem) -> Solution:
    return v4_bit_closed_early(problem)


# Random-search arm: implement exactly the six frozen random compositions.
def random_contiguous_layout(problem: Problem) -> Solution:
    return _solve_cp_sat(problem, vectorized_constraints=False, use_hint=False)


def random_zero_sparse_risk(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=False, degree_prune=False, early_certificate=False, node_budget=100_000, risk_fallback=True)


def random_contiguous_dtype_vector(problem: Problem) -> Solution:
    return _solve_cp_sat(problem, vectorized_constraints=True, use_hint=False)


def random_zero_closed_early(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=False, degree_prune=True, early_certificate=True, node_budget=None, risk_fallback=False)


def random_zero_risk_early(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=False, degree_prune=False, early_certificate=True, node_budget=100_000, risk_fallback=True)


def random_sparse_early(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=False, degree_prune=False, early_certificate=True, node_budget=None, risk_fallback=False)


# Template arm: one generic operator per candidate.
def template_bit_parallel(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=True, degree_prune=False, early_certificate=False, node_budget=None, risk_fallback=False)


def template_risk_stage(problem: Problem) -> Solution:
    return _solve_cp_sat(problem, vectorized_constraints=False, use_hint=True)


def template_sparse_frontier(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=False, degree_prune=False, early_certificate=False, node_budget=None, risk_fallback=False)


def template_closed_form(problem: Problem) -> Solution:
    return _solve_association(problem, use_coloring=True, degree_prune=True, early_certificate=False, node_budget=None, risk_fallback=False)


def template_early_certificate(problem: Problem) -> Solution:
    return _solve_early_then_cp_sat(problem)


def template_vectorized_batch(problem: Problem) -> Solution:
    return _solve_cp_sat(problem, vectorized_constraints=True, use_hint=False)


# v3-compatible: only shallow wrapper/layout/backend changes.
def v3_vectorized_batch(problem: Problem) -> Solution:
    return _solve_cp_sat(problem, vectorized_constraints=True, use_hint=False)


def v3_zero_copy(problem: Problem) -> Solution:
    return _solve_cp_sat(problem, vectorized_constraints=False, use_hint=False)


def v3_dtype_specialization(problem: Problem) -> Solution:
    A, B = _normalise(problem)
    A = np.asarray(A, dtype=np.int8).tolist()
    B = np.asarray(B, dtype=np.int8).tolist()
    return _solve_cp_sat_matrices(A, B, vectorized_constraints=False, hint=None)


def v3_contiguous_layout(problem: Problem) -> Solution:
    A, B = _normalise(problem)
    A = np.ascontiguousarray(A, dtype=np.int8).tolist()
    B = np.ascontiguousarray(B, dtype=np.int8).tolist()
    return _solve_cp_sat_matrices(A, B, vectorized_constraints=False, hint=None)


CANDIDATES_BY_ARM: dict[str, dict[str, Callable[[Problem], Solution]]] = {
    "v4_full": {
        "v4_bit_frontier_risk": v4_bit_frontier_risk,
        "v4_bit_closed_risk": v4_bit_closed_risk,
        "v4_bit_frontier_closed": v4_bit_frontier_closed,
        "v4_bit_risk_early": v4_bit_risk_early,
        "v4_bit_frontier_early": v4_bit_frontier_early,
        "v4_bit_closed_early": v4_bit_closed_early,
    },
    "v4_no_transfer": {
        "no_transfer_bit_frontier_risk": no_transfer_bit_frontier_risk,
        "no_transfer_bit_closed_risk": no_transfer_bit_closed_risk,
        "no_transfer_bit_frontier_closed": no_transfer_bit_frontier_closed,
        "no_transfer_bit_risk_early": no_transfer_bit_risk_early,
        "no_transfer_bit_frontier_early": no_transfer_bit_frontier_early,
        "no_transfer_bit_closed_early": no_transfer_bit_closed_early,
    },
    "random_search": {
        "random_contiguous_layout": random_contiguous_layout,
        "random_zero_sparse_risk": random_zero_sparse_risk,
        "random_contiguous_dtype_vector": random_contiguous_dtype_vector,
        "random_zero_closed_early": random_zero_closed_early,
        "random_zero_risk_early": random_zero_risk_early,
        "random_sparse_early": random_sparse_early,
    },
    "template_synthesis": {
        "template_bit_parallel": template_bit_parallel,
        "template_risk_stage": template_risk_stage,
        "template_sparse_frontier": template_sparse_frontier,
        "template_closed_form": template_closed_form,
        "template_early_certificate": template_early_certificate,
        "template_vectorized_batch": template_vectorized_batch,
    },
    "v3_compatible": {
        "v3_vectorized_batch": v3_vectorized_batch,
        "v3_zero_copy": v3_zero_copy,
        "v3_dtype_specialization": v3_dtype_specialization,
        "v3_contiguous_layout": v3_contiguous_layout,
    },
}
