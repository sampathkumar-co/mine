from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from ortools.sat.python import cp_model

Problem = list[list[int]]
Solution = list[int]


def _normalise(problem: Problem) -> list[list[int]]:
    if not isinstance(problem, list):
        raise ValueError("problem must be an adjacency matrix list")
    n = len(problem)
    matrix: list[list[int]] = []
    for row in problem:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("adjacency matrix must be square")
        matrix.append([1 if int(value) else 0 for value in row])
    for i in range(n):
        matrix[i][i] = 0
        for j in range(i + 1, n):
            edge = 1 if matrix[i][j] or matrix[j][i] else 0
            matrix[i][j] = edge
            matrix[j][i] = edge
    return matrix


def _closed_masks(matrix: Sequence[Sequence[int]]) -> list[int]:
    masks: list[int] = []
    for i, row in enumerate(matrix):
        mask = 1 << i
        for j, value in enumerate(row):
            if value:
                mask |= 1 << j
        masks.append(mask)
    return masks


def _components(masks: Sequence[int]) -> list[list[int]]:
    n = len(masks)
    remaining = (1 << n) - 1
    components: list[list[int]] = []
    while remaining:
        seed = remaining & -remaining
        frontier = seed
        component_mask = 0
        while frontier:
            component_mask |= frontier
            remaining &= ~frontier
            next_frontier = 0
            work = frontier
            while work:
                bit = work & -work
                work -= bit
                vertex = bit.bit_length() - 1
                next_frontier |= masks[vertex]
            frontier = next_frontier & remaining
        component: list[int] = []
        work = component_mask
        while work:
            bit = work & -work
            work -= bit
            component.append(bit.bit_length() - 1)
        components.append(component)
    return components


def _dominance_reduced(vertices: Sequence[int], masks: Sequence[int]) -> list[int]:
    component_mask = sum(1 << vertex for vertex in vertices)
    active: list[int] = []
    for u in vertices:
        neighborhood_u = masks[u] & component_mask
        removable = False
        for v in vertices:
            if u == v:
                continue
            neighborhood_v = masks[v] & component_mask
            if neighborhood_u & ~neighborhood_v == 0:
                if neighborhood_u != neighborhood_v or v < u:
                    removable = True
                    break
        if not removable:
            active.append(u)
    return active or [min(vertices)]


def _greedy_cover(vertices: Sequence[int], candidates: Sequence[int], masks: Sequence[int]) -> list[int]:
    target = sum(1 << vertex for vertex in vertices)
    uncovered = target
    selected: list[int] = []
    available = list(candidates)
    while uncovered:
        best = max(
            available,
            key=lambda vertex: (
                (masks[vertex] & uncovered).bit_count(),
                (masks[vertex] & target).bit_count(),
                -vertex,
            ),
        )
        if (masks[best] & uncovered) == 0:
            raise RuntimeError("candidate set cannot dominate the component")
        selected.append(best)
        uncovered &= ~masks[best]
    changed = True
    while changed:
        changed = False
        for vertex in list(reversed(selected)):
            trial = [other for other in selected if other != vertex]
            covered = 0
            for other in trial:
                covered |= masks[other]
            if target & ~covered == 0:
                selected.remove(vertex)
                changed = True
    return sorted(selected)


def _lower_bound(vertices: Sequence[int], candidates: Sequence[int], masks: Sequence[int]) -> int:
    target = sum(1 << vertex for vertex in vertices)
    maximum = max((masks[vertex] & target).bit_count() for vertex in candidates)
    return max(1, math.ceil(len(vertices) / maximum))


def _build_model(
    vertices: Sequence[int],
    candidates: Sequence[int],
    masks: Sequence[int],
    *,
    hint: Sequence[int] | None,
    upper_bound: int | None,
) -> tuple[cp_model.CpModel, dict[int, cp_model.IntVar]]:
    model = cp_model.CpModel()
    variables = {vertex: model.NewBoolVar(f"x_{vertex}") for vertex in candidates}
    for dominated in vertices:
        covering = [variables[vertex] for vertex in candidates if masks[vertex] & (1 << dominated)]
        if not covering:
            raise RuntimeError(f"no active candidate dominates vertex {dominated}")
        model.Add(sum(covering) >= 1)
    objective = sum(variables.values())
    if upper_bound is not None:
        model.Add(objective <= int(upper_bound))
    model.Minimize(objective)
    if hint is not None:
        hinted = set(hint)
        for vertex, variable in variables.items():
            model.AddHint(variable, 1 if vertex in hinted else 0)
    return model, variables


def _solve_model(
    vertices: Sequence[int],
    candidates: Sequence[int],
    masks: Sequence[int],
    *,
    use_hint: bool,
    use_upper_bound: bool,
) -> tuple[list[int] | None, int]:
    greedy = _greedy_cover(vertices, candidates, masks) if use_hint or use_upper_bound else None
    model, variables = _build_model(
        vertices,
        candidates,
        masks,
        hint=greedy if use_hint else None,
        upper_bound=len(greedy) if use_upper_bound and greedy is not None else None,
    )
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        return None, int(status)
    return sorted(vertex for vertex, variable in variables.items() if solver.Value(variable) == 1), int(status)


def _solve_component(
    vertices: Sequence[int],
    masks: Sequence[int],
    *,
    dominance: bool,
    early_certificate: bool,
    use_hint: bool,
    use_upper_bound: bool,
    risk_fallback: bool,
) -> list[int]:
    if not vertices:
        return []
    if len(vertices) == 1:
        return [vertices[0]]
    component_mask = sum(1 << vertex for vertex in vertices)
    for vertex in vertices:
        if masks[vertex] & component_mask == component_mask:
            return [vertex]

    candidates = _dominance_reduced(vertices, masks) if dominance else list(vertices)
    greedy = _greedy_cover(vertices, candidates, masks)
    if early_certificate and len(greedy) == _lower_bound(vertices, candidates, masks):
        return greedy

    selected, _ = _solve_model(
        vertices,
        candidates,
        masks,
        use_hint=use_hint,
        use_upper_bound=use_upper_bound,
    )
    if selected is not None:
        return selected
    if not risk_fallback or set(candidates) == set(vertices):
        raise RuntimeError("exact reduced CP-SAT model did not return OPTIMAL")
    selected, _ = _solve_model(
        vertices,
        list(vertices),
        masks,
        use_hint=use_hint,
        use_upper_bound=use_upper_bound,
    )
    if selected is None:
        raise RuntimeError("exact fallback CP-SAT model did not return OPTIMAL")
    return selected


def _solve_bitset(
    problem: Problem,
    *,
    decompose: bool,
    dominance: bool,
    early_certificate: bool,
    use_hint: bool,
    use_upper_bound: bool,
    risk_fallback: bool,
) -> Solution:
    matrix = _normalise(problem)
    masks = _closed_masks(matrix)
    groups = _components(masks) if decompose else [list(range(len(matrix)))]
    selected: list[int] = []
    for vertices in groups:
        selected.extend(
            _solve_component(
                vertices,
                masks,
                dominance=dominance,
                early_certificate=early_certificate,
                use_hint=use_hint,
                use_upper_bound=use_upper_bound,
                risk_fallback=risk_fallback,
            )
        )
    return sorted(selected)


def _solve_numpy(
    problem: Problem,
    *,
    dtype: np.dtype[np.integer] | type[np.integer],
    contiguous: bool,
    dominance: bool,
    use_hint: bool,
    risk_fallback: bool,
) -> Solution:
    matrix = np.asarray(_normalise(problem), dtype=dtype)
    if contiguous:
        matrix = np.ascontiguousarray(matrix)
    n = int(matrix.shape[0])
    masks: list[int] = []
    for i in range(n):
        neighbors = np.flatnonzero(matrix[i]).tolist()
        mask = 1 << i
        for vertex in neighbors:
            mask |= 1 << int(vertex)
        masks.append(mask)
    vertices = list(range(n))
    candidates = _dominance_reduced(vertices, masks) if dominance else vertices
    selected, _ = _solve_model(
        vertices,
        candidates,
        masks,
        use_hint=use_hint,
        use_upper_bound=use_hint,
    )
    if selected is not None:
        return selected
    if risk_fallback and set(candidates) != set(vertices):
        selected, _ = _solve_model(vertices, vertices, masks, use_hint=use_hint, use_upper_bound=use_hint)
        if selected is not None:
            return selected
    raise RuntimeError("NumPy-backed exact model did not return OPTIMAL")


def _reference_style(problem: Problem, *, use_hint: bool = False) -> Solution:
    matrix = _normalise(problem)
    n = len(matrix)
    model = cp_model.CpModel()
    nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    for i in range(n):
        covering = [nodes[i]] + [nodes[j] for j in range(n) if matrix[i][j]]
        model.Add(sum(covering) >= 1)
    if use_hint:
        greedy = _greedy_cover(list(range(n)), list(range(n)), _closed_masks(matrix))
        hinted = set(greedy)
        for i, variable in enumerate(nodes):
            model.AddHint(variable, 1 if i in hinted else 0)
        model.Add(sum(nodes) <= len(greedy))
    model.Minimize(sum(nodes))
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError("reference-style CP-SAT model did not return OPTIMAL")
    return [i for i, variable in enumerate(nodes) if solver.Value(variable) == 1]


# v4_full: exact mappings from the six frozen proposal compositions.
def v4_bit_sparse_risk(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=True, dominance=True, early_certificate=False, use_hint=True, use_upper_bound=True, risk_fallback=True)


def v4_bit_closed_risk(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=False, dominance=True, early_certificate=True, use_hint=True, use_upper_bound=True, risk_fallback=True)


def v4_bit_sparse_closed(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=True, dominance=True, early_certificate=True, use_hint=True, use_upper_bound=True, risk_fallback=False)


def v4_bit_risk_early(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=False, dominance=False, early_certificate=True, use_hint=True, use_upper_bound=True, risk_fallback=True)


def v4_bit_sparse_early(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=True, dominance=False, early_certificate=True, use_hint=True, use_upper_bound=False, risk_fallback=False)


def v4_bit_closed_early(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=False, dominance=True, early_certificate=True, use_hint=True, use_upper_bound=False, risk_fallback=False)


# The no-transfer ablation produced the same mechanisms on Task 2 and intentionally aliases them.
def no_transfer_bit_sparse_risk(problem: Problem) -> Solution:
    return v4_bit_sparse_risk(problem)


def no_transfer_bit_closed_risk(problem: Problem) -> Solution:
    return v4_bit_closed_risk(problem)


def no_transfer_bit_sparse_closed(problem: Problem) -> Solution:
    return v4_bit_sparse_closed(problem)


def no_transfer_bit_risk_early(problem: Problem) -> Solution:
    return v4_bit_risk_early(problem)


def no_transfer_bit_sparse_early(problem: Problem) -> Solution:
    return v4_bit_sparse_early(problem)


def no_transfer_bit_closed_early(problem: Problem) -> Solution:
    return v4_bit_closed_early(problem)


# Frozen random-search compositions.
def random_sparse_risk_early(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=True, dominance=False, early_certificate=True, use_hint=True, use_upper_bound=True, risk_fallback=True)


def random_vector_closed_risk(problem: Problem) -> Solution:
    return _solve_numpy(problem, dtype=np.int64, contiguous=True, dominance=True, use_hint=True, risk_fallback=True)


def random_zero_sparse_closed(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=True, dominance=True, early_certificate=True, use_hint=False, use_upper_bound=False, risk_fallback=False)


def random_contiguous_dtype_risk(problem: Problem) -> Solution:
    return _solve_numpy(problem, dtype=np.uint8, contiguous=True, dominance=False, use_hint=True, risk_fallback=True)


def random_bit_risk_early(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=False, dominance=False, early_certificate=True, use_hint=True, use_upper_bound=True, risk_fallback=True)


def random_dtype_risk(problem: Problem) -> Solution:
    return _solve_numpy(problem, dtype=np.uint8, contiguous=False, dominance=False, use_hint=True, risk_fallback=True)


# Single-operator template arm, completed by unchanged exact CP-SAT semantics.
def template_bit_parallel(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=False, dominance=False, early_certificate=False, use_hint=False, use_upper_bound=False, risk_fallback=False)


def template_risk_stage(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=False, dominance=True, early_certificate=False, use_hint=True, use_upper_bound=True, risk_fallback=True)


def template_sparse_frontier(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=True, dominance=False, early_certificate=False, use_hint=False, use_upper_bound=False, risk_fallback=False)


def template_closed_form(problem: Problem) -> Solution:
    return _solve_bitset(problem, decompose=False, dominance=True, early_certificate=True, use_hint=False, use_upper_bound=False, risk_fallback=False)


def template_early_certificate(problem: Problem) -> Solution:
    matrix = _normalise(problem)
    masks = _closed_masks(matrix)
    vertices = list(range(len(matrix)))
    if not vertices:
        return []
    greedy = _greedy_cover(vertices, vertices, masks)
    if len(greedy) == _lower_bound(vertices, vertices, masks):
        return greedy
    return _reference_style(matrix)


def template_vectorized_batch(problem: Problem) -> Solution:
    return _solve_numpy(problem, dtype=np.int64, contiguous=True, dominance=False, use_hint=False, risk_fallback=False)


# Reproducible v3-compatible arm.
def v3_vectorized_batch(problem: Problem) -> Solution:
    return _solve_numpy(problem, dtype=np.int64, contiguous=True, dominance=False, use_hint=False, risk_fallback=False)


def v3_zero_copy(problem: Problem) -> Solution:
    return _reference_style(problem, use_hint=False)


def v3_dtype_specialization(problem: Problem) -> Solution:
    return _solve_numpy(problem, dtype=np.uint8, contiguous=False, dominance=False, use_hint=False, risk_fallback=False)


def v3_contiguous_layout(problem: Problem) -> Solution:
    return _solve_numpy(problem, dtype=np.int64, contiguous=True, dominance=False, use_hint=False, risk_fallback=False)


CANDIDATES_BY_ARM: dict[str, dict[str, Callable[[Problem], Solution]]] = {
    "v4_full": {
        "v4_bit_sparse_risk": v4_bit_sparse_risk,
        "v4_bit_closed_risk": v4_bit_closed_risk,
        "v4_bit_sparse_closed": v4_bit_sparse_closed,
        "v4_bit_risk_early": v4_bit_risk_early,
        "v4_bit_sparse_early": v4_bit_sparse_early,
        "v4_bit_closed_early": v4_bit_closed_early,
    },
    "v4_no_transfer": {
        "no_transfer_bit_sparse_risk": no_transfer_bit_sparse_risk,
        "no_transfer_bit_closed_risk": no_transfer_bit_closed_risk,
        "no_transfer_bit_sparse_closed": no_transfer_bit_sparse_closed,
        "no_transfer_bit_risk_early": no_transfer_bit_risk_early,
        "no_transfer_bit_sparse_early": no_transfer_bit_sparse_early,
        "no_transfer_bit_closed_early": no_transfer_bit_closed_early,
    },
    "random_search": {
        "random_sparse_risk_early": random_sparse_risk_early,
        "random_vector_closed_risk": random_vector_closed_risk,
        "random_zero_sparse_closed": random_zero_sparse_closed,
        "random_contiguous_dtype_risk": random_contiguous_dtype_risk,
        "random_bit_risk_early": random_bit_risk_early,
        "random_dtype_risk": random_dtype_risk,
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
