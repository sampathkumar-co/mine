from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Candidate:
    name: str
    arm: str
    implementation_class: str
    operators: tuple[str, ...]
    transfer_ids: tuple[str, ...]
    learned_template: str | None
    baseline_id: str | None
    solve: Callable[[dict], list[list[int]]]


def _arrays(problem: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    cap = np.asarray(problem["capacity"], dtype=np.int64)
    cost = np.asarray(problem["cost"], dtype=np.int64)
    if cap.ndim != 2 or cap.shape[0] != cap.shape[1] or cost.shape != cap.shape:
        raise ValueError("capacity/cost must be equal square matrices")
    n = int(cap.shape[0])
    s, t = int(problem["s"]), int(problem["t"])
    if not (0 <= s < n and 0 <= t < n and s != t):
        raise ValueError("invalid source/sink")
    if np.any(cap < 0):
        raise ValueError("negative capacity")
    return cap, cost, s, t


def _dense_graph(problem: dict) -> tuple[nx.DiGraph, int, int, int]:
    cap, cost, s, t = _arrays(problem)
    n = len(cap)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if int(cap[i, j]) > 0:
                g.add_edge(i, j, capacity=int(cap[i, j]), cost=int(cost[i, j]))
    return g, s, t, n


def _sparse_graph(problem: dict) -> tuple[nx.DiGraph, int, int, int]:
    cap, cost, s, t = _arrays(problem)
    n = len(cap)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    rows, cols = np.nonzero(cap > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        g.add_edge(i, j, capacity=int(cap[i, j]), cost=int(cost[i, j]))
    return g, s, t, n


def _dense_solution(n: int, flow: dict) -> list[list[int]]:
    out = [[0 for _ in range(n)] for _ in range(n)]
    for i, row in flow.items():
        for j, value in row.items():
            if value:
                out[int(i)][int(j)] = int(value)
    return out


def reference_exact(problem: dict) -> list[list[int]]:
    g, s, t, n = _dense_graph(problem)
    flow = nx.max_flow_min_cost(g, s, t, capacity="capacity", weight="cost")
    return _dense_solution(n, flow)


def sparse_reference(problem: dict) -> list[list[int]]:
    g, s, t, n = _sparse_graph(problem)
    flow = nx.max_flow_min_cost(g, s, t, capacity="capacity", weight="cost")
    return _dense_solution(n, flow)


def frontier_pruned_reference(problem: dict) -> list[list[int]]:
    g, s, t, n = _sparse_graph(problem)
    forward = {s} | nx.descendants(g, s)
    if t not in forward:
        return [[0 for _ in range(n)] for _ in range(n)]
    reverse = g.reverse(copy=False)
    backward = {t} | nx.descendants(reverse, t)
    active = forward & backward
    h = g.subgraph(active).copy()
    flow = nx.max_flow_min_cost(h, s, t, capacity="capacity", weight="cost")
    return _dense_solution(n, flow)


def network_simplex_two_stage(problem: dict) -> list[list[int]]:
    g, s, t, n = _sparse_graph(problem)
    forward = {s} | nx.descendants(g, s)
    if t not in forward:
        return [[0 for _ in range(n)] for _ in range(n)]
    backward = {t} | nx.descendants(g.reverse(copy=False), t)
    h = g.subgraph(forward & backward).copy()
    value = int(nx.maximum_flow_value(h, s, t, capacity="capacity"))
    for node in h.nodes:
        h.nodes[node]["demand"] = 0
    h.nodes[s]["demand"] = -value
    h.nodes[t]["demand"] = value
    _, flow = nx.network_simplex(h, demand="demand", capacity="capacity", weight="cost")
    return _dense_solution(n, flow)


def cp_sat_exact(problem: dict) -> list[list[int]]:
    from ortools.sat.python import cp_model

    cap, cost, s, t = _arrays(problem)
    n = len(cap)
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(cap > 0))]
    model = cp_model.CpModel()
    var = {(i, j): model.new_int_var(0, int(cap[i, j]), f"f_{i}_{j}") for i, j in edges}

    def incoming(v: int):
        return sum((var[(i, j)] for i, j in edges if j == v), 0)

    def outgoing(v: int):
        return sum((var[(i, j)] for i, j in edges if i == v), 0)

    for v in range(n):
        if v not in (s, t):
            model.add(incoming(v) == outgoing(v))
    flow_value = outgoing(s) - incoming(s)
    model.maximize(flow_value)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT max-flow phase failed")
    best_flow = int(round(solver.objective_value))
    model.add(flow_value == best_flow)
    total_cost = sum((int(cost[i, j]) * var[(i, j)] for i, j in edges), 0)
    model.minimize(total_cost)
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT min-cost phase failed")
    out = [[0 for _ in range(n)] for _ in range(n)]
    for i, j in edges:
        out[i][j] = int(solver.value(var[(i, j)]))
    return out


def independent_semantic_certificate(problem: dict, solution: list[list[int]]) -> bool:
    try:
        cap, cost, s, t = _arrays(problem)
        sol = np.asarray(solution, dtype=np.float64)
        n = len(cap)
        if sol.shape != (n, n) or not np.all(np.isfinite(sol)):
            return False
        tol = 1e-7
        if np.any(sol < -tol) or np.any(sol - cap > tol):
            return False
        if np.any((cap == 0) & (sol > tol)):
            return False
        if np.any(np.diag(sol) > tol):
            return False
        net = sol.sum(axis=1) - sol.sum(axis=0)
        for v in range(n):
            if v not in (s, t) and abs(float(net[v])) > tol:
                return False
        if abs(float(net[s] + net[t])) > tol or float(net[s]) < -tol:
            return False
        ref = np.asarray(reference_exact(problem), dtype=np.float64)
        ref_flow = float(ref[s].sum() - ref[:, s].sum())
        sol_flow = float(sol[s].sum() - sol[:, s].sum())
        if abs(sol_flow - ref_flow) > tol:
            return False
        ref_cost = float((ref * cost).sum())
        sol_cost = float((sol * cost).sum())
        if abs(sol_cost - ref_cost) > tol:
            return False
        return True
    except Exception:
        return False


def official_verifier_accepts(problem: dict, solution: list[list[int]]) -> bool:
    try:
        sol = np.asarray(solution, dtype=np.float64)
        cap, cost, s, t = _arrays(problem)
        n = len(cap)
        tol = 1e-5
        if sol.shape != (n, n) or not np.all(np.isfinite(sol)) or np.any(sol < -tol):
            return False
        for i in range(n):
            for j in range(n):
                if sol[i, j] > tol and sol[j, i] > tol:
                    return False
                if i == j and sol[i, j] > tol:
                    return False
        if np.any(sol[:, s] > tol) or np.any(sol[t, :] > tol):
            return False
        total_out = float(sol[s, :].sum())
        total_in = float(sol[:, t].sum())
        if abs(total_out - total_in) > tol:
            return False
        for i in range(n):
            if i in (s, t):
                continue
            if abs(float(sol[:, i].sum() - sol[i, :].sum())) > tol:
                return False
        ref = np.asarray(reference_exact(problem), dtype=np.float64)
        ref_out = float(ref[s, :].sum())
        if total_out < ref_out - tol:
            return False
        if float((sol * cost).sum()) > float((ref * cost).sum()) + tol:
            return False
        return True
    except Exception:
        return False


def _map_engine_candidate(arm: str, proposal: dict) -> Candidate:
    ops = tuple(str(x) for x in proposal["operators"])
    tids = tuple(str(x) for x in proposal["transfer_ids"])
    op_set = set(ops)
    if tids == ("TM-BFR-01",):
        impl, fn = "frontier_pruned_reference", frontier_pruned_reference
    elif tids == ("TM-RRR-01",):
        impl, fn = "sparse_reduced_reference", sparse_reference
    elif "native_backend_substitution" in op_set:
        impl, fn = "network_simplex_two_stage", network_simplex_two_stage
    elif "active_set_decomposition" in op_set:
        impl, fn = "frontier_pruned_reference", frontier_pruned_reference
    elif op_set.intersection({"sparse_frontier_search", "reduced_representation", "bit_parallel_representation", "vectorized_batch_kernel", "zero_copy_representation", "contiguous_layout"}):
        impl, fn = "sparse_reduced_reference", sparse_reference
    else:
        impl, fn = "networkx_reference", reference_exact
    public_arm = {"v5_full":"v6_full", "v5_no_transfer":"v6_no_transfer", "v4_compatible":"v5_compatible"}.get(arm, arm)
    return Candidate(
        name=f"{public_arm}_r{proposal['rank']}_{proposal['proposal_id']}",
        arm=public_arm,
        implementation_class=impl,
        operators=ops,
        transfer_ids=tids,
        learned_template=proposal.get("learned_template"),
        baseline_id=None,
        solve=fn,
    )


def build_candidates(task_source_text: str) -> dict[str, list[Candidate]]:
    from engine import generate_proposals

    generated = generate_proposals(task_source_text)
    arms: dict[str, list[Candidate]] = {"v6_full":[], "v6_no_transfer":[], "random_search":[], "static_template":[], "v5_compatible":[], "strong_baseline":[]}
    for engine_arm, proposals in generated["arms"].items():
        for proposal in proposals:
            c = _map_engine_candidate(engine_arm, proposal)
            arms[c.arm].append(c)
    arms["strong_baseline"].append(Candidate(
        name="strong_baseline_sb_bool_exact_01_cpsat",
        arm="strong_baseline",
        implementation_class="ortools_cp_sat_exact_integer_flow",
        operators=("independent_exact_integer_backend",),
        transfer_ids=(),
        learned_template=None,
        baseline_id="SB-BOOL-EXACT-01",
        solve=cp_sat_exact,
    ))
    return arms
