from __future__ import annotations

import json

from engine import generate_proposals, verify_transfer_memory


def ids(result: dict[str, object], arm: str) -> set[str]:
    proposals = result["arms"][arm]
    return {tid for proposal in proposals for tid in proposal["transfer_ids"]}


def operator_sets(result: dict[str, object], arm: str) -> set[tuple[str, ...]]:
    return {tuple(proposal["operators"]) for proposal in result["arms"][arm]}


def main() -> None:
    memory = verify_transfer_memory()

    discrete_source = """
def solve(problem):
    graph = problem['graph']
    active = set(graph)
    while active:
        node = active.pop()
        mask = graph[node]
    return list(active)
def is_solution(problem, solution):
    return isinstance(solution, list)
"""
    discrete = generate_proposals(discrete_source, discrete_source)
    assert "TM-BFR-01" in ids(discrete, "v5_full")
    assert "TM-BFR-01" not in ids(discrete, "v5_no_transfer")
    assert ("bit_parallel_representation", "sparse_frontier_search", "early_certificate_exit") not in operator_sets(discrete, "v5_no_transfer")

    projection_source = """
import numpy as np
def solve(problem):
    x = np.asarray(problem['x'], dtype=float)
    threshold = 0.01
    while np.any(x < threshold):
        x = np.maximum(x, threshold)
    return x
def is_solution(problem, solution):
    return np.allclose(np.sum(solution), 1.0, rtol=1e-5, atol=1e-8)
"""
    projection = generate_proposals(projection_source, projection_source)
    assert "TM-CAC-01" in ids(projection, "v5_full")
    assert "TM-CAC-01" not in ids(projection, "v5_no_transfer")

    matrix_source = """
import numpy as np
def solve(problem):
    a = np.asarray(problem['matrix'], dtype=float)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    return (u * s) @ vh
def is_solution(problem, solution):
    return np.allclose(solution, problem['matrix'], rtol=1e-5, atol=1e-8)
"""
    matrix = generate_proposals(matrix_source, matrix_source)
    assert "TM-RRR-01" in ids(matrix, "v5_full")
    assert "TM-PBEB-01" in ids(matrix, "v5_full")

    dynamics_source = """
import numpy as np
from scipy.integrate import solve_ivp
def solve(problem):
    def derivative(t, y): return y
    return solve_ivp(derivative, [0.0, 1000.0], np.asarray(problem['y']), rtol=1e-8, atol=1e-8).y[:, -1]
def is_solution(problem, solution):
    return np.allclose(solution, solution, rtol=1e-5, atol=1e-8)
"""
    dynamics = generate_proposals(dynamics_source, dynamics_source)
    assert "TM-PBEB-01" not in ids(dynamics, "v5_full"), "long-horizon negative lesson must suppress precision/backend transfer"

    again = generate_proposals(discrete_source, discrete_source)
    assert json.dumps(discrete, sort_keys=True) == json.dumps(again, sort_keys=True), "engine must be deterministic"

    print(json.dumps({
        "status": "passed",
        "transfer_memory": memory,
        "discrete_transfer_ids": sorted(ids(discrete, "v5_full")),
        "projection_transfer_ids": sorted(ids(projection, "v5_full")),
        "matrix_transfer_ids": sorted(ids(matrix, "v5_full")),
        "dynamics_transfer_ids": sorted(ids(dynamics, "v5_full")),
        "no_transfer_contains_learned_ids": False,
        "deterministic": True
    }, indent=2))


if __name__ == "__main__":
    main()
