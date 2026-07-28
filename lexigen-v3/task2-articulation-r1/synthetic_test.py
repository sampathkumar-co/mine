from __future__ import annotations

import random

import networkx as nx

from candidates import CANDIDATES, Problem


def reference(problem: Problem) -> list[int]:
    graph = nx.Graph()
    graph.add_nodes_from(range(problem["num_nodes"]))
    graph.add_edges_from(problem["edges"])
    return sorted(nx.articulation_points(graph))


problems: list[Problem] = [
    {"num_nodes": 2, "edges": []},
    {"num_nodes": 2, "edges": [[0, 1]]},
    {"num_nodes": 7, "edges": []},
    {"num_nodes": 7, "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]},
    {"num_nodes": 7, "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]]},
    {"num_nodes": 8, "edges": [[0, node] for node in range(1, 8)]},
    {"num_nodes": 8, "edges": [[u, v] for u in range(8) for v in range(u + 1, 8)]},
    {
        "num_nodes": 10,
        "edges": [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 3], [6, 7]],
    },
]

for seed in range(20):
    rng = random.Random(seed)
    node_count = 2 + seed
    probability = (seed % 7) / 10
    edges = [
        [u, v]
        for u in range(node_count)
        for v in range(u + 1, node_count)
        if rng.random() < probability
    ]
    problems.append({"num_nodes": node_count, "edges": edges})

for problem_index, problem in enumerate(problems):
    expected = reference(problem)
    for name, candidate in CANDIDATES.items():
        result = candidate(problem)
        points = result["articulation_points"]
        assert isinstance(points, list), (problem_index, name, type(points))
        assert all(isinstance(node, int) for node in points), (problem_index, name)
        assert points == expected, (problem_index, name, expected, points)

print("synthetic articulation-point exactness passed")
