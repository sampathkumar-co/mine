from __future__ import annotations

import json
import random
from pathlib import Path

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

failures: list[dict[str, object]] = []
for problem_index, problem in enumerate(problems):
    expected = reference(problem)
    for name, candidate in CANDIDATES.items():
        try:
            result = candidate(problem)
            points = result["articulation_points"]
            if not isinstance(points, list):
                failures.append({
                    "problem_index": problem_index,
                    "candidate": name,
                    "kind": "wrong_type",
                    "actual_type": type(points).__name__,
                })
            elif not all(isinstance(node, int) for node in points):
                failures.append({
                    "problem_index": problem_index,
                    "candidate": name,
                    "kind": "non_integer_node",
                    "actual": repr(points),
                })
            elif points != expected:
                failures.append({
                    "problem_index": problem_index,
                    "candidate": name,
                    "kind": "mismatch",
                    "expected": expected,
                    "actual": points,
                    "num_nodes": problem["num_nodes"],
                    "edges": problem["edges"],
                })
        except Exception as exc:
            failures.append({
                "problem_index": problem_index,
                "candidate": name,
                "kind": "exception",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "num_nodes": problem["num_nodes"],
                "edges": problem["edges"],
            })

Path("synthetic-diagnostics.json").write_text(
    json.dumps({"failures": failures}, indent=2),
    encoding="utf-8",
)
if failures:
    print(json.dumps(failures[:5], indent=2), flush=True)
    raise RuntimeError(f"synthetic exactness found {len(failures)} failures")

print("synthetic articulation-point exactness passed")
