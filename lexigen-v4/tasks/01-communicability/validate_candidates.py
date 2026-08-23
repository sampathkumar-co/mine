from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import networkx as nx
import numpy as np
from threadpoolctl import threadpool_limits

from candidates import CANDIDATES_BY_ARM, Problem, Solution

RTOL = 1e-5
ATOL = 1e-8
APPROXIMATE_RISK = {"random_float32_sparse", "v3_dtype_specialization"}


def problem_from_graph(graph: nx.Graph) -> Problem:
    n = graph.number_of_nodes()
    return {"adjacency_list": [sorted(int(v) for v in graph.neighbors(u)) for u in range(n)]}


def reference(problem: Problem) -> Solution:
    adjacency_list = problem["adjacency_list"]
    assert isinstance(adjacency_list, list)
    n = len(adjacency_list)
    if n == 0:
        return {"communicability": {}}
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for u, neighbors in enumerate(adjacency_list):
        assert isinstance(neighbors, list)
        for v in neighbors:
            if u < int(v):
                graph.add_edge(u, int(v))
    raw = nx.communicability(graph)
    return {"communicability": {u: {v: float(raw[u][v]) for v in range(n)} for u in range(n)}}


def validate(problem: Problem, proposed: Solution, expected: Solution) -> tuple[bool, float, str | None]:
    adjacency_list = problem["adjacency_list"]
    assert isinstance(adjacency_list, list)
    n = len(adjacency_list)
    value = proposed.get("communicability") if isinstance(proposed, dict) else None
    if not isinstance(value, dict) or set(value) != set(range(n)):
        return False, float("inf"), "outer_keys"
    expected_value = expected["communicability"]
    maximum_error = 0.0
    for u in range(n):
        row = value.get(u)
        if not isinstance(row, dict) or set(row) != set(range(n)):
            return False, float("inf"), f"inner_keys_{u}"
        for v in range(n):
            try:
                actual = float(row[v])
                target = float(expected_value[u][v])
            except Exception as exc:
                return False, float("inf"), f"value_{u}_{v}_{type(exc).__name__}"
            if not math.isfinite(actual):
                return False, float("inf"), f"nonfinite_{u}_{v}"
            maximum_error = max(maximum_error, abs(actual - target))
            if not math.isclose(actual, target, rel_tol=RTOL, abs_tol=ATOL):
                return False, maximum_error, f"mismatch_{u}_{v}"
    return True, maximum_error, None


def cases() -> list[tuple[str, Problem]]:
    empty = nx.Graph()
    isolated = nx.empty_graph(7)
    edge = nx.Graph([(0, 1)])
    disconnected = nx.Graph()
    disconnected.add_nodes_from(range(9))
    disconnected.add_edges_from([(0, 1), (2, 3), (3, 4), (4, 2), (6, 7)])
    path = nx.path_graph(18)
    cycle = nx.cycle_graph(31)
    erdos_32 = nx.erdos_renyi_graph(32, 8 / 31, seed=4101)
    erdos_64 = nx.erdos_renyi_graph(64, 8 / 63, seed=4102)
    return [
        ("empty", problem_from_graph(empty)),
        ("isolated", problem_from_graph(isolated)),
        ("edge", problem_from_graph(edge)),
        ("disconnected", problem_from_graph(disconnected)),
        ("path", problem_from_graph(path)),
        ("cycle", problem_from_graph(cycle)),
        ("erdos_32", problem_from_graph(erdos_32)),
        ("erdos_64", problem_from_graph(erdos_64)),
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    all_cases = cases()
    references: dict[str, tuple[Problem, Solution]] = {}
    with threadpool_limits(limits=1):
        for label, problem in all_cases:
            references[label] = (problem, reference(problem))

    candidate_reports: list[dict[str, object]] = []
    failures: list[str] = []
    for arm, candidates in CANDIDATES_BY_ARM.items():
        for candidate_name, candidate in candidates.items():
            case_reports: list[dict[str, object]] = []
            exceptions = 0
            valid_count = 0
            for label, (problem, expected) in references.items():
                try:
                    with threadpool_limits(limits=1):
                        start = time.perf_counter()
                        proposed = candidate(problem)
                        elapsed = time.perf_counter() - start
                    valid, maximum_error, reason = validate(problem, proposed, expected)
                except Exception as exc:
                    elapsed = None
                    valid = False
                    maximum_error = float("inf")
                    reason = f"{type(exc).__name__}: {exc}"
                    exceptions += 1
                valid_count += int(valid)
                case_reports.append({
                    "case": label,
                    "valid": valid,
                    "maximum_absolute_error": maximum_error,
                    "elapsed_s": elapsed,
                    "failure_reason": reason,
                })
            expected_exact = candidate_name not in APPROXIMATE_RISK
            if exceptions:
                failures.append(f"{arm}/{candidate_name} raised {exceptions} exceptions")
            if expected_exact and valid_count != len(all_cases):
                failures.append(f"{arm}/{candidate_name} exact synthetic validity {valid_count}/{len(all_cases)}")
            candidate_reports.append({
                "arm": arm,
                "candidate": candidate_name,
                "expected_exact": expected_exact,
                "valid": valid_count,
                "count": len(all_cases),
                "exceptions": exceptions,
                "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in case_reports),
                "cases": case_reports,
            })

    report = {
        "task": "communicability",
        "revision": 1,
        "candidate_source_sha256": sha256(Path(__file__).resolve().parent / "candidates.py"),
        "requirements_sha256": sha256(Path(__file__).resolve().parent / "requirements.txt"),
        "case_count": len(all_cases),
        "candidate_count": len(candidate_reports),
        "candidate_reports": candidate_reports,
        "approximate_risk_candidates": sorted(APPROXIMATE_RISK),
        "infrastructure_gate_passed": not failures,
        "failures": failures,
        "official_task_data_accessed": False,
        "training_revision_consumed": False,
    }
    output = Path("synthetic-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "synthetic-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "case_count": report["case_count"],
        "infrastructure_gate_passed": report["infrastructure_gate_passed"],
        "failures": failures,
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
