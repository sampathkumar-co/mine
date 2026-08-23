from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
from pathlib import Path

from candidates import CANDIDATES_BY_ARM, Problem, Solution


def matrix(n: int, edges: list[tuple[int, int]]) -> Problem:
    result = [[0 for _ in range(n)] for _ in range(n)]
    for u, v in edges:
        result[u][v] = 1
        result[v][u] = 1
    return result


def dominates(problem: Problem, solution: Solution) -> bool:
    n = len(problem)
    selected = set(solution)
    if len(selected) != len(solution) or any(vertex < 0 or vertex >= n for vertex in selected):
        return False
    for vertex in range(n):
        if vertex in selected:
            continue
        if not any(problem[vertex][chosen] for chosen in selected):
            return False
    return True


def brute_optimum(problem: Problem) -> tuple[int, list[int]]:
    n = len(problem)
    for size in range(n + 1):
        for combination in itertools.combinations(range(n), size):
            candidate = list(combination)
            if dominates(problem, candidate):
                return size, candidate
    raise RuntimeError("no dominating set found")


def random_graph(n: int, probability: float, seed: int) -> Problem:
    rng = random.Random(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < probability]
    return matrix(n, edges)


def cases() -> list[tuple[str, Problem]]:
    disconnected_edges = [(0, 1), (1, 2), (3, 4), (4, 5), (5, 3), (7, 8)]
    return [
        ("empty", []),
        ("isolated_5", matrix(5, [])),
        ("single_edge", matrix(2, [(0, 1)])),
        ("path_7", matrix(7, [(i, i + 1) for i in range(6)])),
        ("cycle_8", matrix(8, [(i, (i + 1) % 8) for i in range(8)])),
        ("star_10", matrix(10, [(0, i) for i in range(1, 10)])),
        ("disconnected_10", matrix(10, disconnected_edges)),
        ("random_sparse_11", random_graph(11, 0.15, 5201)),
        ("random_dense_12", random_graph(12, 0.35, 5202)),
        ("twins_and_dominance", matrix(9, [(0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)])),
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    oracle = {label: brute_optimum(problem) for label, problem in cases()}
    reports: list[dict[str, object]] = []
    failures: list[str] = []
    for arm, candidates in CANDIDATES_BY_ARM.items():
        for name, candidate in candidates.items():
            valid_count = 0
            exceptions = 0
            case_reports: list[dict[str, object]] = []
            for label, problem in cases():
                optimum_size, witness = oracle[label]
                try:
                    start = time.perf_counter()
                    solution = candidate(problem)
                    elapsed = time.perf_counter() - start
                    valid = dominates(problem, solution) and len(solution) == optimum_size
                    reason = None if valid else f"size_or_domination: returned={solution}, optimum_size={optimum_size}, witness={witness}"
                except Exception as exc:
                    solution = []
                    elapsed = None
                    valid = False
                    reason = f"{type(exc).__name__}: {exc}"
                    exceptions += 1
                valid_count += int(valid)
                case_reports.append({
                    "case": label,
                    "valid": valid,
                    "returned": solution,
                    "returned_size": len(solution),
                    "optimum_size": optimum_size,
                    "elapsed_s": elapsed,
                    "failure_reason": reason,
                })
            if valid_count != len(oracle) or exceptions:
                failures.append(f"{arm}/{name}: valid={valid_count}/{len(oracle)}, exceptions={exceptions}")
            reports.append({
                "arm": arm,
                "candidate": name,
                "valid": valid_count,
                "count": len(oracle),
                "exceptions": exceptions,
                "cases": case_reports,
            })

    report = {
        "task": "min_dominating_set",
        "revision": 1,
        "candidate_source_sha256": sha256(Path(__file__).resolve().parent / "candidates.py"),
        "requirements_sha256": sha256(Path(__file__).resolve().parent / "requirements.txt"),
        "oracle": "independent exhaustive subset enumeration",
        "case_count": len(oracle),
        "candidate_count": len(reports),
        "candidate_reports": reports,
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
