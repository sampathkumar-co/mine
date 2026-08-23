from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from candidates import CANDIDATES_BY_ARM, Problem, Solution


def matrix(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    out = [[0] * n for _ in range(n)]
    for u, v in edges:
        out[u][v] = 1
        out[v][u] = 1
    return out


def valid_mapping(problem: Problem, solution: Solution) -> tuple[bool, str | None]:
    if not isinstance(solution, list):
        return False, "solution_not_list"
    A = problem["A"]
    B = problem["B"]
    n, m = len(A), len(B)
    left: set[int] = set()
    right: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for raw in solution:
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            return False, "pair_shape"
        i, p = int(raw[0]), int(raw[1])
        if not (0 <= i < n and 0 <= p < m):
            return False, "pair_bounds"
        if i in left or p in right:
            return False, "not_one_to_one"
        left.add(i)
        right.add(p)
        pairs.append((i, p))
    for a in range(len(pairs)):
        i, p = pairs[a]
        for b in range(a + 1, len(pairs)):
            j, q = pairs[b]
            if int(A[i][j]) != int(B[p][q]):
                return False, "induced_edge_mismatch"
    return True, None


def oracle_optimum(problem: Problem) -> int:
    A = problem["A"]
    B = problem["B"]
    n, m = len(A), len(B)
    limit = min(n, m)
    for k in range(limit, -1, -1):
        for left in itertools.combinations(range(n), k):
            for right in itertools.permutations(range(m), k):
                ok = True
                for a in range(k):
                    i, p = left[a], right[a]
                    for b in range(a + 1, k):
                        j, q = left[b], right[b]
                        if int(A[i][j]) != int(B[p][q]):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return k
    raise RuntimeError("oracle failed to find even the empty mapping")


def cases() -> list[tuple[str, Problem]]:
    path4 = matrix(4, [(0, 1), (1, 2), (2, 3)])
    star4 = matrix(4, [(0, 1), (0, 2), (0, 3)])
    cycle4 = matrix(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    matching4 = matrix(4, [(0, 1), (2, 3)])
    triangle3 = matrix(3, [(0, 1), (1, 2), (0, 2)])
    path3 = matrix(3, [(0, 1), (1, 2)])
    g5a = matrix(5, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    g5b = matrix(5, [(0, 1), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)])
    g5c = matrix(5, [(0, 2), (0, 4), (1, 2), (1, 3), (2, 4), (3, 4)])
    return [
        ("empty", {"A": [], "B": []}),
        ("single", {"A": matrix(1, []), "B": matrix(1, [])}),
        ("edge_vs_nonedge", {"A": matrix(2, [(0, 1)]), "B": matrix(2, [])}),
        ("path3_vs_triangle3", {"A": path3, "B": triangle3}),
        ("cycle4_identity", {"A": cycle4, "B": cycle4}),
        ("path4_vs_star4", {"A": path4, "B": star4}),
        ("matching_vs_cycle", {"A": matching4, "B": cycle4}),
        ("five_a_vs_b", {"A": g5a, "B": g5b}),
        ("five_b_vs_c", {"A": g5b, "B": g5c}),
        ("different_sizes", {"A": g5a, "B": path4}),
    ]


def main() -> None:
    suite = cases()
    expected = {name: oracle_optimum(problem) for name, problem in suite}
    output = Path("synthetic-evidence")
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    failures = 0
    candidate_count = 0

    for arm, candidates in CANDIDATES_BY_ARM.items():
        for candidate_name, candidate in candidates.items():
            candidate_count += 1
            valid_cases = 0
            for case_name, problem in suite:
                start = time.perf_counter()
                exception: str | None = None
                proposed: Solution | None = None
                try:
                    proposed = candidate(problem)
                except Exception as exc:
                    exception = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - start
                structural_ok = False
                structural_reason: str | None = None
                size_ok = False
                if exception is None and proposed is not None:
                    structural_ok, structural_reason = valid_mapping(problem, proposed)
                    size_ok = structural_ok and len(proposed) == expected[case_name]
                valid = exception is None and structural_ok and size_ok
                if valid:
                    valid_cases += 1
                else:
                    failures += 1
                row = {
                    "arm": arm,
                    "candidate": candidate_name,
                    "case": case_name,
                    "oracle_optimum": expected[case_name],
                    "solution_size": len(proposed) if isinstance(proposed, list) else None,
                    "valid": valid,
                    "exception": exception,
                    "structural_reason": structural_reason,
                    "elapsed_s": elapsed,
                }
                results.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            print(f"SUMMARY {arm}/{candidate_name}: {valid_cases}/{len(suite)}", flush=True)

    summary = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 3,
        "task": "max_common_subgraph",
        "stage": "synthetic_revision1",
        "candidate_count": candidate_count,
        "synthetic_case_count": len(suite),
        "candidate_case_checks": len(results),
        "passing_checks": len(results) - failures,
        "failing_checks": failures,
        "all_candidates_exact": failures == 0,
        "official_training_manifest_accessed": False,
        "official_test_manifest_accessed": False,
        "training_revision_consumed": False,
        "oracle": "independent exhaustive subset/injection enumeration; no OR-Tools and no candidate helpers",
        "oracle_optima": expected,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "results.jsonl").write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in results) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(f"synthetic correctness gate failed: {failures} candidate-case failures")


if __name__ == "__main__":
    main()
