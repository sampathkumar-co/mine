from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from ortools.sat.python import cp_model

# Apply only the preregistered generic empty-input correction, then use the
# unchanged frozen candidate mapping.
from candidates_r1b import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "min_dominating_set"
MANIFEST = "min_dominating_set_T100ms_n9_size100_train.jsonl"
EXPECTED_SHA256 = "203e66d13ecbf3f50789df42a47857dfe58a1d79135f87c2be6979f90283fbda"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task2-train-r1b"})
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"training fetch exhausted retries: {url}") from last


def normalise(problem: object) -> Problem:
    if not isinstance(problem, list):
        raise ValueError("problem must be an adjacency-matrix list")
    n = len(problem)
    matrix: Problem = []
    for row in problem:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("adjacency matrix must be square")
        matrix.append([1 if int(value) else 0 for value in row])
    for i in range(n):
        matrix[i][i] = 0
        for j in range(i + 1, n):
            if matrix[i][j] != matrix[j][i]:
                raise ValueError("adjacency matrix must be symmetric")
    return matrix


def reference(problem: Problem) -> Solution:
    n = len(problem)
    if n == 0:
        return []
    model = cp_model.CpModel()
    nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    for i in range(n):
        covering = [nodes[i]] + [nodes[j] for j in range(n) if problem[i][j]]
        model.Add(sum(covering) >= 1)
    model.Minimize(sum(nodes))
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"reference CP-SAT status {int(status)} is not OPTIMAL")
    return [i for i, variable in enumerate(nodes) if solver.Value(variable) == 1]


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        elapsed = time.perf_counter() - start
        return solution, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def dominates(problem: Problem, solution: Solution) -> bool:
    n = len(problem)
    if not isinstance(solution, list):
        return False
    selected: set[int] = set()
    for raw in solution:
        if isinstance(raw, bool):
            return False
        try:
            vertex = int(raw)
        except Exception:
            return False
        if vertex != raw or vertex < 0 or vertex >= n or vertex in selected:
            return False
        selected.add(vertex)
    for i in range(n):
        if i in selected:
            continue
        if not any(problem[i][chosen] for chosen in selected):
            return False
    return True


def validate(problem: Problem, proposed: Solution | None, expected: Solution) -> tuple[bool, str | None]:
    if proposed is None:
        return False, "candidate_exception"
    if not dominates(problem, proposed):
        return False, "not_dominating_or_format"
    if len(proposed) != len(expected):
        return False, f"nonoptimal_size_{len(proposed)}_vs_{len(expected)}"
    return True, None


def graph_stats(problem: Problem) -> tuple[int, int, int]:
    n = len(problem)
    edges = sum(sum(row) for row in problem) // 2
    unseen = set(range(n))
    components = 0
    while unseen:
        components += 1
        start = unseen.pop()
        stack = [start]
        while stack:
            u = stack.pop()
            for v, edge in enumerate(problem[u]):
                if edge and v in unseen:
                    unseen.remove(v)
                    stack.append(v)
    return n, edges, components


def flattened_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    return [
        (arm, name, candidate)
        for arm, candidates in CANDIDATES_BY_ARM.items()
        for name, candidate in candidates.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("training manifest SHA-256 mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, received {len(rows)}")

    candidates = flattened_candidates()
    selected_rows = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    evidence: list[dict[str, object]] = []
    for index, row in selected_rows:
        problem = normalise(row.get("problem"))
        n, edge_count, components = graph_stats(problem)

        shift = index % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]
        if index % 2 == 0:
            expected, reference_s, reference_error = timed(reference, problem)
            candidate_results = [(arm, name, *timed(candidate, problem)) for arm, name, candidate in ordered]
            execution_order = "reference_first"
        else:
            candidate_results = [(arm, name, *timed(candidate, problem)) for arm, name, candidate in ordered]
            expected, reference_s, reference_error = timed(reference, problem)
            execution_order = "candidates_first"
        if expected is None or reference_s is None or reference_error is not None:
            raise RuntimeError(f"reference failed on record {index + 1}: {reference_error}")

        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            valid, validation_error = validate(problem, proposed, expected)
            speedup = reference_s / candidate_s if candidate_s and candidate_s > 0.0 else 0.0
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": valid,
                "failure_reason": candidate_error or validation_error,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "solution_size": len(proposed) if isinstance(proposed, list) else None,
                "reference_size": len(expected),
                "nodes": n,
                "edges": edge_count,
                "components": components,
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0
            })
            print(f"[{index + 1}/100] {arm}/{name} valid={valid} speedup={speedup:.3f} size={len(proposed) if isinstance(proposed, list) else 'NA'}/{len(expected)}", flush=True)
        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in evidence) + "\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
