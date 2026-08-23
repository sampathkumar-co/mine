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

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "max_common_subgraph"
MANIFEST = "max_common_subgraph_T100ms_n4_size100_train.jsonl"
EXPECTED_SHA256 = "534ff69ea6ae33d42734e5a90254fb59257c25991b70a01d488628d4e05b2ea9"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task3-train-r1"})
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


def reference(problem: Problem) -> Solution:
    A = problem["A"]
    B = problem["B"]
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return []
    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{i}_{p}") for p in range(m)] for i in range(n)]
    for i in range(n):
        model.Add(sum(x[i][p] for p in range(m)) <= 1)
    for p in range(m):
        model.Add(sum(x[i][p] for i in range(n)) <= 1)
    for i in range(n):
        for j in range(i + 1, n):
            for p in range(m):
                for q in range(m):
                    if p == q:
                        continue
                    if int(A[i][j]) != int(B[p][q]):
                        model.Add(x[i][p] + x[j][q] <= 1)
    model.Maximize(sum(x[i][p] for i in range(n) for p in range(m)))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"reference CP-SAT status {int(status)} is not OPTIMAL")
    return [(i, p) for i in range(n) for p in range(m) if solver.Value(x[i][p]) == 1]


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        elapsed = time.perf_counter() - start
        return solution, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def validate(problem: Problem, proposed: Solution | None, optimum_size: int) -> tuple[bool, str | None]:
    if not isinstance(proposed, list):
        return False, "solution_not_list"
    A = problem["A"]
    B = problem["B"]
    n, m = len(A), len(B)
    left: set[int] = set()
    right: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for raw in proposed:
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
    if len(pairs) != optimum_size:
        return False, "nonoptimal_size"
    return True, None


def stats(problem: Problem) -> tuple[int, int, int, int]:
    A, B = problem["A"], problem["B"]
    a_edges = sum(sum(int(v) for v in row) for row in A) // 2
    b_edges = sum(sum(int(v) for v in row) for row in B) // 2
    return len(A), len(B), a_edges, b_edges


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
        raw_problem = row.get("problem")
        if not isinstance(raw_problem, dict) or set(raw_problem) != {"A", "B"}:
            raise RuntimeError(f"record {index + 1} has unexpected problem representation")
        problem: Problem = {
            "A": [[int(v) for v in values] for values in raw_problem["A"]],
            "B": [[int(v) for v in values] for values in raw_problem["B"]],
        }
        n, m, a_edges, b_edges = stats(problem)
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
        optimum_size = len(expected)

        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            valid, validation_error = validate(problem, proposed, optimum_size)
            speedup = reference_s / candidate_s if candidate_s and candidate_s > 0.0 else 0.0
            record = {
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": valid and candidate_error is None,
                "failure_reason": candidate_error or validation_error,
                "solution_size": len(proposed) if isinstance(proposed, list) else None,
                "optimum_size": optimum_size,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "A_nodes": n,
                "B_nodes": m,
                "A_edges": a_edges,
                "B_edges": b_edges,
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
            }
            evidence.append(record)
            print(f"[{index + 1}/100] {arm}/{name} valid={record['valid']} size={record['solution_size']}/{optimum_size} speedup={speedup:.3f}", flush=True)

        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in evidence) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
