from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import networkx as nx
from threadpoolctl import threadpool_limits

from candidates import Problem, Solution
from selected_solvers import SELECTED

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "communicability"
MANIFEST = "communicability_T100ms_n61_size100_test.jsonl"
EXPECTED_TREE_OID = "48efda1a3b334b13efb874355985d9f1d87291f2"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
RTOL = 1e-5
ATOL = 1e-8


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task1-one-shot-blind"})
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"blind fetch exhausted infrastructure retries: {url}") from last


def reference(problem: Problem) -> Solution:
    adjacency = problem["adjacency_list"]
    if not isinstance(adjacency, list):
        raise ValueError("adjacency_list must be a list")
    n = len(adjacency)
    if n == 0:
        return {"communicability": {}}
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for u, neighbors in enumerate(adjacency):
        if not isinstance(neighbors, list):
            raise ValueError("adjacency row must be a list")
        for v in neighbors:
            vertex = int(v)
            if u < vertex:
                graph.add_edge(u, vertex)
    raw = nx.communicability(graph)
    return {"communicability": {u: {v: float(raw[u][v]) for v in range(n)} for u in range(n)}}


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            solution = fn(problem)
            elapsed = time.perf_counter() - start
        return solution, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def validate(problem: Problem, proposed: Solution | None, expected: Solution) -> tuple[bool, float, str | None]:
    adjacency = problem["adjacency_list"]
    if not isinstance(adjacency, list):
        return False, float("inf"), "problem_format"
    n = len(adjacency)
    if not isinstance(proposed, dict):
        return False, float("inf"), "solution_not_dict"
    value = proposed.get("communicability")
    if not isinstance(value, dict) or set(value) != set(range(n)):
        return False, float("inf"), "outer_keys"
    target = expected["communicability"]
    maximum_error = 0.0
    for u in range(n):
        row = value.get(u)
        if not isinstance(row, dict) or set(row) != set(range(n)):
            return False, float("inf"), f"inner_keys_{u}"
        for v in range(n):
            try:
                actual = float(row[v])
                wanted = float(target[u][v])
            except Exception as exc:
                return False, float("inf"), f"value_{u}_{v}_{type(exc).__name__}"
            if not math.isfinite(actual):
                return False, float("inf"), f"nonfinite_{u}_{v}"
            maximum_error = max(maximum_error, abs(actual - wanted))
            if not math.isclose(actual, wanted, rel_tol=RTOL, abs_tol=ATOL):
                return False, maximum_error, f"mismatch_{u}_{v}"
    return True, maximum_error, None


def component_count(adjacency: list[list[int]]) -> int:
    unseen = set(range(len(adjacency)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v in unseen:
                    unseen.remove(v)
                    stack.append(v)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError("invalid shard")

    raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind rows, received {len(rows)}")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    manifest_git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()

    selected = list(SELECTED.items())
    evidence: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if index % SHARDS != args.shard:
            continue
        problem_raw = row.get("problem")
        if not isinstance(problem_raw, dict) or not isinstance(problem_raw.get("adjacency_list"), list):
            raise RuntimeError(f"record {index + 1} problem format invalid")
        adjacency = [[int(v) for v in neighbors] for neighbors in problem_raw["adjacency_list"]]
        problem: Problem = {"adjacency_list": adjacency}
        shift = index % len(selected)
        order = selected[shift:] + selected[:shift]
        if index % 2 == 0:
            expected, reference_s, reference_error = timed(reference, problem)
            candidate_results = [(arm, name, *timed(fn, problem)) for arm, (name, fn) in order]
            execution_order = "reference_first"
        else:
            candidate_results = [(arm, name, *timed(fn, problem)) for arm, (name, fn) in order]
            expected, reference_s, reference_error = timed(reference, problem)
            execution_order = "candidates_first"
        if expected is None or reference_s is None or reference_error is not None:
            raise RuntimeError(f"reference failed on blind record {index + 1}: {reference_error}")

        for arm, candidate, proposed, candidate_s, candidate_error in candidate_results:
            valid, maximum_error, validation_error = validate(problem, proposed, expected)
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": candidate,
                "valid": valid,
                "failure_reason": candidate_error or validation_error,
                "maximum_absolute_error": maximum_error,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": reference_s / candidate_s if candidate_s and candidate_s > 0 else 0.0,
                "nodes": len(adjacency),
                "edges": sum(len(neighbors) for neighbors in adjacency) // 2,
                "components": component_count(adjacency),
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
                "test_manifest_sha256": manifest_sha256,
                "test_manifest_git_blob_sha1": manifest_git_blob,
                "expected_test_tree_oid": EXPECTED_TREE_OID
            })
            print(f"[{index + 1}/100] {arm}/{candidate} valid={valid} speedup={evidence[-1]['speedup']:.3f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(record, separators=(",", ":")) for record in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
