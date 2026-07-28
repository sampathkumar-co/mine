from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np

from candidates import CANDIDATES

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "articulation_points"
MANIFEST = "articulation_points_T100ms_n837_size100_train.jsonl"
EXPECTED_MANIFEST_SHA256 = "a12a2dbce09a1e91a9790c209709a374375501589a42d92a8930a99edb5d8a6f"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10


def fetch(url: str, *, attempts: int = 8) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LEXIGEN-v3-articulation-train-r1"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
        except Exception as exc:
            last_error = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"download exhausted retries: {url}") from last_error


def download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LEXIGEN-v3-articulation-train-r1"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                with destination.open("wb") as output:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
        except Exception as exc:
            last_error = exc
        if destination.exists():
            destination.unlink()
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"payload download exhausted retries: {url}") from last_error


def reference(problem: dict[str, Any]) -> dict[str, list[int]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(problem["num_nodes"]))
    graph.add_edges_from(problem["edges"])
    points = list(nx.articulation_points(graph))
    points.sort()
    return {"articulation_points": points}


def execute(
    function: Callable[[dict[str, Any]], dict[str, list[int]]],
    problem: dict[str, Any],
) -> dict[str, object]:
    try:
        started = time.perf_counter()
        solution = function(problem)
        elapsed = time.perf_counter() - started
        points = solution.get("articulation_points")
        if not isinstance(points, list):
            raise TypeError("articulation_points must be a list")
        if not all(isinstance(node, int) for node in points):
            raise TypeError("articulation_points must contain Python integers")
        return {"elapsed_s": elapsed, "points": points, "error": None}
    except Exception as exc:
        return {
            "elapsed_s": None,
            "points": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def warm_up() -> None:
    edges = np.asarray(
        [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 3]],
        dtype=np.int64,
    )
    problem = {"num_nodes": 7, "edges": edges}
    expected = reference(problem)["articulation_points"]
    for name, candidate in CANDIDATES.items():
        result = candidate(problem)["articulation_points"]
        if result != expected:
            raise RuntimeError(f"{name} failed ndarray warm-up")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    manifest_raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(manifest_raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("training manifest SHA-256 mismatch")
    rows = [json.loads(line) for line in manifest_raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")

    warm_up()
    selected = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    records: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.parent / f"edges-shard-{args.shard}.npy"

    for index, row in selected:
        problem_descriptor = row.get("problem")
        if not isinstance(problem_descriptor, dict):
            raise TypeError("expected mapping problem descriptor")
        num_nodes = problem_descriptor.get("num_nodes")
        edge_descriptor = problem_descriptor.get("edges")
        if not isinstance(num_nodes, int):
            raise TypeError("expected inline integer num_nodes")
        if not isinstance(edge_descriptor, dict) or edge_descriptor.get("__type__") != "ndarray_ref":
            raise TypeError("expected edges ndarray_ref descriptor")
        relative = edge_descriptor.get("npy_path")
        if not isinstance(relative, str):
            raise TypeError("edges ndarray_ref is missing npy_path")

        download(f"{BASE}/{relative}?download=true", scratch)
        edges = np.load(scratch, allow_pickle=False)
        scratch.unlink()
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise RuntimeError(f"expected edge matrix shape (m,2), received {edges.shape}")
        if not np.issubdtype(edges.dtype, np.integer):
            raise RuntimeError(f"expected integer edge matrix, received {edges.dtype}")
        problem = {"num_nodes": num_nodes, "edges": edges}

        names = list(CANDIDATES)
        rotation = index % len(names)
        names = names[rotation:] + names[:rotation]
        if index % 2:
            names.reverse()

        candidate_results: dict[str, dict[str, object]] = {}
        if index % 2 == 0:
            reference_result = execute(reference, problem)
            for name in names:
                candidate_results[name] = execute(CANDIDATES[name], problem)
        else:
            for name in names:
                candidate_results[name] = execute(CANDIDATES[name], problem)
            reference_result = execute(reference, problem)

        if reference_result["error"] is not None or reference_result["elapsed_s"] is None:
            raise RuntimeError(f"reference failed on record {index + 1}: {reference_result['error']}")
        reference_points = reference_result["points"]
        if not isinstance(reference_points, list):
            raise RuntimeError("reference did not return a list")
        reference_seconds = float(reference_result["elapsed_s"])

        for name in names:
            candidate_result = candidate_results[name]
            candidate_points = candidate_result["points"]
            candidate_seconds = candidate_result["elapsed_s"]
            valid = bool(
                candidate_result["error"] is None
                and isinstance(candidate_points, list)
                and candidate_points == reference_points
            )
            speedup = reference_seconds / float(candidate_seconds) if candidate_seconds else 0.0
            records.append({
                "index": index + 1,
                "seed": row.get("seed"),
                "candidate": name,
                "valid": valid,
                "failure_reason": candidate_result["error"] if candidate_result["error"] else (None if valid else "output_mismatch"),
                "candidate_s": candidate_seconds,
                "reference_s": reference_seconds,
                "speedup": speedup,
                "num_nodes": num_nodes,
                "edge_count": int(edges.shape[0]),
                "articulation_count": len(reference_points),
                "shard": args.shard,
                "reference_order": "first" if index % 2 == 0 else "last",
            })
            print(
                f"[{index + 1}/100] {name} valid={valid} edges={edges.shape[0]} "
                f"candidate={candidate_seconds!s}s reference={reference_seconds:.6f}s "
                f"speedup={speedup:.3f}",
                flush=True,
            )

        del candidate_results, reference_result, reference_points, problem, edges
        gc.collect()

    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
