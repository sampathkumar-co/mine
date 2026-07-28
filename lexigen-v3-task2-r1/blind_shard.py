from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from candidates import Problem, Solution
from selected_solver import CANDIDATE_NAME, solve

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "kmeans"
MANIFEST = "kmeans_T100ms_n278_size100_test.jsonl"
EXPECTED_TREE_OID = "f5f3fded57e9ca0fc4d1369477fd2176eeac15e2"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
TREE_API = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=false&expand=false"
SHARDS = 10


def request_bytes(url: str) -> bytes:
    last_error: Exception | None = None
    for delay in (0, 5, 15, 30, 60):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-kmeans-r1-blind"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last_error = exc
    raise RuntimeError(f"download exhausted infrastructure retries before execution: {last_error}")


def verify_manifest_identity() -> None:
    entries = json.loads(request_bytes(TREE_API).decode("utf-8"))
    matches = [entry for entry in entries if str(entry.get("path", "")).endswith("/" + MANIFEST)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one test manifest metadata entry, received {len(matches)}")
    actual = str(matches[0].get("oid", ""))
    if actual != EXPECTED_TREE_OID:
        raise RuntimeError(f"test manifest tree identity mismatch: {actual} != {EXPECTED_TREE_OID}")


def reference(problem: Problem) -> Solution:
    data = np.asarray(problem["X"], dtype=np.float64)
    clusters = int(problem["k"])
    return KMeans(n_clusters=clusters).fit_predict(data).astype(np.int64, copy=False).tolist()


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            solution = fn(problem)
            elapsed = time.perf_counter() - start
        return solution, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def inertia(data: np.ndarray, labels: np.ndarray, clusters: int) -> float:
    counts = np.bincount(labels, minlength=clusters).astype(np.float64)
    nonempty = counts > 0.0
    squared_norm_sum = float(np.sum(data * data, dtype=np.float64))
    correction = 0.0
    for column in range(data.shape[1]):
        sums = np.bincount(labels, weights=data[:, column], minlength=clusters)
        correction += float(np.sum((sums[nonempty] * sums[nonempty]) / counts[nonempty]))
    return max(squared_norm_sum - correction, 0.0)


def validate(problem: Problem, solution: Solution | None, reference_solution: Solution) -> tuple[bool, float, float, float, str | None]:
    try:
        data = np.asarray(problem["X"], dtype=np.float64)
        clusters = int(problem["k"])
        candidate_labels = np.asarray(solution if solution is not None else [], dtype=np.int64)
        reference_labels = np.asarray(reference_solution, dtype=np.int64)
        if candidate_labels.shape != (data.shape[0],):
            return False, float("inf"), float("inf"), float("inf"), "shape"
        if np.any(candidate_labels < 0) or np.any(candidate_labels >= clusters):
            return False, float("inf"), float("inf"), float("inf"), "label_range"
        candidate_loss = inertia(data, candidate_labels, clusters)
        reference_loss = inertia(data, reference_labels, clusters)
        ratio = candidate_loss / reference_loss if reference_loss > 0.0 else (1.0 if candidate_loss == 0.0 else float("inf"))
        valid = bool(0.95 * candidate_loss <= reference_loss + 1e-5)
        return valid, candidate_loss, reference_loss, ratio, None if valid else "inertia"
    except Exception as exc:
        return False, float("inf"), float("inf"), float("inf"), f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    verify_manifest_identity()
    raw = request_bytes(f"{BASE}/{MANIFEST}?download=true")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind records, received {len(rows)}")

    records: list[dict[str, object]] = []
    selected = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for index, row in selected:
            relative = str(row["problem"]["X"]["npy_path"])
            payload = temporary / Path(relative).name
            payload.write_bytes(request_bytes(f"{BASE}/{relative}?download=true"))
            data = np.load(payload, allow_pickle=False)
            problem: Problem = {"X": data, "k": int(row["problem"]["k"])}

            if index % 2 == 0:
                reference_solution, reference_s, reference_error = timed(reference, problem)
                candidate_solution, candidate_s, candidate_error = timed(solve, problem)
                execution_order = "reference_first"
            else:
                candidate_solution, candidate_s, candidate_error = timed(solve, problem)
                reference_solution, reference_s, reference_error = timed(reference, problem)
                execution_order = "candidate_first"

            if reference_solution is None or reference_s is None or reference_error is not None:
                raise RuntimeError(f"reference failed on blind record {index + 1}: {reference_error}")
            valid, candidate_loss, reference_loss, loss_ratio, validation_reason = validate(
                problem, candidate_solution, reference_solution
            )
            speedup = reference_s / candidate_s if candidate_s else 0.0
            records.append({
                "index": index + 1,
                "seed": int(row["seed"]),
                "candidate": CANDIDATE_NAME,
                "valid": valid,
                "failure_reason": candidate_error or validation_reason,
                "candidate_loss": candidate_loss,
                "reference_loss": reference_loss,
                "loss_ratio": loss_ratio,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "clusters": int(problem["k"]),
                "samples": int(data.shape[0]),
                "dimensions": int(data.shape[1]),
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions": 1,
                "manifest_sha256": manifest_sha256,
                "test_manifest_tree_oid": EXPECTED_TREE_OID,
            })
            print(
                f"[{index + 1}/100] {CANDIDATE_NAME} valid={valid} speedup={speedup:.3f} loss_ratio={loss_ratio:.6f}",
                flush=True,
            )
            del data, problem, reference_solution, candidate_solution
            payload.unlink(missing_ok=True)
            gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
