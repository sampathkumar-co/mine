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

import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits

from candidates import Problem, Solution
from selected_solver import CANDIDATE_NAME, solve

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "cvar_projection"
MANIFEST = "cvar_projection_T100ms_n9_size100_test.jsonl"
EXPECTED_TREE_OID = "194db2639d41ee0c91b2f171932e7984ad4aed3a"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
TREE_API = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=false&expand=false"
SHARDS = 10


def request_bytes(url: str) -> bytes:
    delays = (0, 5, 15, 30, 60)
    last_error: Exception | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-cvar-r3-blind"})
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
        raise RuntimeError(f"expected one manifest metadata entry, received {len(matches)}")
    actual = str(matches[0].get("oid", ""))
    if actual != EXPECTED_TREE_OID:
        raise RuntimeError(f"test manifest tree identity mismatch: {actual} != {EXPECTED_TREE_OID}")


def reference(problem: Problem) -> Solution:
    x0 = np.asarray(problem["x0"], dtype=np.float64)
    scenarios = np.asarray(problem["loss_scenarios"], dtype=np.float64)
    beta = float(problem["beta"])
    kappa = float(problem["kappa"])
    k = int((1.0 - beta) * scenarios.shape[0])
    variable = cp.Variable(x0.size)
    model = cp.Problem(
        cp.Minimize(cp.sum_squares(variable - x0)),
        [cp.sum_largest(scenarios @ variable, k) <= kappa * k],
    )
    model.solve()
    if model.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or variable.value is None:
        raise RuntimeError(f"reference solver failed: {model.status}")
    return {"x_proj": np.asarray(variable.value, dtype=np.float64).tolist()}


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            solution = fn(problem)
            elapsed = time.perf_counter() - start
        return solution, elapsed, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def metrics(
    problem: Problem,
    solution: Solution | None,
    reference_solution: Solution,
) -> tuple[bool, float, float, float, str | None]:
    try:
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        scenarios = np.asarray(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem["beta"])
        kappa = float(problem["kappa"])
        k = int((1.0 - beta) * scenarios.shape[0])
        projected = (
            np.asarray(solution["x_proj"], dtype=np.float64)
            if solution is not None
            else np.asarray([], dtype=np.float64)
        )
        reference_x = np.asarray(reference_solution["x_proj"], dtype=np.float64)
        if projected.shape != x0.shape or not np.all(np.isfinite(projected)):
            return False, float("inf"), float("inf"), float("inf"), "shape_or_finite"
        losses = scenarios @ projected
        indices = np.argpartition(losses, losses.size - k)[-k:]
        cvar = float(np.sum(losses[indices], dtype=np.float64) / k)
        candidate_distance = float(np.sum((projected - x0) ** 2, dtype=np.float64))
        reference_distance = float(np.sum((reference_x - x0) ** 2, dtype=np.float64))
        objective_ratio = (
            candidate_distance / reference_distance
            if reference_distance > 0.0
            else (1.0 if candidate_distance == 0.0 else float("inf"))
        )
        valid = bool(
            cvar <= kappa + 1e-4
            and candidate_distance <= reference_distance * 1.01 + 1e-10
        )
        return valid, cvar, candidate_distance, objective_ratio, None if valid else "feasibility_or_objective"
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
            relative = str(row["problem"]["loss_scenarios"]["npy_path"])
            payload = temporary / Path(relative).name
            payload.write_bytes(request_bytes(f"{BASE}/{relative}?download=true"))
            scenarios = np.load(payload, allow_pickle=False)
            problem: Problem = {
                "x0": row["problem"]["x0"],
                "loss_scenarios": scenarios,
                "beta": row["problem"]["beta"],
                "kappa": row["problem"]["kappa"],
            }

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
            valid, cvar, distance, objective_ratio, validation_reason = metrics(
                problem, candidate_solution, reference_solution
            )
            speedup = reference_s / candidate_s if candidate_s else 0.0
            record = {
                "index": index + 1,
                "seed": int(row["seed"]),
                "candidate": CANDIDATE_NAME,
                "valid": valid,
                "failure_reason": candidate_error or validation_reason,
                "cvar": cvar,
                "kappa": float(problem["kappa"]),
                "candidate_distance": distance,
                "objective_ratio": objective_ratio,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "shard": args.shard,
                "execution_order": execution_order,
                "candidate_executions": 1,
                "reference_executions": 1,
                "manifest_sha256": manifest_sha256,
                "test_manifest_tree_oid": EXPECTED_TREE_OID,
            }
            records.append(record)
            print(
                f"[{index + 1}/100] {CANDIDATE_NAME} valid={valid} speedup={speedup:.3f} "
                f"cvar={cvar:.6f} objective_ratio={objective_ratio:.6f}",
                flush=True,
            )
            del scenarios, problem, reference_solution, candidate_solution
            payload.unlink(missing_ok=True)
            gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
