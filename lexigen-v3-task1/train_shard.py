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

from candidates import CANDIDATES, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "cvar_projection"
MANIFEST = "cvar_projection_T100ms_n9_size100_train.jsonl"
EXPECTED_SHA256 = "f425911dee9a17939392f6f01fd4622a33ab8365c95e956e03c13351af099eb8"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10


def request_bytes(url: str) -> bytes:
    delays = (0, 5, 15, 30, 60)
    last_error: Exception | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-task1-train"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last_error = exc
    raise RuntimeError(f"download exhausted infrastructure retries: {last_error}")


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


def metrics(problem: Problem, solution: Solution | None, reference_solution: Solution) -> tuple[bool, float, float, float, str | None]:
    try:
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        scenarios = np.asarray(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem["beta"])
        kappa = float(problem["kappa"])
        k = int((1.0 - beta) * scenarios.shape[0])
        projected = np.asarray(solution["x_proj"], dtype=np.float64) if solution is not None else np.asarray([])
        reference_x = np.asarray(reference_solution["x_proj"], dtype=np.float64)
        if projected.shape != x0.shape or not np.all(np.isfinite(projected)):
            return False, float("inf"), float("inf"), float("inf"), "shape_or_finite"
        losses = scenarios @ projected
        indices = np.argpartition(losses, losses.size - k)[-k:]
        cvar = float(np.sum(losses[indices], dtype=np.float64) / k)
        candidate_distance = float(np.sum((projected - x0) ** 2, dtype=np.float64))
        reference_distance = float(np.sum((reference_x - x0) ** 2, dtype=np.float64))
        objective_ratio = candidate_distance / reference_distance if reference_distance > 0.0 else (1.0 if candidate_distance == 0.0 else float("inf"))
        valid = bool(cvar <= kappa + 1e-4 and candidate_distance <= reference_distance * 1.01 + 1e-10)
        reason = None if valid else "feasibility_or_objective"
        return valid, cvar, candidate_distance, objective_ratio, reason
    except Exception as exc:
        return False, float("inf"), float("inf"), float("inf"), f"{type(exc).__name__}: {exc}"


def warm_up() -> None:
    rng = np.random.default_rng(7)
    problem: Problem = {
        "x0": rng.standard_normal(8).tolist(),
        "loss_scenarios": rng.standard_normal((80, 8)),
        "beta": 0.95,
        "kappa": 0.1,
    }
    reference_solution = reference(problem)
    for name, candidate in CANDIDATES.items():
        solution = candidate(problem)
        valid, _, _, _, reason = metrics(problem, solution, reference_solution)
        if not valid:
            raise RuntimeError(f"synthetic warm-up failed for {name}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    raw = request_bytes(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("training manifest content hash mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, received {len(rows)}")

    warm_up()
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
            candidate_order = list(CANDIDATES)
            shift = index % len(candidate_order)
            candidate_order = candidate_order[shift:] + candidate_order[:shift]
            if index % 2 == 0:
                reference_solution, reference_s, reference_error = timed(reference, problem)
                candidate_results = [
                    (name, *timed(CANDIDATES[name], problem)) for name in candidate_order
                ]
                execution_order = "reference_first"
            else:
                candidate_results = [
                    (name, *timed(CANDIDATES[name], problem)) for name in candidate_order
                ]
                reference_solution, reference_s, reference_error = timed(reference, problem)
                execution_order = "candidates_first"
            if reference_solution is None or reference_s is None or reference_error is not None:
                raise RuntimeError(f"reference failed on record {index + 1}: {reference_error}")

            for name, solution, candidate_s, candidate_error in candidate_results:
                valid, cvar, distance, objective_ratio, validation_reason = metrics(
                    problem, solution, reference_solution
                )
                failure_reason = candidate_error or validation_reason
                speedup = reference_s / candidate_s if candidate_s else 0.0
                record = {
                    "index": index + 1,
                    "seed": int(row["seed"]),
                    "candidate": name,
                    "valid": valid,
                    "failure_reason": failure_reason,
                    "cvar": cvar,
                    "kappa": float(problem["kappa"]),
                    "candidate_distance": distance,
                    "objective_ratio": objective_ratio,
                    "candidate_s": candidate_s,
                    "reference_s": reference_s,
                    "speedup": speedup,
                    "shard": args.shard,
                    "execution_order": execution_order,
                }
                records.append(record)
                print(
                    f"[{index + 1}/100] {name} valid={valid} speedup={speedup:.3f} "
                    f"cvar={cvar:.6f} objective_ratio={objective_ratio:.6f}",
                    flush=True,
                )
            del scenarios, problem, reference_solution, candidate_results
            payload.unlink(missing_ok=True)
            gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
