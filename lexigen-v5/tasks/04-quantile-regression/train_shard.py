from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import QuantileRegressor

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "quantile_regression"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
TRAIN_NAME = "quantile_regression_T100ms_n356_size100_train.jsonl"
TRAIN_OID = "0e0c8234f88d313d8f4e2066c05ba72577e3c161"
TRAIN_SIZE = 691744
TEST_NAME = "quantile_regression_T100ms_n356_size100_test.jsonl"
TEST_OID = "fa47823b663ed89c95d00ac4315fb0086000ae80"
TEST_SIZE = 691920
SHARDS = 10
EXPECTED_RECORDS = 100

if any(len(rows) != 6 for rows in CANDIDATES_BY_ARM.values()) or sum(len(rows) for rows in CANDIDATES_BY_ARM.values()) != 30:
    raise RuntimeError("Task 4 training requires exactly 30 frozen synthetic-eligible candidates")


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v5-task4-train-r1"})
            with urllib.request.urlopen(req, timeout=240) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"training fetch exhausted retries: {url}") from last


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch_training_manifest() -> bytes:
    raw = fetch(f"{BASE}/{TRAIN_NAME}?download=true")
    if len(raw) != TRAIN_SIZE:
        raise RuntimeError(f"training manifest size mismatch: {len(raw)} != {TRAIN_SIZE}")
    actual_blob = git_blob_sha1(raw)
    if actual_blob != TRAIN_OID:
        raise RuntimeError(f"training manifest Git blob mismatch: {actual_blob} != {TRAIN_OID}")
    return raw


def decode_array(value: object) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, dict):
        raise RuntimeError(f"unsupported array encoding: {type(value).__name__}")
    kind = value.get("__type__")
    if kind == "ndarray_ref":
        relative = str(value.get("npy_path", ""))
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe ndarray_ref path: {relative}")
        return np.load(io.BytesIO(fetch(f"{BASE}/{relative}?download=true")), allow_pickle=False)
    if kind == "ndarray_b64":
        raw = base64.b64decode(str(value["data_b64"]).encode("ascii"), validate=True)
        return np.frombuffer(raw, dtype=np.dtype(str(value["dtype"]))).reshape(tuple(int(x) for x in value["shape"]))
    if kind == "ndarray":
        return np.asarray(value["data"], dtype=np.dtype(str(value.get("dtype", "float64"))))
    raise RuntimeError(f"unsupported ndarray wrapper: {kind!r}")


def decode_problem(raw: object) -> Problem:
    if not isinstance(raw, dict):
        raise RuntimeError("training problem is not a dictionary")
    required = {"X", "y", "quantile", "fit_intercept"}
    if not required.issubset(raw):
        raise RuntimeError(f"training problem missing keys: {sorted(required-set(raw))}")
    X = np.asarray(decode_array(raw["X"]), dtype=np.float64)
    y = np.asarray(decode_array(raw["y"]), dtype=np.float64)
    q = float(raw["quantile"])
    fit = bool(raw["fit_intercept"])
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] < 4 or X.shape[1] < 1:
        raise RuntimeError(f"invalid official quantile-regression shapes: X={X.shape} y={y.shape}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or not 0.0 < q < 1.0:
        raise RuntimeError("invalid official quantile-regression values")
    return {"X": X, "y": y, "quantile": q, "fit_intercept": fit}


def reference(problem: Problem) -> Solution:
    X = np.asarray(problem["X"], dtype=float)
    y = np.asarray(problem["y"], dtype=float)
    q = float(problem["quantile"])
    fit = bool(problem["fit_intercept"])
    model = QuantileRegressor(quantile=q, alpha=0.0, fit_intercept=fit, solver="highs")
    model.fit(X, y)
    intercept = float(model.intercept_) if fit else 0.0
    return {
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": [intercept],
        "predictions": np.asarray(model.predict(X), dtype=float).tolist(),
    }


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        return solution, time.perf_counter() - start, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def verify(proposed: Solution | None, expected: Solution) -> tuple[bool, str | None, float | None]:
    if not isinstance(proposed, dict) or not all(k in proposed for k in ("coef", "intercept", "predictions")):
        return False, "missing_required_key", None
    try:
        coef = np.asarray(proposed["coef"], dtype=float)
        intercept = np.asarray(proposed["intercept"], dtype=float)
        predictions = np.asarray(proposed["predictions"], dtype=float)
        expected_coef = np.asarray(expected["coef"], dtype=float)
        expected_intercept = np.asarray(expected["intercept"], dtype=float)
        expected_predictions = np.asarray(expected["predictions"], dtype=float)
    except Exception as exc:
        return False, f"decode_failure:{type(exc).__name__}", None
    if coef.shape != expected_coef.shape or intercept.shape != expected_intercept.shape or predictions.shape != expected_predictions.shape:
        return False, "shape_mismatch", None
    max_abs_error = max(
        float(np.max(np.abs(coef - expected_coef))) if coef.size else 0.0,
        float(np.max(np.abs(intercept - expected_intercept))) if intercept.size else 0.0,
        float(np.max(np.abs(predictions - expected_predictions))) if predictions.size else 0.0,
    )
    valid = bool(
        np.allclose(coef, expected_coef, atol=1e-5)
        and np.allclose(intercept, expected_intercept, atol=1e-5)
        and np.allclose(predictions, expected_predictions, atol=1e-5)
    )
    return valid, None if valid else "authoritative_tolerance_mismatch", max_abs_error


def flattened_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    result: list[tuple[str, str, Callable[[Problem], Solution]]] = []
    for arm, rows in CANDIDATES_BY_ARM.items():
        for name, fn in rows:
            result.append((arm, name, fn))
    if len(result) != 30:
        raise RuntimeError(f"expected 30 candidates, got {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0,{SHARDS})")

    raw = fetch_training_manifest()
    train_sha256 = hashlib.sha256(raw).hexdigest()
    train_blob = git_blob_sha1(raw)
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected {EXPECTED_RECORDS} training records, got {len(records)}")

    candidates = flattened_candidates()
    evidence: list[dict[str, object]] = []
    for index, row in ((i, r) for i, r in enumerate(records) if i % SHARDS == args.shard):
        problem = decode_problem(row.get("problem"))
        shift = index % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]
        if index % 2 == 0:
            expected, reference_s, reference_error = timed(reference, problem)
            candidate_results = [(arm, name, *timed(fn, problem)) for arm, name, fn in ordered]
            execution_order = "reference_first"
        else:
            candidate_results = [(arm, name, *timed(fn, problem)) for arm, name, fn in ordered]
            expected, reference_s, reference_error = timed(reference, problem)
            execution_order = "candidates_first"
        if expected is None or reference_s is None or reference_error is not None:
            raise RuntimeError(f"reference failed on training record {index + 1}: {reference_error}")
        X = np.asarray(problem["X"])
        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            if candidate_error is None:
                valid, failure_reason, max_abs_error = verify(proposed, expected)
            else:
                valid, failure_reason, max_abs_error = False, "exception", None
            speedup = reference_s / candidate_s if candidate_s is not None and candidate_s > 0 else 0.0
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": bool(valid and candidate_error is None),
                "failure_reason": candidate_error or failure_reason,
                "max_abs_error": max_abs_error,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "X_shape": list(X.shape),
                "quantile": float(problem["quantile"]),
                "fit_intercept": bool(problem["fit_intercept"]),
                "train_manifest_name": TRAIN_NAME,
                "train_manifest_tree_oid": TRAIN_OID,
                "train_manifest_git_blob_sha1": train_blob,
                "train_manifest_sha256": train_sha256,
                "expected_test_manifest_name": TEST_NAME,
                "expected_test_manifest_tree_oid": TEST_OID,
                "expected_test_manifest_size": TEST_SIZE,
                "execution_order": execution_order,
                "shard": args.shard,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
                "test_manifest_contents_opened": False,
                "test_payloads_opened": 0,
            })
            print(f"[{index+1}/100] {arm}/{name} valid={valid and candidate_error is None} speedup={speedup:.3f} reason={candidate_error or failure_reason}", flush=True)
        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
