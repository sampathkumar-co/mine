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

import numpy as np

from candidates import build_candidates, reference_exact

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "max_flow_min_cost"
SHARDS = 10
EXPECTED_RECORDS = 100
TRAIN_NAME = "max_flow_min_cost_T100ms_n64_size100_train.jsonl"
TRAIN_OID = "0dd539d95028d524b813f510ebd5c35e526007f9"
TRAIN_SIZE = 24742
TEST_NAME = "max_flow_min_cost_T100ms_n64_size100_test.jsonl"
TEST_OID = "74816e40f0cd9df79c88917013f24937940ee91f"
TEST_SIZE = 24800
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
ARM_ORDER = ("v6_full", "v6_no_transfer", "random_search", "static_template", "v5_compatible", "strong_baseline")


def fetch(url: str) -> bytes:
    last = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"LEXIGEN-v6-task1-train-r1"})
            with urllib.request.urlopen(req, timeout=240) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            last = exc
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"fetch exhausted {url}") from last


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def decode_value(value):
    if isinstance(value, list):
        return [decode_value(x) for x in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind is None:
        return {k: decode_value(v) for k, v in value.items()}
    if kind == "ndarray_ref":
        rel = str(value.get("npy_path", ""))
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise RuntimeError(f"unsafe ndarray_ref {rel}")
        raw = fetch(f"{BASE}/{rel}?download=true")
        return np.load(io.BytesIO(raw), allow_pickle=False)
    if kind == "ndarray_b64":
        raw = base64.b64decode(str(value.get("data_b64", "")).encode("ascii"))
        array = np.frombuffer(raw, dtype=np.dtype(value["dtype"]))
        shape = tuple(value.get("shape", []))
        return array.reshape(shape) if shape else array
    if kind == "ndarray":
        return np.array(value["data"], dtype=np.dtype(value.get("dtype")))
    if kind == "tuple":
        return tuple(decode_value(x) for x in value.get("data", []))
    return {k: decode_value(v) for k, v in value.items() if k != "__type__"}


def decode_problem(raw):
    problem = decode_value(raw)
    if not isinstance(problem, dict) or not {"capacity", "cost", "s", "t"}.issubset(problem):
        raise RuntimeError("invalid official max_flow_min_cost problem")
    cap = np.asarray(problem["capacity"])
    cost = np.asarray(problem["cost"])
    if cap.ndim != 2 or cap.shape[0] != cap.shape[1] or cost.shape != cap.shape or cap.shape[0] == 0:
        raise RuntimeError("invalid capacity/cost shape")
    n = int(cap.shape[0])
    s, t = int(problem["s"]), int(problem["t"])
    if not (0 <= s < n and 0 <= t < n and s != t):
        raise RuntimeError("invalid source/sink")
    return problem


def timed(fn, problem):
    try:
        started = time.perf_counter_ns()
        result = fn(problem)
        return result, time.perf_counter_ns() - started, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def verify(problem: dict, got, ref) -> tuple[bool, str | None, dict]:
    try:
        cap = np.asarray(problem["capacity"], dtype=np.float64)
        cost = np.asarray(problem["cost"], dtype=np.float64)
        s, t = int(problem["s"]), int(problem["t"])
        n = len(cap)
        sol = np.asarray(got, dtype=np.float64)
        refa = np.asarray(ref, dtype=np.float64)
        tol_sem = 1e-7
        tol_official = 1e-5
        if sol.shape != (n, n) or not np.all(np.isfinite(sol)):
            return False, "format_or_nonfinite", {}
        if np.any(sol < -tol_sem):
            return False, "negative_flow", {}
        if np.any(sol - cap > tol_sem):
            return False, "capacity_exceeded", {}
        if np.any((cap == 0) & (sol > tol_sem)):
            return False, "nonedge_flow", {}
        if np.any(np.diag(sol) > tol_sem):
            return False, "self_loop", {}
        if np.any(sol[:, s] > tol_sem) or np.any(sol[t, :] > tol_sem):
            return False, "source_sink_direction", {}
        net = sol.sum(axis=1) - sol.sum(axis=0)
        for node in range(n):
            if node not in (s, t) and abs(float(net[node])) > tol_sem:
                return False, "conservation", {}
        if abs(float(net[s] + net[t])) > tol_sem or float(net[s]) < -tol_sem:
            return False, "source_sink_balance", {}
        ref_flow = float(refa[s, :].sum() - refa[:, s].sum())
        got_flow = float(sol[s, :].sum() - sol[:, s].sum())
        ref_cost = float((refa * cost).sum())
        got_cost = float((sol * cost).sum())
        if abs(got_flow - ref_flow) > tol_sem:
            return False, "not_max_flow", {"candidate_flow":got_flow,"reference_flow":ref_flow,"candidate_cost":got_cost,"reference_cost":ref_cost}
        if abs(got_cost - ref_cost) > tol_sem:
            return False, "not_min_cost", {"candidate_flow":got_flow,"reference_flow":ref_flow,"candidate_cost":got_cost,"reference_cost":ref_cost}
        # Reproduce the official acceptance conditions as a second layer, using the already-computed reference.
        for i in range(n):
            for j in range(n):
                if sol[i, j] > tol_official and sol[j, i] > tol_official:
                    return False, "official_two_way", {}
        total_out = float(sol[s, :].sum())
        total_in = float(sol[:, t].sum())
        if abs(total_out - total_in) > tol_official or total_out < float(refa[s, :].sum()) - tol_official:
            return False, "official_flow", {}
        if got_cost > ref_cost + tol_official:
            return False, "official_cost", {}
        return True, None, {"candidate_flow":got_flow,"reference_flow":ref_flow,"candidate_cost":got_cost,"reference_cost":ref_cost}
    except Exception as exc:
        return False, f"verify_exception:{type(exc).__name__}:{exc}", {}


def flat(source_text: str):
    arms = build_candidates(source_text)
    rows = []
    for arm in ARM_ORDER:
        for candidate in arms[arm]:
            rows.append(candidate)
    if len(rows) != 31:
        raise RuntimeError(f"expected 31 candidates got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError("invalid shard")

    source_raw = args.source.read_bytes()
    source_sha = hashlib.sha256(source_raw).hexdigest()
    if source_sha != "25fc1054d288b4593e53e7ab8fed7e2eb54dce5f5b130952842305af2f29fe86":
        raise RuntimeError("source identity mismatch")
    candidates = flat(source_raw.decode("utf-8"))

    manifest = fetch(f"{BASE}/{TRAIN_NAME}?download=true")
    if len(manifest) != TRAIN_SIZE or git_blob(manifest) != TRAIN_OID:
        raise RuntimeError(f"train manifest identity mismatch size={len(manifest)} blob={git_blob(manifest)}")
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    records = [json.loads(line) for line in manifest.decode("utf-8").splitlines() if line.strip()]
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected 100 records got {len(records)}")

    evidence = []
    for idx, row in ((i, r) for i, r in enumerate(records) if i % SHARDS == args.shard):
        problem = decode_problem(row["problem"])
        shift = idx % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]
        if idx % 2 == 0:
            ref, ref_ns, ref_error = timed(reference_exact, problem)
            candidate_runs = [(c, *timed(c.solve, problem)) for c in ordered]
            execution_order = "reference_first"
        else:
            candidate_runs = [(c, *timed(c.solve, problem)) for c in ordered]
            ref, ref_ns, ref_error = timed(reference_exact, problem)
            execution_order = "candidates_first"
        if ref is None or ref_ns is None or ref_error:
            raise RuntimeError(f"reference failed record {idx+1}: {ref_error}")
        for candidate, got, candidate_ns, error in candidate_runs:
            if error is None:
                valid, reason, metrics = verify(problem, got, ref)
            else:
                valid, reason, metrics = False, "exception", {}
            evidence.append({
                "index":idx + 1,
                "seed":int(row.get("seed", idx + 1)),
                "arm":candidate.arm,
                "candidate":candidate.name,
                "implementation_class":candidate.implementation_class,
                "operators":list(candidate.operators),
                "transfer_ids":list(candidate.transfer_ids),
                "learned_template":candidate.learned_template,
                "baseline_id":candidate.baseline_id,
                "valid":bool(valid and error is None),
                "semantic_and_official_certificate":bool(valid and error is None),
                "failure_reason":error or reason,
                "candidate_ns":candidate_ns,
                "reference_ns":ref_ns,
                "speedup":(ref_ns / candidate_ns) if candidate_ns and candidate_ns > 0 else 0.0,
                **metrics,
                "n":len(problem["capacity"]),
                "train_manifest_name":TRAIN_NAME,
                "train_manifest_git_blob_sha1":TRAIN_OID,
                "train_manifest_sha256":manifest_sha256,
                "expected_test_manifest_name":TEST_NAME,
                "expected_test_manifest_tree_oid":TEST_OID,
                "expected_test_manifest_size":TEST_SIZE,
                "source_sha256":source_sha,
                "execution_order":execution_order,
                "shard":args.shard,
                "invalid_output_retries":0,
                "candidate_executions":1,
                "reference_executions_for_record":1,
                "test_manifest_contents_opened":False,
                "test_payloads_opened":0,
                "verifier_capacity_loophole_exploited":False,
            })
        del problem, ref, candidate_runs
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in evidence) + "\n")
    print(json.dumps({"shard":args.shard,"rows":len(evidence),"manifest_sha256":manifest_sha256}, indent=2))


if __name__ == "__main__":
    main()
