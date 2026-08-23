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

import hdbscan
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "clustering_outliers"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=true&expand=false"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
EXPECTED_RECORDS = 100

ELIGIBLE = {
    "v5_full": {
        "v5_full_r2_41510e43e8fafb598496",
        "v5_full_r3_a6102573c9f355414229",
        "v5_full_r4_4abf2b51384c560522e8",
        "v5_full_r5_c50e493c5549a408f3e5",
        "v5_full_r6_514b3e8a41ba1f8b73a1",
    },
    "v5_no_transfer": {
        "v5_no_transfer_r1_91e027e622f2d9a98240",
        "v5_no_transfer_r2_b2614109e1a5ccc10c14",
        "v5_no_transfer_r3_20375ceceffce4d406a4",
        "v5_no_transfer_r4_4a4e1871b7f7b48b9485",
        "v5_no_transfer_r5_d69e86803f54c5a83d06",
        "v5_no_transfer_r6_66c5848a3c8a4f51b562",
    },
    "random_search": {
        "random_search_r1_281d4a03f9bc5812f7af",
        "random_search_r2_0d6f85272e04650e490b",
        "random_search_r3_38776670db84b717ed92",
        "random_search_r4_c7951713adcdd3e83e67",
        "random_search_r5_57b17f2971c60d7d437b",
    },
    "static_template": {
        "static_template_r2_8fd871e046faa7e4d37c",
        "static_template_r3_820b1c309b6117eb268d",
        "static_template_r4_8f1dafda0d3fbc099aa9",
        "static_template_r5_357e80313b8b9dc3cf36",
        "static_template_r6_d044a19fd4551034dc11",
    },
    "v4_compatible": {
        "v4_compatible_r1_f9f3239b6866512e4f68",
        "v4_compatible_r2_9f5f55df04a5ad23f542",
        "v4_compatible_r3_ec4b9c17aaa3767d4f6",
        "v4_compatible_r4_7c30efb65d2c20ff8cc9",
        "v4_compatible_r5_3df5ed91505aea4ed6cb",
        "v4_compatible_r6_0dde88a4a159a3ad0e40",
    },
}


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v5-task1-train-r1"})
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


def discover_training() -> tuple[str, str, str, str, bytes]:
    tree = json.loads(fetch(TREE_URL))
    files = [x for x in tree if x.get("type") == "file"]
    train = [x for x in files if str(x.get("path", "")).endswith("_train.jsonl")]
    test = [x for x in files if str(x.get("path", "")).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError(f"expected exactly one train/test manifest identity, got train={len(train)} test={len(test)}")
    train_name = Path(str(train[0]["path"])).name
    test_name = Path(str(test[0]["path"])).name
    raw = fetch(f"{BASE}/{train_name}?download=true")
    return train_name, str(train[0].get("oid", "")), test_name, str(test[0].get("oid", "")), raw


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
    if "dataset" not in raw:
        raise RuntimeError("training problem lacks dataset")
    dataset = np.asarray(decode_array(raw["dataset"]), dtype=np.float64)
    if dataset.ndim != 2 or dataset.shape[0] < 2 or dataset.shape[1] < 1 or not np.all(np.isfinite(dataset)):
        raise RuntimeError(f"invalid official training dataset shape/values: {dataset.shape}")
    min_cluster_size = int(raw.get("min_cluster_size", 5))
    min_samples = int(raw.get("min_samples", 3))
    if min_cluster_size < 2 or min_samples < 1:
        raise RuntimeError("invalid official clustering parameters")
    return {"dataset": dataset, "min_cluster_size": min_cluster_size, "min_samples": min_samples}


def reference(problem: Problem) -> Solution:
    dataset = np.asarray(problem["dataset"], dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(problem["min_cluster_size"]), min_samples=int(problem["min_samples"]))
    clusterer.fit(dataset)
    labels = clusterer.labels_
    return {
        "labels": labels.tolist(),
        "probabilities": clusterer.probabilities_.tolist(),
        "cluster_persistence": clusterer.cluster_persistence_.tolist(),
        "num_clusters": len(set(labels[labels != -1])),
        "num_noise_points": int(np.sum(labels == -1)),
    }


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        return solution, time.perf_counter() - start, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def verify(problem: Problem, proposed: Solution | None, expected: Solution) -> tuple[bool, str | None, dict[str, float]]:
    if not isinstance(proposed, dict) or not all(k in proposed for k in ("labels", "probabilities", "cluster_persistence")):
        return False, "missing_required_key", {}
    try:
        labels = np.asarray(proposed["labels"])
        probabilities = np.asarray(proposed["probabilities"], dtype=np.float64)
    except Exception:
        return False, "decode_failure", {}
    dataset = np.asarray(problem["dataset"])
    if len(labels) != len(dataset):
        return False, "labels_length", {}
    if not np.all(np.logical_or(labels == -1, labels >= 0)):
        return False, "invalid_labels", {}
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1):
        return False, "invalid_probabilities", {}
    ref_labels = np.asarray(expected["labels"])
    num_clusters = len(set(labels[labels != -1]))
    ref_num_clusters = len(set(ref_labels[ref_labels != -1]))
    cluster_deviation = abs(num_clusters - ref_num_clusters) / max(1, ref_num_clusters)
    if cluster_deviation > 0.3:
        return False, "cluster_count_deviation", {"cluster_deviation": float(cluster_deviation)}
    noise_ratio = float(np.sum(labels == -1) / len(labels))
    ref_noise_ratio = float(np.sum(ref_labels == -1) / len(ref_labels))
    noise_deviation = abs(noise_ratio - ref_noise_ratio)
    if noise_deviation > 0.2:
        return False, "noise_deviation", {"cluster_deviation": float(cluster_deviation), "noise_deviation": float(noise_deviation)}
    ari = 1.0
    if num_clusters > 0 and ref_num_clusters > 0:
        ari = float(adjusted_rand_score(ref_labels, labels))
        if ari < 0.5:
            return False, "ari", {"ari": ari, "cluster_deviation": float(cluster_deviation), "noise_deviation": float(noise_deviation)}
    return True, None, {"ari": ari, "cluster_deviation": float(cluster_deviation), "noise_deviation": float(noise_deviation)}


def flattened_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    result: list[tuple[str, str, Callable[[Problem], Solution]]] = []
    for arm, rows in CANDIDATES_BY_ARM.items():
        for name, fn in rows:
            if name in ELIGIBLE[arm]:
                result.append((arm, name, fn))
    if len(result) != 27:
        raise RuntimeError(f"expected 27 synthetic-eligible candidates, got {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0,{SHARDS})")

    train_name, train_oid, test_name, test_oid, raw = discover_training()
    train_sha256 = hashlib.sha256(raw).hexdigest()
    train_blob = git_blob_sha1(raw)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected {EXPECTED_RECORDS} training records, got {len(rows)}")

    candidates = flattened_candidates()
    evidence: list[dict[str, object]] = []
    for index, row in ((i, r) for i, r in enumerate(rows) if i % SHARDS == args.shard):
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
        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            if candidate_error is None:
                valid, failure_reason, metrics = verify(problem, proposed, expected)
            else:
                valid, failure_reason, metrics = False, "exception", {}
            speedup = reference_s / candidate_s if candidate_s is not None and candidate_s > 0 else 0.0
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": bool(valid and candidate_error is None),
                "failure_reason": candidate_error or failure_reason,
                "metrics": metrics,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "dataset_shape": list(np.asarray(problem["dataset"]).shape),
                "min_cluster_size": int(problem["min_cluster_size"]),
                "min_samples": int(problem["min_samples"]),
                "train_manifest_name": train_name,
                "train_manifest_tree_oid": train_oid,
                "train_manifest_git_blob_sha1": train_blob,
                "train_manifest_sha256": train_sha256,
                "expected_test_manifest_name": test_name,
                "expected_test_manifest_tree_oid": test_oid,
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
