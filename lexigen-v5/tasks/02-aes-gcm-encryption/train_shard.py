from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from candidates import CANDIDATES_BY_ARM, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "aes_gcm_encryption"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=true&expand=false"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
EXPECTED_RECORDS = 100

ELIGIBLE = {arm: {name for name, _ in rows} for arm, rows in CANDIDATES_BY_ARM.items()}
if any(len(names) != 6 for names in ELIGIBLE.values()) or sum(len(names) for names in ELIGIBLE.values()) != 30:
    raise RuntimeError("Task 2 training must contain exactly six synthetic-eligible candidates per arm")


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v5-task2-train-r1"})
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


def decode_bytes(value: object, *, allow_none: bool = False) -> bytes | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bytes):
        return value
    if not isinstance(value, dict):
        raise RuntimeError(f"unsupported bytes encoding: {type(value).__name__}")
    kind = value.get("__type__")
    if kind == "bytes":
        return base64.b64decode(str(value["data_b64"]).encode("ascii"), validate=True)
    if kind == "bytes_ref":
        relative = str(value.get("bin_path", ""))
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe bytes_ref path: {relative}")
        raw = fetch(f"{BASE}/{relative}?download=true")
        expected_size = value.get("size")
        if expected_size is not None and len(raw) != int(expected_size):
            raise RuntimeError(f"bytes_ref size mismatch for {relative}: {len(raw)} != {expected_size}")
        return raw
    raise RuntimeError(f"unsupported bytes wrapper: {kind!r}")


def decode_problem(raw: object) -> Problem:
    if not isinstance(raw, dict):
        raise RuntimeError("training problem is not a dictionary")
    required = {"key", "nonce", "plaintext", "associated_data"}
    if not required.issubset(raw):
        raise RuntimeError(f"training problem missing keys: {sorted(required-set(raw))}")
    key = decode_bytes(raw["key"])
    nonce = decode_bytes(raw["nonce"])
    plaintext = decode_bytes(raw["plaintext"])
    aad = decode_bytes(raw["associated_data"], allow_none=True)
    if not isinstance(key, bytes) or len(key) not in (16, 24, 32):
        raise RuntimeError("invalid official AES key")
    if not isinstance(nonce, bytes):
        raise RuntimeError("invalid official nonce")
    if not isinstance(plaintext, bytes):
        raise RuntimeError("invalid official plaintext")
    if aad is not None and not isinstance(aad, bytes):
        raise RuntimeError("invalid official associated data")
    return {"key": key, "nonce": nonce, "plaintext": plaintext, "associated_data": aad}


def reference(problem: Problem) -> Solution:
    key = problem["key"]
    nonce = problem["nonce"]
    plaintext = problem["plaintext"]
    aad = problem.get("associated_data")
    if not isinstance(key, bytes) or not isinstance(nonce, bytes) or not isinstance(plaintext, bytes):
        raise TypeError("reference problem bytes invalid")
    if aad is not None and not isinstance(aad, bytes):
        raise TypeError("reference AAD invalid")
    combined = AESGCM(key).encrypt(nonce, plaintext, aad)
    return {"ciphertext": combined[:-16], "tag": combined[-16:]}


def timed(fn: Callable[[Problem], Solution], problem: Problem) -> tuple[Solution | None, float | None, str | None]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        return solution, time.perf_counter() - start, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def verify(problem: Problem, proposed: Solution | None, expected: Solution) -> tuple[bool, str | None]:
    if not isinstance(proposed, dict) or set(proposed) != {"ciphertext", "tag"}:
        return False, "invalid_solution_shape"
    ciphertext = proposed.get("ciphertext")
    tag = proposed.get("tag")
    if not isinstance(ciphertext, bytes) or not isinstance(tag, bytes):
        return False, "non_bytes_output"
    plaintext = problem["plaintext"]
    if not isinstance(plaintext, bytes) or len(ciphertext) != len(plaintext):
        return False, "ciphertext_length"
    if len(tag) != 16:
        return False, "tag_length"
    if not hmac.compare_digest(expected["ciphertext"], ciphertext):
        return False, "ciphertext_mismatch"
    if not hmac.compare_digest(expected["tag"], tag):
        return False, "tag_mismatch"
    return True, None


def flattened_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    result: list[tuple[str, str, Callable[[Problem], Solution]]] = []
    for arm, rows in CANDIDATES_BY_ARM.items():
        for name, fn in rows:
            if name in ELIGIBLE[arm]:
                result.append((arm, name, fn))
    if len(result) != 30:
        raise RuntimeError(f"expected 30 synthetic-eligible candidates, got {len(result)}")
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
        plaintext = problem["plaintext"]
        aad = problem.get("associated_data")
        key = problem["key"]
        nonce = problem["nonce"]
        for arm, name, proposed, candidate_s, candidate_error in candidate_results:
            if candidate_error is None:
                valid, failure_reason = verify(problem, proposed, expected)
            else:
                valid, failure_reason = False, "exception"
            speedup = reference_s / candidate_s if candidate_s is not None and candidate_s > 0 else 0.0
            evidence.append({
                "index": index + 1,
                "seed": int(row.get("seed", index + 1)),
                "arm": arm,
                "candidate": name,
                "valid": bool(valid and candidate_error is None),
                "failure_reason": candidate_error or failure_reason,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "key_size": len(key) if isinstance(key, bytes) else None,
                "nonce_size": len(nonce) if isinstance(nonce, bytes) else None,
                "plaintext_size": len(plaintext) if isinstance(plaintext, bytes) else None,
                "aad_size": len(aad) if isinstance(aad, bytes) else None,
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
                "test_payloads_opened": 0
            })
            print(f"[{index+1}/100] {arm}/{name} valid={valid and candidate_error is None} speedup={speedup:.3f} reason={candidate_error or failure_reason}", flush=True)
        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
