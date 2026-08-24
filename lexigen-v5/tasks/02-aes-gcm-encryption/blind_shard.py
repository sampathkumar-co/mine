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
MANIFEST = "aes_gcm_encryption_T100ms_n291598_size100_test.jsonl"
EXPECTED_GIT_BLOB_SHA1 = "249219ce000f4c0938a715f723b62f24e94b0484"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10
EXPECTED_RECORDS = 100
EXPECTED_PLAINTEXT_SIZE = 298596352
SELECTED = {
    "v5_full": "v5_full_r4_7653e3865aa7a6def4dc",
    "v5_no_transfer": "v5_no_transfer_r5_0a141855078f60fe2b98",
    "random_search": "random_search_r5_ae3b52160647eaf9707e",
    "static_template": "static_template_r3_820b1c309b6117eb268d",
    "v4_compatible": "v4_compatible_r1_bd9a928b0a959b433de2",
}


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v5-task2-blind-r1"})
            with urllib.request.urlopen(req, timeout=240) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"blind fetch exhausted retries: {url}") from last


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


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
        raise RuntimeError("blind problem is not a dictionary")
    required = {"key", "nonce", "plaintext", "associated_data"}
    if not required.issubset(raw):
        raise RuntimeError(f"blind problem missing keys: {sorted(required-set(raw))}")
    key = decode_bytes(raw["key"])
    nonce = decode_bytes(raw["nonce"])
    plaintext = decode_bytes(raw["plaintext"])
    aad = decode_bytes(raw["associated_data"], allow_none=True)
    if not isinstance(key, bytes) or len(key) != 16:
        raise RuntimeError("unexpected blind AES key")
    if not isinstance(nonce, bytes) or len(nonce) != 12:
        raise RuntimeError("unexpected blind nonce")
    if not isinstance(plaintext, bytes) or len(plaintext) != EXPECTED_PLAINTEXT_SIZE:
        raise RuntimeError(f"unexpected blind plaintext size: {len(plaintext) if isinstance(plaintext,bytes) else None}")
    if not isinstance(aad, bytes) or len(aad) != 32:
        raise RuntimeError("unexpected blind AAD")
    return {"key": key, "nonce": nonce, "plaintext": plaintext, "associated_data": aad}


def reference(problem: Problem) -> Solution:
    key = problem["key"]
    nonce = problem["nonce"]
    plaintext = problem["plaintext"]
    aad = problem.get("associated_data")
    if not isinstance(key, bytes) or not isinstance(nonce, bytes) or not isinstance(plaintext, bytes) or not isinstance(aad, bytes):
        raise TypeError("reference problem bytes invalid")
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


def selected_candidates() -> list[tuple[str, str, Callable[[Problem], Solution]]]:
    result: list[tuple[str, str, Callable[[Problem], Solution]]] = []
    for arm, wanted in SELECTED.items():
        matches = [(name, fn) for name, fn in CANDIDATES_BY_ARM[arm] if name == wanted]
        if len(matches) != 1:
            raise RuntimeError(f"selected candidate resolution failed for {arm}/{wanted}")
        result.append((arm, wanted, matches[0][1]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0,{SHARDS})")

    raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    actual_blob = git_blob_sha1(raw)
    if actual_blob != EXPECTED_GIT_BLOB_SHA1:
        raise RuntimeError(f"blind test manifest Git blob SHA-1 mismatch: {actual_blob} != {EXPECTED_GIT_BLOB_SHA1}")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected {EXPECTED_RECORDS} blind records, got {len(rows)}")

    candidates = selected_candidates()
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
            raise RuntimeError(f"reference failed on blind record {index + 1}: {reference_error}")
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
                "key_size": 16,
                "nonce_size": 12,
                "plaintext_size": EXPECTED_PLAINTEXT_SIZE,
                "aad_size": 32,
                "test_manifest_name": MANIFEST,
                "test_manifest_git_blob_sha1": EXPECTED_GIT_BLOB_SHA1,
                "test_manifest_sha256": manifest_sha256,
                "execution_order": execution_order,
                "shard": args.shard,
                "candidate_executions": 1,
                "reference_executions_for_record": 1,
                "invalid_output_retries": 0,
            })
            print(f"[{index+1}/100] {arm}/{name} valid={valid and candidate_error is None} speedup={speedup:.3f} reason={candidate_error or failure_reason}", flush=True)
        del problem, expected, candidate_results
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in evidence) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
