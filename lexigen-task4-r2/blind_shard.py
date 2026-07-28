from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from candidate import Problem, Solution, solve

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "chacha_encryption"
MANIFEST = "chacha_encryption_T100ms_n197380_size100_test.jsonl"
EXPECTED_OID = "b6903231c88291a8eff777a519a9af0736148a0f"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
TAG_SIZE = 16
SHARDS = 10


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LEXIGEN-v2-blind"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                with destination.open("wb") as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            return
        except Exception as exc:  # infrastructure retry only; never candidate retry
            last_error = exc
            destination.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed after three infrastructure attempts: {url}") from last_error


def decode_inline_bytes(value: dict[str, object]) -> bytes:
    if value.get("__type__") != "bytes":
        raise TypeError(f"expected inline bytes descriptor, received {value.get('__type__')!r}")
    return base64.b64decode(str(value["data_b64"]), validate=True)


def reference(problem: Problem) -> Solution:
    combined = ChaCha20Poly1305(problem["key"]).encrypt(
        problem["nonce"],
        problem["plaintext"],
        problem["associated_data"],
    )
    if len(combined) < TAG_SIZE:
        raise RuntimeError("reference output shorter than authentication tag")
    return {"ciphertext": combined[:-TAG_SIZE], "tag": combined[-TAG_SIZE:]}


def fingerprint(solution: Solution, plaintext_size: int) -> tuple[str, int, int]:
    ciphertext = solution.get("ciphertext")
    tag = solution.get("tag")
    if not isinstance(ciphertext, bytes) or not isinstance(tag, bytes):
        raise TypeError("solution values must be bytes")
    if len(ciphertext) != plaintext_size:
        raise ValueError("ciphertext length mismatch")
    if len(tag) != TAG_SIZE:
        raise ValueError("tag length mismatch")
    digest = hashlib.sha256()
    digest.update(len(ciphertext).to_bytes(8, "big"))
    digest.update(ciphertext)
    digest.update(len(tag).to_bytes(8, "big"))
    digest.update(tag)
    return digest.hexdigest(), len(ciphertext), len(tag)


def execute_once(fn: Callable[[Problem], Solution], problem: Problem) -> dict[str, object]:
    try:
        start = time.perf_counter()
        solution = fn(problem)
        elapsed = time.perf_counter() - start
        result_fingerprint = fingerprint(solution, len(problem["plaintext"]))
        del solution
        gc.collect()
        return {
            "elapsed_s": elapsed,
            "fingerprint": result_fingerprint[0],
            "ciphertext_bytes": result_fingerprint[1],
            "tag_bytes": result_fingerprint[2],
            "error": None,
        }
    except Exception as exc:
        gc.collect()
        return {
            "elapsed_s": None,
            "fingerprint": None,
            "ciphertext_bytes": None,
            "tag_bytes": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def warm_up() -> None:
    problem: Problem = {
        "key": bytes(range(32)),
        "nonce": bytes(range(12)),
        "plaintext": b"LEXIGEN blind synthetic warm-up" * 128,
        "associated_data": b"blind synthetic warm-up",
    }
    reference_result = execute_once(reference, problem)
    candidate_result = execute_once(solve, problem)
    if reference_result["error"] is not None:
        raise RuntimeError(str(reference_result["error"]))
    if candidate_result["error"] is not None:
        raise RuntimeError(str(candidate_result["error"]))
    if candidate_result["fingerprint"] != reference_result["fingerprint"]:
        raise RuntimeError("selected candidate synthetic warm-up output mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    request = urllib.request.Request(
        f"{BASE}/{MANIFEST}?download=true",
        headers={"User-Agent": "LEXIGEN-v2-blind"},
    )
    raw = urllib.request.urlopen(request, timeout=60).read()
    if git_blob(raw) != EXPECTED_OID:
        raise RuntimeError("blind manifest object mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind rows, received {len(rows)}")

    warm_up()
    selected = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    records: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.parent / f"blind-payload-shard-{args.shard}.bytes"

    for index, row in selected:
        descriptor = row["problem"]["plaintext"]
        if descriptor.get("__type__") != "bytes_ref":
            raise TypeError("expected plaintext bytes_ref descriptor")
        relative = str(descriptor["bin_path"])
        expected_size = int(descriptor["size"])
        download(f"{BASE}/{relative}?download=true", scratch)
        if scratch.stat().st_size != expected_size:
            raise RuntimeError("downloaded blind plaintext size mismatch")
        plaintext = scratch.read_bytes()
        scratch.unlink()
        problem: Problem = {
            "key": decode_inline_bytes(row["problem"]["key"]),
            "nonce": decode_inline_bytes(row["problem"]["nonce"]),
            "plaintext": plaintext,
            "associated_data": decode_inline_bytes(row["problem"]["associated_data"]),
        }

        # Exactly one candidate execution and one reference execution per record.
        if index % 2 == 0:
            reference_result = execute_once(reference, problem)
            candidate_result = execute_once(solve, problem)
            execution_order = "reference_first"
        else:
            candidate_result = execute_once(solve, problem)
            reference_result = execute_once(reference, problem)
            execution_order = "candidate_first"

        if reference_result["error"] is not None or reference_result["elapsed_s"] is None:
            raise RuntimeError(f"reference failed on blind record {index + 1}: {reference_result['error']}")
        reference_s = float(reference_result["elapsed_s"])
        candidate_s = candidate_result["elapsed_s"]
        valid = bool(
            candidate_result["error"] is None
            and candidate_result["fingerprint"] == reference_result["fingerprint"]
            and candidate_result["ciphertext_bytes"] == reference_result["ciphertext_bytes"]
            and candidate_result["tag_bytes"] == reference_result["tag_bytes"]
        )
        speedup = reference_s / float(candidate_s) if candidate_s else 0.0
        record = {
            "index": index + 1,
            "seed": row["seed"],
            "candidate": "native_parallel4",
            "valid": valid,
            "failure_reason": candidate_result["error"] if candidate_result["error"] else (None if valid else "output_mismatch"),
            "candidate_s": candidate_s,
            "reference_s": reference_s,
            "speedup": speedup,
            "plaintext_bytes": len(plaintext),
            "shard": args.shard,
            "execution_order": execution_order,
            "candidate_executions": 1,
        }
        records.append(record)
        print(
            f"[{index + 1}/100] native_parallel4 valid={valid} "
            f"candidate={candidate_s!s}s reference={reference_s:.6f}s speedup={speedup:.3f}",
            flush=True,
        )

        del problem
        del plaintext
        del candidate_result
        del reference_result
        gc.collect()

    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
