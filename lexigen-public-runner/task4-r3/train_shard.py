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

from candidates import CANDIDATES, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "chacha_encryption"
MANIFEST = "chacha_encryption_T100ms_n197380_size100_train.jsonl"
EXPECTED_OID = "9eb2847936888c7843b54dc94dd8e225a4fc5850"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
TAG_SIZE = 16
SHARDS = 10


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v2-train-r3"})
            with urllib.request.urlopen(request, timeout=180) as response:
                with destination.open("wb") as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            return
        except Exception as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed after three infrastructure attempts: {url}") from last_error


def decode_inline_bytes(value: dict[str, object]) -> bytes:
    if value.get("__type__") != "bytes":
        raise TypeError(f"expected inline bytes descriptor, received {value.get('__type__')!r}")
    return base64.b64decode(str(value["data_b64"]), validate=True)


def reference(problem: Problem) -> Solution:
    combined = ChaCha20Poly1305(problem["key"]).encrypt(
        problem["nonce"], problem["plaintext"], problem["associated_data"]
    )
    return {"ciphertext": combined[:-TAG_SIZE], "tag": combined[-TAG_SIZE:]}


def fingerprint(solution: Solution, plaintext_size: int) -> tuple[str, int, int]:
    ciphertext = solution.get("ciphertext")
    tag = solution.get("tag")
    if not isinstance(ciphertext, bytes) or not isinstance(tag, bytes):
        raise TypeError("solution values must be bytes")
    if len(ciphertext) != plaintext_size or len(tag) != TAG_SIZE:
        raise ValueError("solution length mismatch")
    digest = hashlib.sha256()
    digest.update(len(ciphertext).to_bytes(8, "big"))
    digest.update(ciphertext)
    digest.update(len(tag).to_bytes(8, "big"))
    digest.update(tag)
    return digest.hexdigest(), len(ciphertext), len(tag)


def execute(fn: Callable[[Problem], Solution], problem: Problem) -> dict[str, object]:
    try:
        started = time.perf_counter()
        solution = fn(problem)
        elapsed = time.perf_counter() - started
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
        "plaintext": b"LEXIGEN revision 3 warm-up" * 64,
        "associated_data": b"synthetic-only warm-up",
    }
    reference_result = execute(reference, problem)
    if reference_result["error"] is not None:
        raise RuntimeError(str(reference_result["error"]))
    for name, candidate in CANDIDATES.items():
        result = execute(candidate, problem)
        if result["error"] is not None or result["fingerprint"] != reference_result["fingerprint"]:
            raise RuntimeError(f"{name} warm-up failed or mismatched")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    request = urllib.request.Request(
        f"{BASE}/{MANIFEST}?download=true",
        headers={"User-Agent": "LEXIGEN-v2-train-r3"},
    )
    raw = urllib.request.urlopen(request, timeout=60).read()
    if git_blob(raw) != EXPECTED_OID:
        raise RuntimeError("training manifest object mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")

    warm_up()
    selected = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    records: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.parent / f"payload-shard-{args.shard}.bytes"

    for index, row in selected:
        descriptor = row["problem"]["plaintext"]
        if descriptor.get("__type__") != "bytes_ref":
            raise TypeError("expected plaintext bytes_ref descriptor")
        relative = str(descriptor["bin_path"])
        expected_size = int(descriptor["size"])
        download(f"{BASE}/{relative}?download=true", scratch)
        if scratch.stat().st_size != expected_size:
            raise RuntimeError("downloaded plaintext size mismatch")
        plaintext = scratch.read_bytes()
        scratch.unlink()
        problem: Problem = {
            "key": decode_inline_bytes(row["problem"]["key"]),
            "nonce": decode_inline_bytes(row["problem"]["nonce"]),
            "plaintext": plaintext,
            "associated_data": decode_inline_bytes(row["problem"]["associated_data"]),
        }

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
        reference_seconds = float(reference_result["elapsed_s"])
        for name in names:
            candidate_result = candidate_results[name]
            candidate_seconds = candidate_result["elapsed_s"]
            valid = bool(
                candidate_result["error"] is None
                and candidate_result["fingerprint"] == reference_result["fingerprint"]
                and candidate_result["ciphertext_bytes"] == reference_result["ciphertext_bytes"]
                and candidate_result["tag_bytes"] == reference_result["tag_bytes"]
            )
            speedup = reference_seconds / float(candidate_seconds) if candidate_seconds else 0.0
            records.append({
                "index": index + 1,
                "seed": row["seed"],
                "candidate": name,
                "valid": valid,
                "failure_reason": candidate_result["error"] if candidate_result["error"] else (None if valid else "output_mismatch"),
                "candidate_s": candidate_seconds,
                "reference_s": reference_seconds,
                "speedup": speedup,
                "plaintext_bytes": len(plaintext),
                "shard": args.shard,
                "reference_order": "first" if index % 2 == 0 else "last",
            })
            print(
                f"[{index + 1}/100] {name} valid={valid} candidate={candidate_seconds!s}s "
                f"reference={reference_seconds:.6f}s speedup={speedup:.3f}",
                flush=True,
            )

        del problem, plaintext, candidate_results
        gc.collect()

    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
