from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from candidates import CANDIDATES, Problem, Solution

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "base64_encoding"
MANIFEST = "base64_encoding_T100ms_n48512_size100_train.jsonl"
EXPECTED_MANIFEST_SHA256 = "bf5d9bfb366bde6740c662491e39ee9d93ab2d8cea26967fd71788e9baece63a"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10


def fetch(url: str, *, attempts: int = 8) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LEXIGEN-v3-base64-train-r1"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
        except Exception as exc:
            last_error = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"download exhausted retries: {url}") from last_error


def download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LEXIGEN-v3-base64-train-r1"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                with destination.open("wb") as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
        except Exception as exc:
            last_error = exc
        if destination.exists():
            destination.unlink()
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"payload download exhausted retries: {url}") from last_error


def reference(problem: Problem) -> Solution:
    return {"encoded_data": base64.b64encode(problem["plaintext"])}


def execute(function: Callable[[Problem], Solution], problem: Problem) -> dict[str, object]:
    try:
        started = time.perf_counter()
        solution = function(problem)
        elapsed = time.perf_counter() - started
        encoded = solution.get("encoded_data")
        if not isinstance(encoded, bytes):
            raise TypeError("encoded_data must be bytes")
        return {
            "elapsed_s": elapsed,
            "encoded": encoded,
            "error": None,
        }
    except Exception as exc:
        return {
            "elapsed_s": None,
            "encoded": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def warm_up() -> None:
    problem: Problem = {"plaintext": bytes(range(251)) * 4096}
    expected = reference(problem)["encoded_data"]
    for name, candidate in CANDIDATES.items():
        result = candidate(problem)
        if result["encoded_data"] != expected:
            raise RuntimeError(f"{name} failed synthetic warm-up")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    manifest_raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    if hashlib.sha256(manifest_raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("training manifest SHA-256 mismatch")
    rows = [json.loads(line) for line in manifest_raw.decode().splitlines() if line.strip()]
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
        problem: Problem = {"plaintext": plaintext}

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
        reference_encoded = reference_result["encoded"]
        if not isinstance(reference_encoded, bytes):
            raise RuntimeError("reference did not return bytes")
        reference_seconds = float(reference_result["elapsed_s"])

        for name in names:
            candidate_result = candidate_results[name]
            candidate_encoded = candidate_result["encoded"]
            candidate_seconds = candidate_result["elapsed_s"]
            valid = bool(
                candidate_result["error"] is None
                and isinstance(candidate_encoded, bytes)
                and candidate_encoded == reference_encoded
            )
            speedup = reference_seconds / float(candidate_seconds) if candidate_seconds else 0.0
            records.append({
                "index": index + 1,
                "seed": row.get("seed"),
                "candidate": name,
                "valid": valid,
                "failure_reason": candidate_result["error"] if candidate_result["error"] else (None if valid else "output_mismatch"),
                "candidate_s": candidate_seconds,
                "reference_s": reference_seconds,
                "speedup": speedup,
                "plaintext_bytes": len(plaintext),
                "encoded_bytes": len(reference_encoded),
                "shard": args.shard,
                "reference_order": "first" if index % 2 == 0 else "last",
            })
            print(
                f"[{index + 1}/100] {name} valid={valid} candidate={candidate_seconds!s}s "
                f"reference={reference_seconds:.6f}s speedup={speedup:.3f}",
                flush=True,
            )

        del candidate_results, reference_result, reference_encoded, problem, plaintext
        gc.collect()

    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
