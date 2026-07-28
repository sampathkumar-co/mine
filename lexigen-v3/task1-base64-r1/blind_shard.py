from __future__ import annotations

import argparse
import base64
import gc
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from candidates import Problem, Solution
from selected_solver import solve

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "base64_encoding"
MANIFEST = "base64_encoding_T100ms_n48512_size100_test.jsonl"
EXPECTED_MANIFEST_TREE_OID = "5d093bd50db0d34a682e4f8b6cb492590a7ac750"
TREE_URL = (
    "https://huggingface.co/api/datasets/oripress/AlgoTune/tree/"
    f"{REVISION}/data/{TASK}"
)
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"
SHARDS = 10


def fetch(url: str, *, attempts: int = 8) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LEXIGEN-v3-base64-blind-r1"},
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
                headers={"User-Agent": "LEXIGEN-v3-base64-blind-r1"},
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
    reference_result = reference(problem)["encoded_data"]
    candidate_result = solve(problem)["encoded_data"]
    if candidate_result != reference_result:
        raise RuntimeError("selected Base64 blind solver failed synthetic warm-up")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < SHARDS:
        raise ValueError(f"shard must be in [0, {SHARDS})")

    entries = json.loads(fetch(TREE_URL))
    matches = [
        entry
        for entry in entries
        if entry.get("type") == "file" and Path(entry["path"]).name == MANIFEST
    ]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one official test manifest entry")
    if matches[0].get("oid") != EXPECTED_MANIFEST_TREE_OID:
        raise RuntimeError("official test manifest object identity mismatch")

    manifest_raw = fetch(f"{BASE}/{MANIFEST}?download=true")
    rows = [json.loads(line) for line in manifest_raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 blind rows, received {len(rows)}")

    warm_up()
    selected_rows = [(index, row) for index, row in enumerate(rows) if index % SHARDS == args.shard]
    records: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.parent / f"blind-payload-shard-{args.shard}.bytes"

    for index, row in selected_rows:
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
        problem: Problem = {"plaintext": plaintext}

        if index % 2 == 0:
            reference_result = execute(reference, problem)
            candidate_result = execute(solve, problem)
            reference_order = "first"
        else:
            candidate_result = execute(solve, problem)
            reference_result = execute(reference, problem)
            reference_order = "last"

        if reference_result["error"] is not None or reference_result["elapsed_s"] is None:
            raise RuntimeError(f"reference failed on blind record {index + 1}: {reference_result['error']}")
        reference_encoded = reference_result["encoded"]
        candidate_encoded = candidate_result["encoded"]
        candidate_seconds = candidate_result["elapsed_s"]
        reference_seconds = float(reference_result["elapsed_s"])
        valid = bool(
            candidate_result["error"] is None
            and isinstance(reference_encoded, bytes)
            and isinstance(candidate_encoded, bytes)
            and candidate_encoded == reference_encoded
        )
        speedup = reference_seconds / float(candidate_seconds) if candidate_seconds else 0.0
        record = {
            "index": index + 1,
            "seed": row.get("seed"),
            "candidate": "pybase64_simd",
            "valid": valid,
            "failure_reason": candidate_result["error"] if candidate_result["error"] else (None if valid else "output_mismatch"),
            "candidate_s": candidate_seconds,
            "reference_s": reference_seconds,
            "speedup": speedup,
            "plaintext_bytes": len(plaintext),
            "encoded_bytes": len(reference_encoded) if isinstance(reference_encoded, bytes) else None,
            "shard": args.shard,
            "reference_order": reference_order,
        }
        records.append(record)
        print(
            f"[{index + 1}/100] pybase64_simd valid={valid} candidate={candidate_seconds!s}s "
            f"reference={reference_seconds:.6f}s speedup={speedup:.3f}",
            flush=True,
        )
        del candidate_result, reference_result, candidate_encoded, reference_encoded, problem, plaintext
        gc.collect()

    args.output.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
