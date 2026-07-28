from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes

from candidates import CANDIDATES


def reference(data: bytes) -> bytes:
    digest = hashes.Hash(hashes.SHA256())
    digest.update(data)
    return digest.finalize()


def main() -> None:
    cases = [
        ("empty", b""),
        ("one_byte", b"a"),
        ("block_minus_one", bytes(range(63))),
        ("one_block", bytes(range(64))),
        ("block_plus_one", bytes(range(65))),
        ("one_megabyte", bytes(range(256)) * 4096),
        ("sixty_four_megabytes", bytes(range(256)) * 262144),
    ]
    rows: list[dict[str, object]] = []
    for name, data in cases:
        start = time.perf_counter()
        expected = reference(data)
        reference_s = time.perf_counter() - start
        for candidate_name, candidate in CANDIDATES.items():
            start = time.perf_counter()
            solution = candidate({"plaintext": data})
            candidate_s = time.perf_counter() - start
            actual = solution.get("digest")
            valid = isinstance(actual, bytes) and actual == expected and len(actual) == 32
            speedup = reference_s / candidate_s if candidate_s > 0.0 else 0.0
            rows.append({
                "case": name,
                "bytes": len(data),
                "candidate": candidate_name,
                "valid": valid,
                "candidate_s": candidate_s,
                "reference_s": reference_s,
                "speedup": speedup,
                "digest_hex": actual.hex() if isinstance(actual, bytes) else None,
            })
            print(
                f"{name} {candidate_name} valid={valid} speedup={speedup:.3f}",
                flush=True,
            )

    summaries = []
    for candidate_name in sorted(CANDIDATES):
        selected = [row for row in rows if row["candidate"] == candidate_name]
        large = next(row for row in selected if row["case"] == "sixty_four_megabytes")
        summaries.append({
            "candidate": candidate_name,
            "valid_cases": sum(bool(row["valid"]) for row in selected),
            "cases": len(selected),
            "large_payload_speedup": large["speedup"],
        })
    report = {
        "task": "sha256_hashing",
        "candidate_revision": 1,
        "scope": "synthetic_only_no_benchmark_access",
        "rows": rows,
        "summaries": summaries,
        "passes_exactness_gate": all(bool(row["valid"]) for row in rows),
        "benchmark_data_accessed": False,
        "synthetic_speed_is_diagnostic_only": True,
        "python_hashlib_name": hashlib.sha256().name,
    }
    Path("synthetic-result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passes_exactness_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
