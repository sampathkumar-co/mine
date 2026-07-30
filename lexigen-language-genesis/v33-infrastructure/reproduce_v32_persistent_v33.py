from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V32 = HERE.parent / "v32"
for path in (HERE, V32):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scan_one_v32 as frozen_scanner
from persistent_generator_client_v33 import PersistentGeneratorClient

EXPECTED_TASK = "9caf5b84"
EXPECTED_REPORT_SHA256 = "ab8c1411484ac4cb7516d3a4ec9aa808f5fe7d1f3b2029be923dc78d29ec6f06"
REFERENCE = V32 / "V32_SUCCESS_9caf5b84.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(REFERENCE) != EXPECTED_REPORT_SHA256:
        raise RuntimeError("frozen v32 reference report hash mismatch")
    client = PersistentGeneratorClient(
        args.arcgen_root,
        EXPECTED_TASK,
        maximum_same_seed_retries=1,
    )
    original_generate_case = frozen_scanner.generate_case
    original_argv = sys.argv[:]
    try:
        frozen_scanner.generate_case = client.generate_case
        sys.argv = [
            "scan_one_v32.py",
            "--arcgen-root",
            str(args.arcgen_root),
            "--task-id",
            EXPECTED_TASK,
            "--output",
            str(args.output),
        ]
        frozen_scanner.main()
    finally:
        frozen_scanner.generate_case = original_generate_case
        sys.argv = original_argv
        client.close()

    actual_hash = sha256_file(args.output)
    byte_identical = args.output.read_bytes() == REFERENCE.read_bytes()
    audit = {
        "schema": "lexigen-v33-persistent-generator-equivalence-v1",
        "task_id": EXPECTED_TASK,
        "expected_report_sha256": EXPECTED_REPORT_SHA256,
        "actual_report_sha256": actual_hash,
        "byte_identical": byte_identical,
        "worker_restart_count": client.restart_count,
        "same_seed_retry_count": client.same_seed_retry_count,
        "protocol_error_count": client.protocol_error_count,
        "future_use_authorized": (
            byte_identical
            and actual_hash == EXPECTED_REPORT_SHA256
            and client.restart_count == 0
            and client.same_seed_retry_count == 0
            and client.protocol_error_count == 0
        ),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_bytes(
        (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(audit, sort_keys=True))
    if not audit["future_use_authorized"]:
        raise SystemExit("persistent generator equivalence gate failed")


if __name__ == "__main__":
    main()
