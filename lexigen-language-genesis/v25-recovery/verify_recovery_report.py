from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
EXPECTED_SHA256 = "4d6b326e3f8334aa5d5542cae59344a76a1baa55b1734ddacd3a70c783440091"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    original = V25 / "V25_FIRST_DISCOVERY_REPORT.json"
    if sha256(original) != EXPECTED_SHA256:
        raise RuntimeError("frozen original report changed")
    if args.report.read_bytes() != original.read_bytes():
        raise RuntimeError("recovery report is not byte-identical")
    print(f"BYTE_IDENTICAL {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
