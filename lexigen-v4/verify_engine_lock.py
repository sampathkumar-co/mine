from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path(__file__).resolve().parent / "ENGINE_LOCK.json"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    expected_files = lock.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeError("ENGINE_LOCK contains no files")
    verified: dict[str, str] = {}
    for relative, expected in sorted(expected_files.items()):
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"locked file missing: {relative}")
        actual = git_blob(path.read_bytes())
        if actual != expected:
            raise RuntimeError(f"locked file mismatch: {relative}: {actual} != {expected}")
        verified[relative] = actual
    if lock.get("holdout_inventory_accessed_before_lock") is not False:
        raise RuntimeError("lock does not certify zero prelock inventory access")
    if lock.get("task_contents_accessed_before_lock") is not False:
        raise RuntimeError("lock does not certify zero prelock task-content access")
    artifact = lock.get("validation_artifact")
    if not isinstance(artifact, dict) or not artifact.get("digest", "").startswith("sha256:"):
        raise RuntimeError("lock lacks a SHA-256 validation artifact digest")
    print(json.dumps({"lock_status": "verified", "validated_engine_commit": lock.get("validated_engine_commit"), "verified_files": verified}, indent=2))


if __name__ == "__main__":
    main()
