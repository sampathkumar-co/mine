from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path(__file__).resolve().parent / "SELECTION_LOCK.json"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    result_meta = lock["selection_result"]
    result_path = ROOT / result_meta["path"]
    raw = result_path.read_bytes()
    actual_blob = git_blob(raw)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_blob != result_meta["git_blob_sha1"]:
        raise RuntimeError(f"selection result Git blob mismatch: {actual_blob}")
    if actual_sha256 != result_meta["sha256"]:
        raise RuntimeError(f"selection result SHA-256 mismatch: {actual_sha256}")
    result = json.loads(raw)
    selected = [row["task"] for row in result["selected"]]
    if selected != lock["selected_tasks"]:
        raise RuntimeError("selected task order differs from SELECTION_LOCK")
    if result["inventory_sha256"] != lock["inventory_sha256"]:
        raise RuntimeError("inventory hash differs from SELECTION_LOCK")
    if lock.get("task_source_contents_opened_before_lock") is not False:
        raise RuntimeError("selection lock does not certify zero task-source access")
    print(json.dumps({"selection_lock": "verified", "selection_run_id": lock["selection_run_id"], "selected_tasks": selected}, indent=2))


if __name__ == "__main__":
    main()
