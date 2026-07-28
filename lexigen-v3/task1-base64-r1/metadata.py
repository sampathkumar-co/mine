from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "base64_encoding"
TREE_URL = (
    "https://huggingface.co/api/datasets/oripress/AlgoTune/tree/"
    f"{REVISION}/data/{TASK}"
)
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str, *, attempts: int = 8) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LEXIGEN-v3-metadata-only"},
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
    raise RuntimeError(f"metadata fetch exhausted retries: {url}") from last_error


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def descriptor(value):
    if not isinstance(value, dict):
        return {"python_type": type(value).__name__}
    result = {"descriptor_type": value.get("__type__")}
    for key in ("size", "bin_path", "npy_path", "shape", "dtype"):
        if key in value:
            result[key] = value[key]
    return result


def main() -> None:
    entries = json.loads(fetch(TREE_URL))
    files = [entry for entry in entries if entry.get("type") == "file"]
    train_entries = [entry for entry in files if entry["path"].endswith("_train.jsonl")]
    test_entries = [entry for entry in files if entry["path"].endswith("_test.jsonl")]
    if len(train_entries) != 1 or len(test_entries) != 1:
        raise RuntimeError("expected exactly one train and one test manifest")
    train_entry = train_entries[0]
    test_entry = test_entries[0]
    train_name = Path(train_entry["path"]).name
    test_name = Path(test_entry["path"]).name

    train_raw = fetch(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in train_raw.decode().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("training manifest is empty")
    first = rows[0]
    problem = first.get("problem")
    if not isinstance(problem, dict):
        raise RuntimeError("expected mapping problem")

    report = {
        "task": TASK,
        "dataset_revision": REVISION,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_entry.get("oid"),
        "train_manifest_lfs": train_entry.get("lfs"),
        "train_manifest_resolved_git_blob": git_blob(train_raw),
        "train_manifest_resolved_sha256": hashlib.sha256(train_raw).hexdigest(),
        "test_manifest_name": test_name,
        "test_manifest_tree_oid": test_entry.get("oid"),
        "test_manifest_lfs": test_entry.get("lfs"),
        "training_records": len(rows),
        "row_keys": sorted(first),
        "problem_keys": sorted(problem),
        "problem_descriptors": {key: descriptor(value) for key, value in problem.items()},
        "k_values": sorted({row.get("k") for row in rows}, key=lambda value: str(value)),
        "seed_min": min(row.get("seed") for row in rows),
        "seed_max": max(row.get("seed") for row in rows),
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
    }
    Path("metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
