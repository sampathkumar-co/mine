from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "sha256_hashing"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    last_error: Exception | None = None
    for delay in (0, 3, 10, 30, 60):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-sha256-metadata"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last_error = exc
    raise RuntimeError(f"metadata fetch failed: {last_error}")


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def byte_descriptor_size(value: Any) -> tuple[str, int, str | None]:
    if isinstance(value, dict):
        descriptor_type = str(value.get("__type__"))
        if descriptor_type == "bytes_ref":
            return descriptor_type, int(value["size"]), str(value["bin_path"])
        if descriptor_type == "bytes":
            decoded = base64.b64decode(str(value["data_b64"]), validate=True)
            return descriptor_type, len(decoded), None
    if isinstance(value, str):
        return "string", len(value.encode()), None
    raise TypeError(f"unsupported plaintext descriptor: {type(value).__name__}")


def main() -> None:
    entries = json.loads(fetch(TREE_URL))
    files = [entry for entry in entries if entry.get("type") == "file"]
    train = [entry for entry in files if str(entry["path"]).endswith("_train.jsonl")]
    test = [entry for entry in files if str(entry["path"]).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError("expected one training and one test manifest")
    train_entry, test_entry = train[0], test[0]
    train_name = Path(train_entry["path"]).name
    test_name = Path(test_entry["path"]).name
    raw = fetch(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, received {len(rows)}")

    descriptor_types: set[str] = set()
    sizes: list[int] = []
    paths: list[str] = []
    for row in rows:
        problem = row["problem"]
        if sorted(problem) != ["plaintext"]:
            raise RuntimeError(f"unexpected problem keys: {sorted(problem)}")
        descriptor_type, size, relative = byte_descriptor_size(problem["plaintext"])
        descriptor_types.add(descriptor_type)
        sizes.append(size)
        if relative is not None:
            paths.append(relative)

    report = {
        "task": TASK,
        "dataset_revision": REVISION,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_entry.get("oid"),
        "train_manifest_git_blob_sha1": git_blob(raw),
        "train_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "test_manifest_name": test_name,
        "test_manifest_tree_oid": test_entry.get("oid"),
        "training_records": len(rows),
        "problem_keys": ["plaintext"],
        "plaintext_descriptor_types": sorted(descriptor_types),
        "plaintext_size_minimum": min(sizes),
        "plaintext_size_maximum": max(sizes),
        "plaintext_size_values": sorted(set(sizes)),
        "referenced_payload_count": len(paths),
        "training_payloads_downloaded": 0,
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
    }
    Path("metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
