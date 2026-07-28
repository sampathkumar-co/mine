from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "kmeans"
API = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=false&expand=false"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def request_bytes(url: str) -> tuple[bytes, int]:
    delays = (0, 5, 15, 30, 60)
    last_error: Exception | None = None
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-task2-metadata"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read(), attempt
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last_error = exc
    raise RuntimeError(f"metadata request exhausted infrastructure retries: {last_error}")


def describe(value: object) -> dict[str, object]:
    if isinstance(value, list):
        result: dict[str, object] = {"encoding": "list", "length": len(value)}
        if value and isinstance(value[0], list):
            result["first_inner_length"] = len(value[0])
        return result
    if isinstance(value, dict):
        result = {"encoding": str(value.get("__type__", "mapping")), "keys": sorted(value)}
        for key in ("shape", "dtype", "npy_path", "bin_path", "size"):
            if key in value:
                result[key] = value[key]
        return result
    return {"encoding": type(value).__name__, "value": value}


def main() -> None:
    api_raw, api_attempts = request_bytes(API)
    entries = json.loads(api_raw)
    if isinstance(entries, dict):
        entries = entries.get("items", entries.get("siblings", []))
    files = [entry for entry in entries if entry.get("type") == "file"]
    train = next(entry for entry in files if str(entry["path"]).endswith("_train.jsonl"))
    test = next(entry for entry in files if str(entry["path"]).endswith("_test.jsonl"))
    train_name = Path(str(train["path"])).name
    test_name = Path(str(test["path"])).name

    raw, train_attempts = request_bytes(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, received {len(rows)}")
    first_problem = rows[0]["problem"]
    report = {
        "dataset_revision": REVISION,
        "task": TASK,
        "directory_request_attempts": api_attempts,
        "training_request_attempts": train_attempts,
        "train_manifest": train_name,
        "train_tree_oid": train.get("oid"),
        "train_lfs": train.get("lfs"),
        "train_size": train.get("size"),
        "train_content_sha256": hashlib.sha256(raw).hexdigest(),
        "training_records": len(rows),
        "first_row_keys": sorted(rows[0]),
        "problem_keys": sorted(first_problem),
        "X_structure": describe(first_problem["X"]),
        "k_structure": describe(first_problem["k"]),
        "seed_min": min(int(row["seed"]) for row in rows),
        "seed_max": max(int(row["seed"]) for row in rows),
        "test_manifest": test_name,
        "test_tree_oid": test.get("oid"),
        "test_lfs": test.get("lfs"),
        "test_size": test.get("size"),
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
    }
    output = Path("metadata-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
