from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "kmeans"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-kmeans-metadata"})
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code != 429:
                raise
        except Exception as exc:
            last = exc
        time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"metadata fetch failed: {url}") from last


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> None:
    entries = json.loads(fetch(TREE_URL))
    files = [entry for entry in entries if entry.get("type") == "file"]
    train = [entry for entry in files if str(entry["path"]).endswith("_train.jsonl")]
    test = [entry for entry in files if str(entry["path"]).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError("expected one train and one test manifest")
    train_entry, test_entry = train[0], test[0]
    train_name = Path(train_entry["path"]).name
    test_name = Path(test_entry["path"]).name
    raw = fetch(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")
    shapes = []
    clusters = []
    for row in rows:
        problem = row["problem"]
        x = problem["X"]
        if not isinstance(x, list) or not x or not isinstance(x[0], list):
            raise RuntimeError("expected inline two-dimensional X")
        shapes.append([len(x), len(x[0])])
        clusters.append(int(problem["k"]))
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
        "sample_count_min": min(shape[0] for shape in shapes),
        "sample_count_max": max(shape[0] for shape in shapes),
        "dimension_values": sorted({shape[1] for shape in shapes}),
        "cluster_count_min": min(clusters),
        "cluster_count_max": max(clusters),
        "samples_per_cluster_values": sorted({shape[0] // k for shape, k in zip(shapes, clusters)}),
        "test_manifest_downloaded": False,
        "test_payloads_downloaded": 0,
    }
    Path("metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
