from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "spectral_clustering"
API = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    url = API + "?" + urllib.parse.urlencode({"recursive": "false", "expand": "true"})
    req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v5-task3-metadata-only-r1"})
    with urllib.request.urlopen(req, timeout=180) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Hugging Face tree metadata response is not a list")

    rows: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError("unexpected metadata row")
        path = str(item.get("path", ""))
        if not path.startswith(f"data/{TASK}/"):
            raise RuntimeError(f"unexpected path outside Task 3 dataset directory: {path}")
        # Metadata-only boundary: never follow resolve/download URLs and never open any file.
        rows.append({
            "path": path,
            "type": item.get("type"),
            "size": item.get("size"),
            "oid": item.get("oid"),
            "lfs": item.get("lfs"),
            "lastCommit": item.get("lastCommit"),
        })

    jsonls = sorted(
        (row for row in rows if str(row["path"]).endswith(".jsonl")),
        key=lambda row: str(row["path"]),
    )
    train = [row for row in jsonls if str(row["path"]).endswith("_train.jsonl")]
    test = [row for row in jsonls if str(row["path"]).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError(f"expected exactly one train and one test JSONL; got train={len(train)} test={len(test)}")
    if "_T100ms_" not in str(train[0]["path"]) or "_size100_" not in str(train[0]["path"]):
        raise RuntimeError("training manifest does not match frozen T100ms/size100 benchmark shape")
    if str(train[0]["path"]).replace("_train.jsonl", "_test.jsonl") != str(test[0]["path"]):
        raise RuntimeError("train/test manifest stems differ")

    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 3,
        "task": TASK,
        "stage": "dataset_directory_metadata_r1",
        "dataset_revision": REVISION,
        "metadata_api": API,
        "directory_entry_count": len(rows),
        "jsonl_entries": jsonls,
        "training_manifest_metadata": train[0],
        "test_manifest_metadata": test[0],
        "manifest_contents_opened": False,
        "payload_contents_opened": False,
        "resolve_or_download_urls_followed": False,
        "metadata_fields_preserved": ["path", "type", "size", "oid", "lfs", "lastCommit"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
