from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "max_flow_min_cost"
URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=false&expand=false&limit=1000"


def main() -> None:
    req = urllib.request.Request(URL, headers={"User-Agent":"LEXIGEN-v6-task1-metadata-r1","Accept":"application/json"})
    with urllib.request.urlopen(req, timeout=240) as response:
        raw = response.read()
        etag = response.headers.get("etag")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise RuntimeError("dataset directory metadata is not a list")
    files = []
    for row in payload:
        if not isinstance(row, dict) or row.get("type") != "file":
            continue
        path = str(row.get("path", ""))
        name = path.rsplit("/", 1)[-1]
        files.append({"path":path,"name":name,"size":int(row.get("size",0)),"oid":str(row.get("oid",""))})
    trains = [r for r in files if "_T100ms_" in r["name"] and "_size100_train.jsonl" in r["name"]]
    tests = [r for r in files if "_T100ms_" in r["name"] and "_size100_test.jsonl" in r["name"]]
    pairs = []
    for tr in trains:
        stem = tr["name"].replace("_train.jsonl", "")
        for te in tests:
            if te["name"].replace("_test.jsonl", "") == stem:
                pairs.append((tr, te))
    if len(pairs) != 1:
        raise RuntimeError(f"expected exactly one T100ms size100 train/test pair, got {len(pairs)}")
    train, test = pairs[0]
    result = {
        "campaign":"LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication",
        "task_index":1,
        "task":TASK,
        "stage":"dataset_metadata_r1",
        "dataset_revision":REVISION,
        "directory_metadata_url":URL,
        "directory_response_sha256":hashlib.sha256(raw).hexdigest(),
        "directory_etag":etag,
        "file_count":len(files),
        "train":train,
        "test":test,
        "train_manifest_opened":False,
        "test_manifest_opened":False,
        "payloads_opened":0,
        "reports_opened":False,
    }
    Path("dataset-metadata.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
