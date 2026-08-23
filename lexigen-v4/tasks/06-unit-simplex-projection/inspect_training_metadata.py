from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "unit_simplex_projection"
TREE_URL = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=true&expand=false"
RAW_BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-task6-metadata-r1"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def main() -> None:
    tree = json.loads(fetch(TREE_URL))
    files = [x for x in tree if x.get("type") == "file"]
    train = [x for x in files if str(x.get("path", "")).endswith("_train.jsonl")]
    test = [x for x in files if str(x.get("path", "")).endswith("_test.jsonl")]
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError(f"expected exactly one train/test manifest, got train={len(train)} test={len(test)}")
    train_item, test_item = train[0], test[0]
    train_name = str(train_item["path"]).split("/")[-1]
    test_name = str(test_item["path"]).split("/")[-1]

    raw = fetch(f"{RAW_BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training records, got {len(rows)}")

    lengths: dict[str, int] = {}
    encodings: dict[str, int] = {}
    scalar_categories: dict[str, int] = {}
    for row in rows:
        problem = row.get("problem")
        if not isinstance(problem, dict) or "y" not in problem:
            raise RuntimeError("unexpected training problem schema")
        y = problem["y"]
        if not isinstance(y, list):
            raise RuntimeError("Task 6 metadata inspector permits only inline y lists")
        lengths[str(len(y))] = lengths.get(str(len(y)), 0) + 1
        encodings["inline_json_list"] = encodings.get("inline_json_list", 0) + 1
        category = "json_number" if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in y) else "other"
        scalar_categories[category] = scalar_categories.get(category, 0) + 1

    report = {
        "campaign": "LEXIGEN v4 Frozen Generalization Experiment",
        "task_index": 6,
        "task": TASK,
        "stage": "metadata_revision1",
        "dataset_revision": REVISION,
        "train_manifest_name": train_name,
        "train_manifest_tree_oid": train_item.get("oid"),
        "train_manifest_git_blob_sha1": train_item.get("oid"),
        "train_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "train_manifest_size": len(raw),
        "test_manifest_name": test_name,
        "test_manifest_tree_oid": test_item.get("oid"),
        "training_records": len(rows),
        "vector_length_counts": lengths,
        "vector_encoding_counts": encodings,
        "vector_scalar_category_counts": scalar_categories,
        "training_manifest_contains_inline_payload": True,
        "training_numerical_payload_downloaded_due_inline_manifest": True,
        "training_values_statistically_summarized": False,
        "training_values_used_for_candidate_design_or_adaptation": False,
        "candidate_family_changed_after_metadata": False,
        "test_manifest_downloaded": False,
        "test_payloads_opened": 0,
        "candidate_execution_count": 0,
        "reference_execution_count": 0,
        "training_revision_consumed": False,
    }
    out = Path("metadata-evidence")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
