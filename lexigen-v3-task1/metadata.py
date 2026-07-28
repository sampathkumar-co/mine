from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "bb02811fa47ca1c833baaa344949bcd8fb307ac8"
TASK = "cvar_projection"
API = f"https://huggingface.co/api/datasets/oripress/AlgoTune/tree/{REVISION}/data/{TASK}?recursive=false&expand=false"
BASE = f"https://huggingface.co/datasets/oripress/AlgoTune/resolve/{REVISION}/data/{TASK}"


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v3-task1-metadata"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def main() -> None:
    entries = json.loads(request_bytes(API))
    files = [entry for entry in entries if entry.get("type") == "file"]
    train = next(entry for entry in files if str(entry["path"]).endswith("_train.jsonl"))
    test = next(entry for entry in files if str(entry["path"]).endswith("_test.jsonl"))
    train_name = Path(str(train["path"])).name
    test_name = Path(str(test["path"])).name

    raw = request_bytes(f"{BASE}/{train_name}?download=true")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 training rows, received {len(rows)}")
    first_problem = rows[0]["problem"]
    x0 = first_problem["x0"]
    scenarios = first_problem["loss_scenarios"]
    report = {
        "dataset_revision": REVISION,
        "task": TASK,
        "train_manifest": train_name,
        "train_tree_oid": train.get("oid"),
        "train_lfs": train.get("lfs"),
        "train_size": train.get("size"),
        "train_content_sha256": hashlib.sha256(raw).hexdigest(),
        "training_records": len(rows),
        "first_row_keys": sorted(rows[0]),
        "problem_keys": sorted(first_problem),
        "n_dims": len(x0),
        "n_scenarios": len(scenarios),
        "scenario_width": len(scenarios[0]),
        "beta": first_problem["beta"],
        "kappa": first_problem["kappa"],
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
