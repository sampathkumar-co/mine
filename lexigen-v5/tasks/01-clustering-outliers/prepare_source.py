from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

from engine import LEARNED_SIGNATURES, generate_proposals, verify_transfer_memory

SOURCE_COMMIT = "dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"
TASK = "clustering_outliers"
RAW_BASE = f"https://raw.githubusercontent.com/oripress/AlgoTune/{SOURCE_COMMIT}"


def fetch(relative_path: str, *, optional: bool = False) -> bytes | None:
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise RuntimeError(f"unsafe source path: {relative_path}")
    req = urllib.request.Request(f"{RAW_BASE}/{relative_path}", headers={"User-Agent": "LEXIGEN-v5-task1-source-r1"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if optional and exc.code == 404:
            return None
        raise


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = json.loads(args.task_start.read_text(encoding="utf-8"))
    if start["task"] != TASK or int(start["task_index"]) != 1:
        raise RuntimeError("Task 1 identity mismatch")
    if start["source_commit"] != SOURCE_COMMIT:
        raise RuntimeError("source commit mismatch")
    forbidden = (
        start["task_source_opened_before_task_start"],
        start["task_description_opened_before_task_start"],
        start["training_manifest_opened_before_task_start"],
        start["training_payloads_opened_before_task_start"],
        start["test_manifest_opened_before_task_start"],
        start["test_payloads_opened_before_task_start"],
        start["reports_opened"],
        start["public_solvers_opened"],
    )
    if any(forbidden):
        raise RuntimeError("Task 1 staged boundary was not clean before source access")

    memory = verify_transfer_memory()
    source_path = str(start["expected_source_path"])
    description_path = str(start["expected_description_path"])
    source_raw = fetch(source_path)
    if source_raw is None:
        raise RuntimeError("required Task 1 source is missing")
    description_raw = fetch(description_path, optional=True)
    source_text = source_raw.decode("utf-8")
    proposals = generate_proposals(source_text)

    arms = proposals["arms"]
    expected_arms = {"v5_full", "v5_no_transfer", "random_search", "static_template", "v4_compatible"}
    if set(arms) != expected_arms:
        raise RuntimeError(f"arm mismatch: {set(arms)}")
    if any(len(rows) > 6 for rows in arms.values()):
        raise RuntimeError("proposal budget exceeded")

    learned_signatures = {tuple(value) for value in LEARNED_SIGNATURES.values()}
    for row in arms["v5_no_transfer"]:
        if row["transfer_ids"] or row["learned_template"] is not None:
            raise RuntimeError("no-transfer arm contains learned transfer metadata")
        if tuple(row["operators"]) in learned_signatures:
            raise RuntimeError("no-transfer arm contains an exact learned recipe signature")

    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 1,
        "task": TASK,
        "family": start["family"],
        "stage": "source_analysis_r1",
        "source_commit": SOURCE_COMMIT,
        "source_path": source_path,
        "source_git_blob_sha1": git_blob_sha1(source_raw),
        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "description_path": description_path,
        "description_present": description_raw is not None,
        "description_git_blob_sha1": git_blob_sha1(description_raw) if description_raw is not None else None,
        "description_sha256": hashlib.sha256(description_raw).hexdigest() if description_raw is not None else None,
        "transfer_memory": memory,
        "engine_output": proposals,
        "official_training_manifest_opened": False,
        "official_training_payloads_opened": 0,
        "official_test_manifest_opened": False,
        "official_test_payloads_opened": 0,
        "reports_opened": False,
        "public_solvers_opened": False,
        "human_task_specific_solver_design": False,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source-analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "task-source.py").write_bytes(source_raw)
    if description_raw is not None:
        (args.output / "description.txt").write_bytes(description_raw)
    print(json.dumps({
        "task": TASK,
        "source_sha256": report["source_sha256"],
        "features": proposals["fingerprint"]["features"],
        "applicable_transfer_templates": proposals["applicable_transfer_templates"],
        "proposal_counts": {arm: len(rows) for arm, rows in arms.items()},
        "top_proposals": {arm: rows[0] if rows else None for arm, rows in arms.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
