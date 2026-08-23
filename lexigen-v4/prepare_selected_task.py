from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import asdict
from pathlib import Path

from engine_v4 import ENGINE_VERSION, fingerprint, generate_proposals

ROOT = Path(__file__).resolve().parent
SOURCE_REPOSITORY = "oripress/AlgoTune"
ARMS = ("v4_full", "v4_no_transfer", "random_search", "template_synthesis", "v3_compatible")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LEXIGEN-v4-frozen-task-source-analysis"})
    with urllib.request.urlopen(request, timeout=240) as response:
        return response.read()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_start = json.loads(args.task_start.read_text(encoding="utf-8"))
    if task_start.get("task_source_opened") is not False:
        raise RuntimeError("TASK_START does not certify unopened source")
    if task_start.get("training_manifest_opened") is not False or task_start.get("test_manifest_opened") is not False:
        raise RuntimeError("TASK_START data boundary is invalid")

    task = str(task_start["task"])
    commit = str(task_start["source_commit"])
    source_path = str(task_start["source_path"])
    description_path = str(task_start["description_path"])
    expected_source_path = f"AlgoTuneTasks/{task}/{task}.py"
    expected_description_path = f"AlgoTuneTasks/{task}/description.txt"
    if source_path != expected_source_path or description_path != expected_description_path:
        raise RuntimeError("task paths do not match frozen source layout")

    base = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{commit}"
    source_raw = fetch(f"{base}/{source_path}")
    description_raw = fetch(f"{base}/{description_path}")
    source = source_raw.decode("utf-8")

    task_fingerprint = fingerprint(source, source)
    proposals = {
        arm: [asdict(value) for value in generate_proposals(
            task_fingerprint,
            arm=arm,
            limit=int(task_start["proposal_limit_per_arm"]),
            random_seed="LEXIGEN-V4-GENERALIZATION-2026-07-28-A",
        )]
        for arm in ARMS
    }
    report = {
        "campaign": task_start["campaign"],
        "engine_version": ENGINE_VERSION,
        "task_index": task_start["task_index"],
        "task": task,
        "selector_family": task_start["selector_family"],
        "revision": task_start["revision"],
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": commit,
        "source_path": source_path,
        "source_git_blob_sha1": git_blob(source_raw),
        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "source_size_bytes": len(source_raw),
        "description_path": description_path,
        "description_git_blob_sha1": git_blob(description_raw),
        "description_sha256": hashlib.sha256(description_raw).hexdigest(),
        "description_size_bytes": len(description_raw),
        "fingerprint": asdict(task_fingerprint),
        "arms": proposals,
        "proposal_counts": {arm: len(values) for arm, values in proposals.items()},
        "task_source_opened": True,
        "description_opened": True,
        "training_manifest_opened": False,
        "training_payloads_opened": False,
        "test_manifest_opened": False,
        "test_payloads_opened": False,
        "reports_opened": False,
        "public_solvers_opened": False,
        "source_contents_emitted": False,
        "human_task_specific_solver_design": False
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["analysis_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source-analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "task": task,
        "engine_version": ENGINE_VERSION,
        "features": task_fingerprint.features,
        "proposal_counts": report["proposal_counts"],
        "analysis_sha256": report["analysis_sha256"]
    }, indent=2))


if __name__ == "__main__":
    main()
