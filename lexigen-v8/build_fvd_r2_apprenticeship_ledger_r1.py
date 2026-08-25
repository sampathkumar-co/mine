from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from fvd_r2_common import canonical_sha256, classes_from_candidate, nested_get


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def git_blob_sha1(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, default=Path("lexigen-v8/FVD_R2_APPRENTICESHIP_OUTCOMES_R1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    descriptors = load_json(args.descriptors)
    outcomes = load_json(args.outcomes)
    if descriptors.get("calibration_outcomes_loaded") is not False:
        raise RuntimeError("descriptor artifact crossed calibration outcome boundary")
    if outcomes.get("calibration_outcome_paths_present") is not False:
        raise RuntimeError("apprenticeship manifest contains calibration outcome paths")

    descriptor_map = {str(row["task"]): row for row in descriptors["records"]}
    records: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for row in outcomes["tasks"]:
        task = str(row["task"])
        descriptor = descriptor_map.get(task)
        if descriptor is None or descriptor.get("partition") != "apprenticeship":
            raise RuntimeError(f"non-apprenticeship task in ledger build: {task}")
        path = Path(str(row["outcome_path"]))
        observed_blob = git_blob_sha1(path)
        if observed_blob != row["outcome_blob_sha1"]:
            raise RuntimeError(f"outcome blob mismatch for {task}: {observed_blob}")
        outcome = load_json(path)
        candidate = str(nested_get(outcome, str(row["candidate_field"])))
        classes = classes_from_candidate(candidate)
        if not classes:
            raise RuntimeError(f"generic candidate mapper produced no classes for {task}: {candidate}")
        gate_success = bool(row["gate_success"])
        tasks.append({
            "task": task,
            "candidate": candidate,
            "mapped_classes": classes,
            "gate_success": gate_success,
            "outcome_blob_sha1": observed_blob,
        })
        for class_id in classes:
            records.append({
                "task": task,
                "proposal_class_id": class_id,
                "traits": list(descriptor["traits"]),
                "gate_success": gate_success,
                "candidate": candidate,
                "outcome_blob_sha1": observed_blob,
            })

    payload: dict[str, Any] = {
        "schema": "lexigen-v8-fvd-r2-apprenticeship-ledger-r1",
        "development_only": True,
        "calibration_outcomes_loaded": False,
        "official_fvd_holdouts_accessed": False,
        "blocked_ipwm_real_data_accessed": False,
        "source_descriptor_artifact_sha256": descriptors["artifact_sha256"],
        "task_count": len(tasks),
        "tasks": sorted(tasks, key=lambda x: x["task"]),
        "records": sorted(records, key=lambda x: (x["proposal_class_id"], x["task"])),
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact_sha256": payload["artifact_sha256"],
        "task_count": payload["task_count"],
        "record_count": len(payload["records"]),
        "calibration_outcomes_loaded": payload["calibration_outcomes_loaded"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
