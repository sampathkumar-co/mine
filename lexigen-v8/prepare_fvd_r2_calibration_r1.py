from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fvd_allocator_r2 import run_allocator
from fvd_r2_common import canonical_sha256


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--proposal-classes", type=Path, default=Path("lexigen-v8/FVD_PROPOSAL_CLASSES_R1.json"))
    parser.add_argument("--controller", type=Path, default=Path("lexigen-v8/FVD_R2_CONTROLLER_SPEC.json"))
    parser.add_argument("--protocol", type=Path, default=Path("lexigen-v8/FVD_R2_CALIBRATION_PROTOCOL_R1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    descriptors = load_json(args.descriptors)
    ledger = load_json(args.ledger)
    proposal_classes = load_json(args.proposal_classes)
    controller = load_json(args.controller)
    protocol = load_json(args.protocol)

    if descriptors.get("calibration_outcomes_loaded") is not False:
        raise RuntimeError("source descriptor artifact crossed outcome boundary")
    if ledger.get("calibration_outcomes_loaded") is not False:
        raise RuntimeError("apprenticeship ledger crossed outcome boundary")
    if protocol.get("status") != "frozen_before_calibration_execution":
        raise RuntimeError("calibration protocol is not frozen")
    budget = int(protocol["proposal_budget_per_task_per_arm"])
    arms = list(map(str, protocol["arms"]))

    allocations: list[dict[str, Any]] = []
    for record in descriptors["records"]:
        if record.get("partition") != "calibration":
            continue
        task = dict(record)
        task["descriptor_sha256"] = canonical_sha256(record)
        task["official_fvd_holdouts_accessed"] = False
        for arm in arms:
            result = run_allocator(proposal_classes, controller, ledger, task, arm, budget)
            if result["budget_used"] != budget:
                raise RuntimeError(f"budget mismatch: {task['task']} {arm}")
            allocations.append(result)

    expected = int(protocol["positive_task_count"]) + int(protocol["negative_task_count"])
    observed_tasks = sorted({row["task"] for row in allocations})
    if len(observed_tasks) != expected:
        raise RuntimeError(f"expected {expected} calibration tasks, got {len(observed_tasks)}")
    if len(allocations) != expected * len(arms):
        raise RuntimeError("allocation arm count mismatch")

    payload: dict[str, Any] = {
        "schema": "lexigen-v8-fvd-r2-calibration-stage1-r1",
        "development_only": True,
        "source_descriptor_artifact_sha256": descriptors["artifact_sha256"],
        "apprenticeship_ledger_sha256": ledger["artifact_sha256"],
        "controller_spec_sha256": canonical_sha256(controller),
        "protocol_sha256": canonical_sha256(protocol),
        "proposal_budget_per_task_per_arm": budget,
        "calibration_tasks": observed_tasks,
        "all_source_hashes_verified": bool(descriptors["all_source_hashes_verified"]),
        "all_split_memberships_verified": bool(descriptors["all_split_memberships_verified"]),
        "all_equal_budget_checks_pass": True,
        "calibration_outcomes_loaded": False,
        "calibration_outcome_file_opened": False,
        "official_fvd_holdouts_accessed": False,
        "blocked_ipwm_real_data_accessed": False,
        "scientific_transfer_evidence": False,
        "allocations": allocations,
    }
    payload["stage1_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "stage1_sha256": payload["stage1_sha256"],
        "calibration_tasks": observed_tasks,
        "allocation_count": len(allocations),
        "calibration_outcomes_loaded": payload["calibration_outcomes_loaded"],
        "all_equal_budget_checks_pass": payload["all_equal_budget_checks_pass"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
