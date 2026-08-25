from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import fvd_allocator_r1 as allocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT_PATH = HERE / "FVD_HISTORICAL_REPLAY_INPUT_R1.json"
BUILDER = HERE / "build_fvd_experience_r1.py"
ARMS = ["fvd_full", "fvd_no_experience", "fvd_shuffled_experience", "retrieval_only", "evolution_only"]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_source_record(task: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(task["source_record_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    got = git_blob_sha1(path)
    expected = str(task["source_record_git_blob_sha1"])
    if got != expected:
        raise RuntimeError(f"source hash mismatch for {task['replay_id']}: {got} != {expected}")
    source = load_json(path)
    for key, expected_value in task["source_record_pre_result_boundary"].items():
        if source.get(key) != expected_value:
            raise RuntimeError(
                f"pre-result boundary mismatch for {task['replay_id']} {key}: "
                f"{source.get(key)!r} != {expected_value!r}"
            )
    return {"path": str(task["source_record_path"]), "git_blob_sha1": got, "pre_result_boundary_verified": True}


def build_leave_one_out(task_name: str, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(BUILDER),
        "--output",
        str(output),
        "--exclude-evidence-task",
        task_name,
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    artifact = load_json(output)
    loo = artifact.get("development_leave_one_out", {})
    if task_name not in loo.get("excluded_evidence_tasks", []):
        raise RuntimeError(f"leave-one-out exclusion missing for {task_name}")
    return artifact


def allocation_view(artifact: dict[str, Any], replay_id: str) -> dict[str, Any]:
    # Deliberately omit historical_summary and global_guards.evidence. The allocator
    # receives only generic profiles after target exclusion and non-outcome guard flags.
    view: dict[str, Any] = {
        "schema": "lexigen-v8-fvd-replay-allocation-view-r1",
        "replay_id": replay_id,
        "proposal_profiles": artifact["proposal_profiles"],
        "official_holdout_data_accessed": False,
        "artifact_sha256": "",
    }
    view["artifact_sha256"] = canonical_sha256({**view, "artifact_sha256": ""})
    return view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=16)
    args = parser.parse_args()
    if args.budget <= 0:
        raise ValueError("budget must be positive")

    replay_input = load_json(INPUT_PATH)
    if replay_input.get("outcome_fields_present") is not False:
        raise RuntimeError("source-only replay input contains outcome fields")
    if replay_input.get("official_holdout_data_accessed") is not False:
        raise RuntimeError("official holdout boundary crossed in replay input")

    records = []
    with tempfile.TemporaryDirectory() as td_raw:
        td = Path(td_raw)
        for task in replay_input["tasks"]:
            replay_id = str(task["replay_id"])
            source_verification = verify_source_record(task)
            task_name = str(task["historical_task_name_for_exclusion"])
            artifact = build_leave_one_out(task_name, td / f"{replay_id}-experience.json")
            view = allocation_view(artifact, replay_id)

            if task_name == "vertex_cover":
                frontier = next(p for p in view["proposal_profiles"] if p["proposal_class_id"] == "PC-FRONTIER-CERT")
                if int(frontier.get("confirmed_causal_successes", -1)) != 0:
                    raise RuntimeError("vertex-cover causal outcome leaked into its leave-one-out allocation view")
                if frontier.get("evidence_grade") != "source_learned_unconfirmed":
                    raise RuntimeError("vertex-cover leave-one-out profile retained confirmed-causal grade")

            descriptor = dict(task["task_descriptor"])
            forbidden_descriptor_keys = {"successful_proposal_classes", "historical_outcome", "outcome", "target_class"}
            if forbidden_descriptor_keys & set(descriptor):
                raise RuntimeError(f"outcome-like key entered source descriptor for {replay_id}")

            allocations: dict[str, Any] = {}
            for arm in ARMS:
                result = allocator.run_allocator(view, descriptor, arm, args.budget)
                if result["budget_used"] != args.budget or sum(result["allocation"].values()) != args.budget:
                    raise RuntimeError(f"budget mismatch for {replay_id}/{arm}")
                if result["official_holdout_data_accessed"] is not False:
                    raise RuntimeError(f"official holdout boundary crossed for {replay_id}/{arm}")
                allocations[arm] = result

            records.append({
                "replay_id": replay_id,
                "source_verification": source_verification,
                "historical_task_name_excluded": task_name,
                "leave_one_out_artifact_sha256": artifact["artifact_sha256"],
                "allocation_view_sha256": view["artifact_sha256"],
                "task_descriptor": descriptor,
                "allocations": allocations,
            })

    payload: dict[str, Any] = {
        "schema": "lexigen-v8-fvd-historical-replay-stage1-r1",
        "stage": "source_only_allocations_complete_before_outcomes",
        "outcomes_loaded": False,
        "outcome_file_opened": False,
        "proposal_budget_per_task_per_arm": args.budget,
        "arms": ARMS,
        "records": records,
        "all_source_hashes_verified": True,
        "all_leave_one_out_checks_pass": True,
        "all_equal_budget_checks_pass": True,
        "official_holdout_data_accessed": False,
        "blocked_ipwm_real_intervention_data_accessed": False,
        "scientific_transfer_evidence": False,
        "stage1_sha256": "",
    }
    payload["stage1_sha256"] = canonical_sha256({**payload, "stage1_sha256": ""})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "stage1_sha256": payload["stage1_sha256"],
        "replay_task_count": len(records),
        "outcomes_loaded": False,
        "all_source_hashes_verified": True,
        "all_leave_one_out_checks_pass": True,
        "all_equal_budget_checks_pass": True,
        "official_holdout_data_accessed": False,
        "scientific_transfer_evidence": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
