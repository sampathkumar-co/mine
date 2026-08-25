from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from fvd_r2_common import canonical_sha256, classes_from_candidate, classes_from_root_cause, nested_get


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def git_blob_sha1(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return float(statistics.mean(values))


def rank_of(row: dict[str, Any], classes: list[str]) -> int:
    positions = {str(item["proposal_class_id"]): index + 1 for index, item in enumerate(row["ranking"])}
    found = [positions[class_id] for class_id in classes if class_id in positions]
    if not found:
        raise RuntimeError(f"none of target classes found in ranking: {classes}")
    return min(found)


def max_allocation(row: dict[str, Any], classes: list[str]) -> int:
    allocation = row["allocation"]
    found = [int(allocation[class_id]) for class_id in classes if class_id in allocation]
    if not found:
        raise RuntimeError(f"none of target classes found in allocation: {classes}")
    return max(found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, default=Path("lexigen-v8/FVD_R2_CALIBRATION_OUTCOMES_R1.json"))
    parser.add_argument("--protocol", type=Path, default=Path("lexigen-v8/FVD_R2_CALIBRATION_PROTOCOL_R1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage1 = load_json(args.stage1)
    outcomes = load_json(args.outcomes)
    protocol = load_json(args.protocol)
    observed_stage1_sha = canonical_sha256({k: v for k, v in stage1.items() if k != "stage1_sha256"})
    if observed_stage1_sha != stage1.get("stage1_sha256"):
        raise RuntimeError("stage1 artifact hash mismatch")
    if stage1.get("calibration_outcomes_loaded") is not False or stage1.get("calibration_outcome_file_opened") is not False:
        raise RuntimeError("stage1 crossed calibration outcome boundary")
    if stage1.get("official_fvd_holdouts_accessed") is not False:
        raise RuntimeError("official FVD holdout boundary crossed")

    allocation_map = {(str(row["task"]), str(row["arm"])): row for row in stage1["allocations"]}
    arms = list(map(str, protocol["arms"]))
    budget = int(protocol["proposal_budget_per_task_per_arm"])
    positive_records: list[dict[str, Any]] = []
    negative_records: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []

    for spec in outcomes["tasks"]:
        task = str(spec["task"])
        evidence: list[dict[str, Any]] = []
        for item in spec["outcome_evidence"]:
            path = Path(str(item["path"]))
            observed_blob = git_blob_sha1(path)
            if observed_blob != item["blob_sha1"]:
                raise RuntimeError(f"calibration outcome blob mismatch for {task}: {path}")
            evidence.append(load_json(path))

        outcome_type = str(spec["outcome_type"])
        if outcome_type == "positive":
            source = evidence[int(spec["candidate_source_evidence_index"])]
            candidate = str(nested_get(source, str(spec["candidate_field"])))
            target_classes = classes_from_candidate(candidate)
            if not target_classes:
                raise RuntimeError(f"no positive target classes mapped for {task}: {candidate}")
        elif outcome_type == "negative_performance":
            source = evidence[int(spec["candidate_source_evidence_index"])]
            candidate = str(nested_get(source, str(spec["candidate_field"])))
            target_classes = classes_from_candidate(candidate)
            if not target_classes:
                raise RuntimeError(f"no negative target classes mapped for {task}: {candidate}")
        elif outcome_type == "negative_correctness":
            source = evidence[int(spec["root_cause_evidence_index"])]
            root_cause = str(source.get("root_cause", ""))
            target_classes = classes_from_root_cause(root_cause)
            candidate = None
            if not target_classes:
                raise RuntimeError(f"no root-cause target classes mapped for {task}: {root_cause}")
        else:
            raise RuntimeError(f"unknown calibration outcome type: {outcome_type}")

        arm_metrics: dict[str, Any] = {}
        for arm in arms:
            row = allocation_map.get((task, arm))
            if row is None:
                raise RuntimeError(f"missing frozen allocation for {task}/{arm}")
            if int(row["budget_used"]) != budget:
                raise RuntimeError(f"budget mismatch for {task}/{arm}")
            target_allocation = max_allocation(row, target_classes)
            target_rank = rank_of(row, target_classes)
            if outcome_type == "positive":
                utility = target_allocation / budget
            else:
                utility = 1.0 - target_allocation / budget
            arm_metrics[arm] = {
                "target_allocation": target_allocation,
                "target_rank": target_rank,
                "task_utility": utility,
            }

        record = {
            "task": task,
            "outcome_type": outcome_type,
            "candidate": candidate,
            "target_classes": target_classes,
            "arm_metrics": arm_metrics,
        }
        task_records.append(record)
        if outcome_type == "positive":
            positive_records.append(record)
        else:
            negative_records.append(record)

    if len(positive_records) != int(protocol["positive_task_count"]):
        raise RuntimeError("positive calibration task count mismatch")
    if len(negative_records) != int(protocol["negative_task_count"]):
        raise RuntimeError("negative calibration task count mismatch")

    full_positive_ranks = [float(r["arm_metrics"]["fvd_full"]["target_rank"]) for r in positive_records]
    retrieval_positive_ranks = [float(r["arm_metrics"]["retrieval_only"]["target_rank"]) for r in positive_records]
    full_positive_alloc = [float(r["arm_metrics"]["fvd_full"]["target_allocation"]) for r in positive_records]
    retrieval_positive_alloc = [float(r["arm_metrics"]["retrieval_only"]["target_allocation"]) for r in positive_records]
    full_negative_alloc = [float(r["arm_metrics"]["fvd_full"]["target_allocation"]) for r in negative_records]
    retrieval_negative_alloc = [float(r["arm_metrics"]["retrieval_only"]["target_allocation"]) for r in negative_records]
    mean_utility = {
        arm: mean([float(r["arm_metrics"][arm]["task_utility"]) for r in task_records])
        for arm in arms
    }
    top3_count = sum(1 for rank in full_positive_ranks if rank <= 3.0)

    metrics = {
        "full_positive_success_class_top3_tasks": top3_count,
        "full_mean_positive_best_success_rank": mean(full_positive_ranks),
        "retrieval_mean_positive_best_success_rank": mean(retrieval_positive_ranks),
        "full_mean_positive_best_success_allocation": mean(full_positive_alloc),
        "retrieval_mean_positive_best_success_allocation": mean(retrieval_positive_alloc),
        "full_minus_retrieval_mean_positive_allocation_slots": mean(full_positive_alloc) - mean(retrieval_positive_alloc),
        "full_mean_negative_max_failed_class_allocation": mean(full_negative_alloc),
        "retrieval_mean_negative_max_failed_class_allocation": mean(retrieval_negative_alloc),
        "full_minus_retrieval_mean_negative_allocation_slots": mean(full_negative_alloc) - mean(retrieval_negative_alloc),
        "mean_task_utility_by_arm": mean_utility,
        "full_minus_retrieval_mean_task_utility": mean_utility["fvd_full"] - mean_utility["retrieval_only"],
        "full_minus_no_experience_mean_task_utility": mean_utility["fvd_full"] - mean_utility["fvd_no_experience"],
        "full_minus_shuffled_mean_task_utility": mean_utility["fvd_full"] - mean_utility["fvd_shuffled_experience"],
    }

    frozen = protocol["frozen_gates"]
    gates = {
        "all_source_hashes_verified": stage1.get("all_source_hashes_verified") is True,
        "all_split_memberships_verified": stage1.get("all_split_memberships_verified") is True,
        "stage1_calibration_outcomes_loaded_false": stage1.get("calibration_outcomes_loaded") is False,
        "all_equal_budget_checks_pass": stage1.get("all_equal_budget_checks_pass") is True,
        "full_positive_success_class_top3_tasks": top3_count >= int(frozen["full_positive_success_class_top3_tasks_min"]),
        "full_mean_positive_best_success_rank_not_worse_than_retrieval": metrics["full_mean_positive_best_success_rank"] <= metrics["retrieval_mean_positive_best_success_rank"],
        "full_mean_positive_best_success_allocation_minus_retrieval": metrics["full_minus_retrieval_mean_positive_allocation_slots"] >= float(frozen["full_mean_positive_best_success_allocation_minus_retrieval_min_slots"]),
        "full_mean_negative_max_failed_class_allocation_minus_retrieval": metrics["full_minus_retrieval_mean_negative_allocation_slots"] <= float(frozen["full_mean_negative_max_failed_class_allocation_minus_retrieval_max_slots"]),
        "full_mean_task_utility_minus_retrieval": metrics["full_minus_retrieval_mean_task_utility"] >= float(frozen["full_mean_task_utility_minus_retrieval_min"]),
        "full_mean_task_utility_minus_no_experience": metrics["full_minus_no_experience_mean_task_utility"] >= float(frozen["full_mean_task_utility_minus_no_experience_min"]),
        "full_mean_task_utility_minus_shuffled": metrics["full_minus_shuffled_mean_task_utility"] >= float(frozen["full_mean_task_utility_minus_shuffled_min"]),
        "official_fvd_holdouts_accessed_false": stage1.get("official_fvd_holdouts_accessed") is False,
        "scientific_transfer_evidence_false": stage1.get("scientific_transfer_evidence") is False,
    }
    passed = all(gates.values())
    payload = {
        "schema": "lexigen-v8-fvd-r2-calibration-result-r1",
        "classification": "development_calibration_passed" if passed else "development_calibration_failed",
        "calibration_gate_passed": passed,
        "stage1_sha256": stage1["stage1_sha256"],
        "proposal_budget_per_task_per_arm": budget,
        "gate_checks": gates,
        "metrics": metrics,
        "records": sorted(task_records, key=lambda x: x["task"]),
        "official_fvd_holdouts_accessed": False,
        "blocked_ipwm_real_data_accessed": False,
        "scientific_transfer_evidence": False,
        "claim_boundary": "Development calibration only. A pass permits only a separate historical confirmatory replay; it does not unlock official FVD holdouts or establish causal transfer.",
    }
    payload["result_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
