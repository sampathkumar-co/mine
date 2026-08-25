from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL_PATH = HERE / "FVD_HISTORICAL_REPLAY_PROTOCOL_R1.json"
OUTCOMES_PATH = HERE / "FVD_HISTORICAL_REPLAY_OUTCOMES_R1.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def git_blob_sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def verify_outcome_evidence(row: dict[str, Any]) -> list[dict[str, Any]]:
    verified = []
    for ev in row["outcome_evidence"]:
        expected = str(ev["git_blob_sha1"])
        path = str(ev["path"])
        commit = ev.get("repository_commit")
        if commit:
            subprocess.run(
                ["git", "fetch", "--quiet", "--depth=1", "origin", str(commit)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        else:
            data = (ROOT / path).read_bytes()
        got = git_blob_sha1_bytes(data)
        if got != expected:
            raise RuntimeError(f"outcome evidence hash mismatch for {row['replay_id']}: {got} != {expected}")
        verified.append({"path": path, "repository_commit": commit, "git_blob_sha1": got})
    return verified


def class_rank(allocation_result: dict[str, Any], success_classes: list[str]) -> int:
    ordered = [str(row["proposal_class_id"]) for row in allocation_result["ranking"]]
    ranks = [ordered.index(pid) + 1 for pid in success_classes if pid in ordered]
    if not ranks:
        raise RuntimeError(f"none of successful classes found in ranking: {success_classes}")
    return min(ranks)


def class_allocation(allocation_result: dict[str, Any], success_classes: list[str]) -> int:
    allocation = allocation_result["allocation"]
    values = [int(allocation.get(pid, 0)) for pid in success_classes]
    if not values:
        raise RuntimeError("successful class list is empty")
    return max(values)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stage1 = load_json(args.stage1)
    if stage1.get("outcomes_loaded") is not False or stage1.get("outcome_file_opened") is not False:
        raise RuntimeError("stage1 is not outcome-blind")
    if stage1.get("all_source_hashes_verified") is not True:
        raise RuntimeError("stage1 source verification failed")
    if stage1.get("all_leave_one_out_checks_pass") is not True:
        raise RuntimeError("stage1 leave-one-out check failed")
    if stage1.get("all_equal_budget_checks_pass") is not True:
        raise RuntimeError("stage1 budget check failed")

    protocol = load_json(PROTOCOL_PATH)
    outcomes = load_json(OUTCOMES_PATH)
    stage_records = {str(row["replay_id"]): row for row in stage1["records"]}
    outcome_records = {str(row["replay_id"]): row for row in outcomes["tasks"]}
    if set(stage_records) != set(outcome_records):
        raise RuntimeError("stage1/outcome replay IDs differ")
    if len(stage_records) != int(protocol["replay_task_count"]):
        raise RuntimeError("replay task count differs from frozen protocol")

    scored = []
    for replay_id in sorted(stage_records):
        source = stage_records[replay_id]
        outcome = outcome_records[replay_id]
        success_classes = [str(x) for x in outcome["successful_proposal_classes"]]
        verified_outcomes = verify_outcome_evidence(outcome)
        arm_metrics: dict[str, Any] = {}
        for arm, allocation_result in source["allocations"].items():
            arm_metrics[arm] = {
                "best_success_class_allocation": class_allocation(allocation_result, success_classes),
                "best_success_class_rank": class_rank(allocation_result, success_classes),
                "experience_view_sha256": allocation_result["experience_view_sha256"],
            }
        scored.append({
            "replay_id": replay_id,
            "historical_outcome": outcome["historical_outcome"],
            "successful_proposal_classes": success_classes,
            "outcome_evidence_verified": verified_outcomes,
            "arm_metrics": arm_metrics,
        })

    full_alloc = [float(x["arm_metrics"]["fvd_full"]["best_success_class_allocation"]) for x in scored]
    noexp_alloc = [float(x["arm_metrics"]["fvd_no_experience"]["best_success_class_allocation"]) for x in scored]
    shuffled_alloc = [float(x["arm_metrics"]["fvd_shuffled_experience"]["best_success_class_allocation"]) for x in scored]
    full_ranks = [float(x["arm_metrics"]["fvd_full"]["best_success_class_rank"]) for x in scored]
    retrieval_ranks = [float(x["arm_metrics"]["retrieval_only"]["best_success_class_rank"]) for x in scored]

    mean_alloc_margin = mean(full_alloc) - mean(noexp_alloc)
    top3_count = sum(1 for rank in full_ranks if rank <= 3)
    wins_vs_shuffled = sum(1 for f, s in zip(full_alloc, shuffled_alloc) if f > s)
    full_mean_rank = mean(full_ranks)
    retrieval_mean_rank = mean(retrieval_ranks)

    gates_cfg = protocol["success_gates_all_required"]
    gate_checks = {
        "source_hashes_verified": stage1["all_source_hashes_verified"] is bool(gates_cfg["source_hashes_verified"]),
        "all_leave_one_target_out_checks_pass": stage1["all_leave_one_out_checks_pass"] is bool(gates_cfg["all_leave_one_target_out_checks_pass"]),
        "all_equal_budget_checks_pass": stage1["all_equal_budget_checks_pass"] is bool(gates_cfg["all_equal_budget_checks_pass"]),
        "full_mean_success_class_allocation_minus_no_experience": mean_alloc_margin >= float(gates_cfg["full_mean_success_class_allocation_minus_no_experience_min"]),
        "full_success_class_top3_tasks": top3_count >= int(gates_cfg["full_success_class_top3_tasks_min"]),
        "full_success_class_allocation_wins_vs_shuffled": wins_vs_shuffled >= int(gates_cfg["full_success_class_allocation_wins_vs_shuffled_min"]),
        "full_mean_best_success_class_rank_not_worse_than_retrieval": full_mean_rank <= retrieval_mean_rank if gates_cfg["full_mean_best_success_class_rank_not_worse_than_retrieval"] else True,
        "stage1_outcomes_loaded_false": stage1["outcomes_loaded"] is bool(gates_cfg["stage1_outcomes_loaded"]),
        "official_holdout_data_accessed_false": stage1["official_holdout_data_accessed"] is bool(gates_cfg["official_holdout_data_accessed"]),
        "scientific_transfer_evidence_false": stage1["scientific_transfer_evidence"] is bool(gates_cfg["scientific_transfer_evidence"]),
    }
    replay_gate_passed = all(gate_checks.values())

    payload = {
        "schema": "lexigen-v8-fvd-historical-replay-result-r1",
        "classification": "development_replay_pass" if replay_gate_passed else "development_replay_failed",
        "replay_gate_passed": replay_gate_passed,
        "stage1_sha256": stage1["stage1_sha256"],
        "replay_task_count": len(scored),
        "proposal_budget_per_task_per_arm": stage1["proposal_budget_per_task_per_arm"],
        "metrics": {
            "full_mean_success_class_allocation": mean(full_alloc),
            "no_experience_mean_success_class_allocation": mean(noexp_alloc),
            "full_minus_no_experience_mean_allocation": mean_alloc_margin,
            "full_success_class_top3_tasks": top3_count,
            "full_success_class_allocation_wins_vs_shuffled": wins_vs_shuffled,
            "full_mean_best_success_class_rank": full_mean_rank,
            "retrieval_mean_best_success_class_rank": retrieval_mean_rank,
        },
        "gate_checks": gate_checks,
        "records": scored,
        "official_holdout_data_accessed": False,
        "blocked_ipwm_real_intervention_data_accessed": False,
        "scientific_transfer_evidence": False,
        "claim_boundary": "Historical development replay only. A pass is not causal-transfer confirmation and does not start the official FVD holdout campaign.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not replay_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
