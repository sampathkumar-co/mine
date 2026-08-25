from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK1_FINAL_SELECTION_COMMIT = "3d2aa3c8a1ee9eeb284e3494655c1110078b394e"
TASK2_PROPOSAL_COMMIT = "8fb61bb3f320d62f7afaeef5d9c6bc5adf44766a"
TASK2_PREFLIGHT_HEAD = "4075a0372fa744182cdc91c64cc71870ae33ce61"
TASK2_PREFLIGHT_RUN_ID = 32747466819

TASK1_BLOBS = {
    "proposals": "ecc41ef14724a0f1ccbd93bc860cbaf99c783b59",
    "full_feedback": "65f32f1b2bca7ec8af1d885e20a21ea63c3da961",
    "revision1": "bf30d9de0129ec90fd7686b3715e6474292c7e53",
    "revision2": "4bf9cfb165655d08fcc2eaa6c66b70ecc34710b0",
}
TASK2_BLOBS = {
    "proposals": "24ad51830d598df47fb25226dba77143a81b373c",
    "selection": "be4a61c472151ff48af18fb0329763e14a3572bc",
}

EXPECTED_TASK1_STAGES = {
    "full_14_test_feedback_round0_r2_sealed",
    "revision_slot_1_14_test_result_sealed",
    "revision_slot_2_14_test_result_sealed",
}
FORBIDDEN_RESULT_FLAGS = (
    "expert_target_timing_accessed",
    "expert_opt_commit_accessed",
    "expert_diff_accessed",
    "hints_accessed",
    "human_task_specific_solver_contribution",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task1_proposal_sequences(payload: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    if payload.get("stage") != "source_only_proposals_frozen_before_feedback":
        raise ValueError("Task1 proposals are not the frozen pre-feedback pool")
    if payload.get("timing_feedback_used") or payload.get("correctness_feedback_used"):
        raise ValueError("Task1 proposal pool crossed a feedback boundary")
    if payload.get("expert_patch_used") or payload.get("hints_used"):
        raise ValueError("Task1 proposal pool crossed an expert boundary")

    out: dict[str, tuple[str, ...]] = {}
    for arm in ("v7_full", "v7_no_library", "v7_random_library"):
        for row in payload["arms"][arm]:
            candidate = str(row["candidate_id"])
            if candidate in out:
                raise ValueError(f"duplicate Task1 candidate {candidate}")
            out[candidate] = tuple(str(x) for x in row["primitive_sequence"])
    return out


def task1_sequence_priors(
    proposals: dict[str, Any],
    result_payloads: list[dict[str, Any]],
) -> dict[tuple[str, ...], dict[str, Any]]:
    sequences = task1_proposal_sequences(proposals)
    priors: dict[tuple[str, ...], dict[str, Any]] = {}
    seen_stages: set[str] = set()

    for payload in result_payloads:
        stage = str(payload.get("stage", ""))
        if stage not in EXPECTED_TASK1_STAGES:
            raise ValueError(f"unsupported Task1 evidence stage {stage}")
        if payload.get("status") != "success":
            raise ValueError(f"Task1 stage {stage} was not successful")
        if not payload.get("all_14_test_hashes_verified"):
            raise ValueError(f"Task1 stage {stage} did not verify all tests")
        for flag in FORBIDDEN_RESULT_FLAGS:
            if bool(payload.get(flag, True)):
                raise ValueError(f"Task1 evidence crossed forbidden boundary: {flag}")
        seen_stages.add(stage)

        for arm in ("v7_full", "v7_no_library", "v7_random_library"):
            row = payload["candidates"][arm]
            if not row.get("all_14_equivalence_checks_passed"):
                continue
            candidate = str(row["candidate_id"])
            sequence = sequences[candidate]
            if sequence in priors:
                raise ValueError(f"multiple evaluated Task1 candidates share sequence {sequence}")
            priors[sequence] = {
                "source_candidate": candidate,
                "source_arm": arm,
                "source_stage": stage,
                "harmonic_speedup_vs_base": float(row["harmonic_speedup_vs_base"]),
                "minimum_speedup_vs_base": float(row["minimum_speedup_vs_base"]),
                "source_patch_sha256": str(row["patch_sha256"]),
            }

    if seen_stages != EXPECTED_TASK1_STAGES:
        raise ValueError("Task1 evidence stages incomplete")
    return priors


def task2_selected_sequences(
    proposals: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if proposals.get("timing_feedback_used"):
        raise ValueError("Task2 proposals used timing feedback")
    for flag in ("expert_opt_commit_accessed", "expert_diff_accessed", "hints_accessed"):
        if bool(proposals.get(flag, True)):
            raise ValueError(f"Task2 proposals crossed forbidden boundary: {flag}")
    if selection.get("stage") != "tasks2_6_concrete_candidate_selection_pre_timing":
        raise ValueError("Task2 selection is not the pre-timing selection")
    if selection.get("candidate_timing_used"):
        raise ValueError("Task2 selection used candidate timing")
    if selection.get("expert_information_used_for_tasks_2_3_4_6"):
        raise ValueError("Task2 selection used expert information")

    rows: dict[str, dict[str, Any]] = {}
    for raw in proposals["proposals"]:
        candidate = str(raw[0])
        rows[candidate] = {
            "arm": str(raw[1]),
            "primitive_sequence": tuple(str(x) for x in raw[2]),
            "macro_ids": tuple(str(x) for x in raw[3]),
            "mechanism": str(raw[4]),
            "correctness_risk": str(raw[6]),
        }

    selected: dict[str, dict[str, Any]] = {}
    for arm in ("v7_full", "v7_no_library", "v7_random_library"):
        for candidate in selection["tasks"]["2"][arm]:
            row = rows[str(candidate)]
            if row["arm"] != arm:
                raise ValueError(f"Task2 arm mismatch for {candidate}")
            selected[str(candidate)] = row
    if len(selected) != 9:
        raise ValueError("expected exactly nine frozen Task2 preflight candidates")
    return selected


def build_prediction(
    task1_proposals: dict[str, Any],
    task1_results: list[dict[str, Any]],
    task2_proposals: dict[str, Any],
    task2_selection: dict[str, Any],
) -> dict[str, Any]:
    priors = task1_sequence_priors(task1_proposals, task1_results)
    target = task2_selected_sequences(task2_proposals, task2_selection)

    covered: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    for candidate in sorted(target):
        row = target[candidate]
        sequence = row["primitive_sequence"]
        prior = priors.get(sequence)
        base = {
            "candidate_id": candidate,
            "arm": row["arm"],
            "primitive_sequence": list(sequence),
            "macro_ids": list(row["macro_ids"]),
        }
        if prior is None:
            abstained.append({**base, "reason": "no_exact_Task1_evaluated_sequence"})
            continue
        covered.append(
            {
                **base,
                "transferred_prior_harmonic_speedup": prior["harmonic_speedup_vs_base"],
                "transferred_prior_minimum_speedup": prior["minimum_speedup_vs_base"],
                "source_Task1_candidate": prior["source_candidate"],
                "source_Task1_arm": prior["source_arm"],
                "source_Task1_stage": prior["source_stage"],
                "source_Task1_patch_sha256": prior["source_patch_sha256"],
            }
        )

    covered.sort(
        key=lambda row: (-row["transferred_prior_harmonic_speedup"], row["candidate_id"])
    )
    for rank, row in enumerate(covered, 1):
        row["prospective_rank"] = rank

    return {
        "project": "LEXIGEN OMEGA",
        "stage": "task2_prospective_exact_sequence_transfer_r1",
        "status": "frozen_before_Task2_outcome",
        "scientific_role": "minimal prospective baseline for cross_repository_mechanism_transfer",
        "prediction_rule": "For each frozen Task2 preflight candidate, if its primitive_sequence exactly matches a Task1 candidate that completed a sealed 14-test feedback round, transfer that Task1 harmonic speedup as the prior score. Otherwise abstain. No fuzzy matching, task-specific reinterpretation, or Task2 timing is allowed.",
        "source_task": "pydantic__pydantic-addf1f9",
        "target_task": "abetlen__llama-cpp-python-2bc1d97",
        "inputs": {
            "Task1_final_selection_commit": TASK1_FINAL_SELECTION_COMMIT,
            "Task1_git_blob_sha1": TASK1_BLOBS,
            "Task2_frozen_proposal_commit": TASK2_PROPOSAL_COMMIT,
            "Task2_git_blob_sha1": TASK2_BLOBS,
            "Task2_preflight_head_sha": TASK2_PREFLIGHT_HEAD,
            "Task2_preflight_run_id": TASK2_PREFLIGHT_RUN_ID,
        },
        "Task2_outcome_accessed": False,
        "Task2_timing_accessed": False,
        "Task2_candidate_logs_accessed_for_prediction": False,
        "covered_candidate_count": len(covered),
        "abstained_candidate_count": len(abstained),
        "prospective_ranking": covered,
        "abstentions": sorted(abstained, key=lambda row: row["candidate_id"]),
        "interpretation_boundary": "This is a deliberately weak exact-sequence transfer baseline, not a learned model and not causal evidence. Near-1.0 Task1 scores may contain timing noise; success or failure is informative about whether primitive-sequence identity alone transfers across repositories.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task1-proposals", type=Path, required=True)
    ap.add_argument("--task1-full-feedback", type=Path, required=True)
    ap.add_argument("--task1-revision1", type=Path, required=True)
    ap.add_argument("--task1-revision2", type=Path, required=True)
    ap.add_argument("--task2-proposals", type=Path, required=True)
    ap.add_argument("--task2-selection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect", type=Path)
    args = ap.parse_args()

    prediction = build_prediction(
        load(args.task1_proposals),
        [load(args.task1_full_feedback), load(args.task1_revision1), load(args.task1_revision2)],
        load(args.task2_proposals),
        load(args.task2_selection),
    )
    text = json.dumps(prediction, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")

    if args.expect is not None:
        expected = json.dumps(load(args.expect), indent=2, sort_keys=True) + "\n"
        if text != expected:
            raise SystemExit("prospective prediction drifted from frozen lock")

    print(json.dumps({
        "status": prediction["status"],
        "covered": prediction["covered_candidate_count"],
        "abstained": prediction["abstained_candidate_count"],
        "ranking": [row["candidate_id"] for row in prediction["prospective_ranking"]],
        "Task2_outcome_accessed": prediction["Task2_outcome_accessed"],
    }, indent=2))


if __name__ == "__main__":
    main()
