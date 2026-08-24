from __future__ import annotations

import argparse
import json
from pathlib import Path

from lexigen_omega.trajectory import (
    attach_proposal_provenance,
    build_mechanism_preferences,
    ingest_v7_gso_task1_revision_result,
    parse_v7_gso_proposals,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    proposal_payload = json.loads(args.proposals.read_text(encoding="utf-8"))
    result_payload = json.loads(args.result.read_text(encoding="utf-8"))

    proposals = parse_v7_gso_proposals(proposal_payload)
    observations = ingest_v7_gso_task1_revision_result(result_payload)
    joined = attach_proposal_provenance(observations, proposals)
    preferences = build_mechanism_preferences(joined)

    if {item.evaluation.candidate_id for item in joined} != {"F1", "N2", "R1"}:
        raise SystemExit("unexpected Task1 revision candidate set")
    if len(preferences) != 3:
        raise SystemExit(f"expected three pairwise preferences, got {len(preferences)}")

    ranked = sorted(
        joined,
        key=lambda item: (
            item.evaluation.score,
            float(item.evaluation.metrics.get("minimum_speedup", 0.0)),
            item.evaluation.candidate_id,
        ),
        reverse=True,
    )
    ranking = [item.evaluation.candidate_id for item in ranked]
    if ranking != ["R1", "N2", "F1"]:
        raise SystemExit(f"unexpected real-evidence ranking: {ranking}")

    by_id = {item.evaluation.candidate_id: item for item in joined}
    if by_id["N2"].evaluation.artifact_fingerprint != by_id["R1"].evaluation.artifact_fingerprint:
        raise SystemExit("expected sealed N2/R1 byte-identical control patches")
    if by_id["F1"].evaluation.artifact_fingerprint == by_id["R1"].evaluation.artifact_fingerprint:
        raise SystemExit("expected learned F1 patch to remain mechanistically distinct")
    if by_id["F1"].evaluation.score >= by_id["N2"].evaluation.score:
        raise SystemExit("historical negative control relation changed: F1 should trail N2")
    if by_id["F1"].evaluation.score >= by_id["R1"].evaluation.score:
        raise SystemExit("historical negative control relation changed: F1 should trail R1")

    report = {
        "experiment": "LEXIGEN OMEGA first real mechanism-level hindsight evidence",
        "status": "verified_negative_learned_macro_example",
        "source_task": result_payload["instance_id"],
        "source_stage": result_payload["stage"],
        "source_workflow_run_id": result_payload["workflow_run_id"],
        "proposal_source_commit": proposal_payload["source_result_commit"],
        "candidate_ranking": ranking,
        "observations": [
            {
                "candidate_id": item.evaluation.candidate_id,
                "arm": item.evaluation.arm,
                "harmonic_speedup": item.evaluation.score,
                "minimum_speedup": item.evaluation.metrics["minimum_speedup"],
                "patch_sha256": item.evaluation.artifact_fingerprint,
                "primitive_sequence": list(item.provenance.primitive_sequence),
                "macro_ids": list(item.provenance.macro_ids),
                "mechanism": item.provenance.mechanism,
                "source_visible_preconditions": item.provenance.source_visible_preconditions,
                "correctness_risk": item.provenance.correctness_risk,
            }
            for item in ranked
        ],
        "mechanism_preferences": [
            {
                "preferred_candidate": item.preferred.candidate_id,
                "preferred_primitive_sequence": list(item.preferred.primitive_sequence),
                "rejected_candidate": item.rejected.candidate_id,
                "rejected_primitive_sequence": list(item.rejected.primitive_sequence),
                "margin": item.margin,
            }
            for item in preferences
        ],
        "scientific_interpretation": {
            "learned_macro_candidate": "F1",
            "learned_macro_ids": ["V7M-001"],
            "learned_candidate_beat_no_library": False,
            "learned_candidate_beat_random_library": False,
            "causal_transfer_credit": False,
            "use_in_omega": "negative development preference only; not a causal-memory admission event"
        },
        "expert_target_timing_accessed": False,
        "expert_opt_commit_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
        "human_task_specific_solver_contribution": False
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_ranking": ranking,
        "preference_count": len(preferences),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
