from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import fvd_allocator_r1 as alloc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def allocation_for(artifact: dict, task: dict, arm: str, budget: int = 16) -> dict:
    result = alloc.run_allocator(artifact, task, arm, budget)
    require(result["budget_used"] == budget, f"budget mismatch for {arm}")
    require(sum(result["allocation"].values()) == budget, f"allocation sum mismatch for {arm}")
    require(result["official_holdout_data_accessed"] is False, f"holdout boundary crossed for {arm}")
    require(result["scientific_transfer_evidence"] is False, f"development allocation mislabeled evidence for {arm}")
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as td_raw:
        td = Path(td_raw)
        artifact_path = td / "experience.json"
        artifact_path_2 = td / "experience2.json"
        subprocess.run(
            [sys.executable, str(HERE / "build_fvd_experience_r1.py"), "--output", str(artifact_path)],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(HERE / "build_fvd_experience_r1.py"), "--output", str(artifact_path_2)],
            check=True,
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact2 = json.loads(artifact_path_2.read_text(encoding="utf-8"))

    require(artifact == artifact2, "experience build is not deterministic")
    require(artifact["artifact_sha256"] == artifact2["artifact_sha256"], "artifact hash is not deterministic")
    require(artifact["status"] == "development_seed_artifact_not_final_holdout_artifact", "bad artifact status")
    require(artifact["official_holdout_data_accessed"] is False, "official holdout boundary crossed")
    require(artifact["blocked_ipwm_real_intervention_data_accessed"] is False, "blocked IPWM corpus boundary crossed")
    require(artifact["scientific_transfer_evidence_created_by_this_build"] is False, "development build cannot create transfer evidence")
    require(len(artifact["proposal_profiles"]) == 8, "unexpected proposal profile count")

    confirmed = [p for p in artifact["proposal_profiles"] if p["evidence_grade"] == "confirmed_causal"]
    require(len(confirmed) == 1, f"expected exactly one historically confirmed causal profile, got {len(confirmed)}")
    require(confirmed[0]["proposal_class_id"] == "PC-FRONTIER-CERT", "historical causal credit assigned to wrong class")
    require(confirmed[0]["confirmed_causal_successes"] == 1, "V5 causal count changed")
    require(artifact["historical_summary"]["v7_confirmed_causal_transfer_wins"] == 0, "V7 failure was incorrectly promoted to positive causal evidence")

    guards = artifact["global_guards"]
    require(guards["semantic_equivalence_dedup_required"] is True, "equivalence guard absent")
    require(guards["performance_is_not_causality"] is True, "causal guard absent")
    require(guards["predictive_signal_is_not_transfer"] is True, "R3 negative lesson absent")
    require(guards["strict_gate_exit_status_required"] is True, "strict gate guard absent")

    frontier_task = {
        "task_descriptor_id": "development-canary-frontier",
        "traits": [
            "discrete",
            "set_or_boolean_structure",
            "sparse_or_frontier_search",
            "exact_or_structural_certificate",
            "fallback_available",
        ],
    }
    arms = ["fvd_full", "fvd_no_experience", "fvd_shuffled_experience", "retrieval_only", "evolution_only"]
    frontier = {arm: allocation_for(artifact, frontier_task, arm) for arm in arms}
    fid = "PC-FRONTIER-CERT"
    require(
        frontier["fvd_full"]["allocation"][fid] > frontier["fvd_no_experience"]["allocation"][fid],
        "confirmed causal frontier experience did not increase frontier budget",
    )
    require(
        frontier["fvd_full"]["experience_view_sha256"] != frontier["fvd_no_experience"]["experience_view_sha256"],
        "full and no-experience views are not physically distinct",
    )
    require(
        frontier["fvd_full"]["experience_view_sha256"] != frontier["fvd_shuffled_experience"]["experience_view_sha256"],
        "full and shuffled-experience views are not physically distinct",
    )
    require(
        frontier["fvd_full"]["allocation"] != frontier["fvd_shuffled_experience"]["allocation"],
        "shuffled experience did not change allocation on the frontier canary",
    )

    precision_risk_task = {
        "task_descriptor_id": "development-canary-long-horizon-precision-risk",
        "traits": [
            "numeric_array",
            "approximate_verifier",
            "native_backend_available",
            "precision_change_possible",
            "iterative_or_dynamical_long_horizon",
        ],
    }
    risk_full = allocation_for(artifact, precision_risk_task, "fvd_full")
    risk_retrieval = allocation_for(artifact, precision_risk_task, "retrieval_only")
    pid = "PC-PRECISION-BACKEND"
    require(
        risk_full["allocation"][pid] < risk_retrieval["allocation"][pid],
        "negative long-horizon precision lesson did not reduce budget relative to retrieval-only",
    )
    full_precision_detail = next(x for x in risk_full["ranking"] if x["proposal_class_id"] == pid)
    require(
        "iterative_or_dynamical_long_horizon" in full_precision_detail["risk_hits"],
        "precision risk tag was not activated",
    )

    unknown_task = {"task_descriptor_id": "development-canary-unknown", "traits": ["novel_unmatched_trait"]}
    unknown_full = allocation_for(artifact, unknown_task, "fvd_full")
    require(len(unknown_full["allocation"]) == 8, "unknown task lost proposal classes")
    require(sum(1 for n in unknown_full["allocation"].values() if n > 0) >= 7, "unknown task exploration collapsed too aggressively")

    print(json.dumps({
        "fvd_engine_dev_r1_passed": True,
        "scientific_transfer_evidence": False,
        "official_holdout_data_accessed": False,
        "experience_artifact_sha256": artifact["artifact_sha256"],
        "frontier_full_allocation": frontier["fvd_full"]["allocation"],
        "frontier_no_experience_allocation": frontier["fvd_no_experience"]["allocation"],
        "frontier_shuffled_allocation": frontier["fvd_shuffled_experience"]["allocation"],
        "precision_risk_full_allocation": risk_full["allocation"],
        "precision_risk_retrieval_allocation": risk_retrieval["allocation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
