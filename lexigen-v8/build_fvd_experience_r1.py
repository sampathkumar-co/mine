from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MANIFEST_PATH = HERE / "FVD_HISTORICAL_SOURCE_MANIFEST_R1.json"
CLASSES_PATH = HERE / "FVD_PROPOSAL_CLASSES_R1.json"
TRANSFER_PATH = ROOT / "lexigen-v5" / "TRANSFER_MEMORY.json"
V5_AUDIT_PATH = ROOT / "lexigen-v5" / "CAMPAIGN_FINAL_AUDIT_R1.json"
V5_TASK1_RESULT_PATH = ROOT / "lexigen-v5" / "tasks" / "01-clustering-outliers" / "TASK_RESULT.json"
V7_AUDIT_PATH = ROOT / "lexigen-v7-gso" / "FINAL_CAMPAIGN_AUDIT_R1.json"
R3_RESULT_PATH = HERE / "COMPOSITIONAL_R3_FAILED_RESULT.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for src in manifest["sources"]:
        path = ROOT / src["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        got = git_blob_sha1(path)
        expected = str(src["git_blob_sha1"])
        if got != expected:
            raise RuntimeError(f"historical source hash mismatch for {src['id']}: {got} != {expected}")
        verified[str(src["id"])] = got
    boundary_flags = (
        "official_new_fvd_holdout_inventory_accessed",
        "blocked_ipwm_real_intervention_corpus_accessed",
        "expert_solution_payloads_imported",
        "future_holdout_task_payloads_embedded",
        "human_task_specific_solver_content",
    )
    for key in boundary_flags:
        if manifest.get(key) is not False:
            raise RuntimeError(f"historical-source boundary is not closed: {key}")
    return verified


def make_profiles(
    classes: dict[str, Any],
    transfer: dict[str, Any],
    v5_audit: dict[str, Any],
    verified: dict[str, str],
    excluded_evidence_tasks: set[str],
) -> list[dict[str, Any]]:
    learned = transfer["learned_templates"]
    causal_wins = [
        row
        for row in v5_audit.get("causal_transfer_wins", [])
        if str(row.get("task", "")) not in excluded_evidence_tasks
    ]
    causal_count: dict[str, int] = {}
    causal_speedups: dict[str, list[float]] = {}
    target_families: dict[str, set[str]] = {}
    for row in causal_wins:
        cid = str(row["learned_causal_id"])
        causal_count[cid] = causal_count.get(cid, 0) + 1
        causal_speedups.setdefault(cid, []).append(float(row["blind_harmonic_speedup"]))
        target_families.setdefault(cid, set()).add(str(row["current_family"]))

    profiles: list[dict[str, Any]] = []
    for cls in classes["proposal_classes"]:
        profile: dict[str, Any] = {
            "proposal_class_id": cls["proposal_class_id"],
            "label": cls["label"],
            "mechanism_sequence": list(cls["mechanism_sequence"]),
            "context_tags": sorted(set(cls["context_tags"])),
            "evidence_grade": "generic_baseline",
            "source_families": [],
            "source_causal_ids": [],
            "confirmed_causal_successes": 0,
            "confirmed_causal_target_families": [],
            "confirmed_causal_harmonic_speedups": [],
            "performance_only_successes": 0,
            "risk_tags": [],
            "provenance": [],
        }
        template_name = cls.get("historical_template")
        if template_name:
            if template_name not in learned:
                raise RuntimeError(f"proposal class references missing historical template: {template_name}")
            tm = learned[template_name]
            cid = str(tm["causal_id"])
            count = int(causal_count.get(cid, 0))
            profile["evidence_grade"] = "confirmed_causal" if count else "source_learned_unconfirmed"
            profile["source_families"] = [str(tm["learned_from_family"])]
            profile["source_causal_ids"] = [cid]
            profile["confirmed_causal_successes"] = count
            profile["confirmed_causal_target_families"] = sorted(target_families.get(cid, set()))
            profile["confirmed_causal_harmonic_speedups"] = causal_speedups.get(cid, [])
            profile["abstract_recipe"] = list(tm["recipe"])
            profile["provenance"] = [
                {"source_id": "v5_transfer_memory", "git_blob_sha1": verified["v5_transfer_memory"]},
                {"source_id": "v5_final_audit", "git_blob_sha1": verified["v5_final_audit"]},
            ]
            if cid == "TM-PBEB-01":
                profile["risk_tags"].append("iterative_or_dynamical_long_horizon")
        profiles.append(profile)
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-evidence-task",
        action="append",
        default=[],
        help="Historical task name whose outcome-derived positive evidence must be removed from this build.",
    )
    args = parser.parse_args()
    excluded_evidence_tasks = {str(x) for x in args.exclude_evidence_task}

    manifest = load_json(MANIFEST_PATH)
    classes = load_json(CLASSES_PATH)
    transfer = load_json(TRANSFER_PATH)
    v5_audit = load_json(V5_AUDIT_PATH)
    v5_task1 = load_json(V5_TASK1_RESULT_PATH)
    v7_audit = load_json(V7_AUDIT_PATH)
    r3 = load_json(R3_RESULT_PATH)
    verified = verify_manifest(manifest)

    profiles = make_profiles(classes, transfer, v5_audit, verified, excluded_evidence_tasks)

    v7_task_ledger = v7_audit.get("task_ledger", {})
    v7_causal_wins = sum(1 for row in v7_task_ledger.values() if row.get("causal_transfer_win") is True)
    v7_perf_positive_causal_negative = sum(
        1
        for row in v7_task_ledger.values()
        if (row.get("clean_gso_performance_win") is True or float(row.get("best_full_preflight_harmonic_speedup", 0.0) or 0.0) > 1.0)
        and row.get("causal_transfer_win") is False
    )
    v7_ineligible = sum(1 for row in v7_task_ledger.values() if row.get("campaign_credit_eligible") is False)

    negative_lessons = []
    for lesson_id, lesson in transfer.get("negative_lessons", {}).items():
        negative_lessons.append({
            "lesson_id": lesson_id,
            "causal_id": lesson.get("causal_id"),
            "source_family": lesson.get("learned_from_family"),
            "source_tasks": lesson.get("source_evidence_tasks") or ([lesson["source_evidence_task"]] if lesson.get("source_evidence_task") else []),
            "rule": lesson["rule"],
            "provenance": [{"source_id": "v5_transfer_memory", "git_blob_sha1": verified["v5_transfer_memory"]}],
        })

    global_guards = {
        "semantic_equivalence_dedup_required": True,
        "performance_is_not_causality": True,
        "contaminated_or_diagnostic_only_records_excluded_from_positive_learning": True,
        "predictive_signal_is_not_transfer": True,
        "strict_gate_exit_status_required": True,
        "evidence": {
            "v5_task1_clean_performance_win_but_no_causal_transfer": None
            if "clustering_outliers" in excluded_evidence_tasks
            else bool(v5_task1.get("v5_full_passed_blind_gate") and not v5_task1.get("causal_transfer_win")),
            "v7_campaign_status": v7_audit.get("status"),
            "v7_confirmed_causal_wins": v7_causal_wins,
            "v7_performance_positive_causal_negative_count": v7_perf_positive_causal_negative,
            "v7_campaign_credit_ineligible_count": v7_ineligible,
            "r3_classification": r3.get("classification"),
            "r3_repository_predictive_signal_passed": bool(r3.get("passed_gate_groups", {}).get("repository_holdout_predictive_signal")),
            "r3_scientific_transfer_evidence": bool(r3.get("scientific_transfer_evidence")),
            "r3_workflow_status_caveat_present": bool(r3.get("workflow_status_caveat")),
        },
        "provenance": [
            {"source_id": "v5_task1_result", "git_blob_sha1": verified["v5_task1_result"]},
            {"source_id": "v7_final_audit", "git_blob_sha1": verified["v7_final_audit"]},
            {"source_id": "v8_ipwm_r3_failed_result", "git_blob_sha1": verified["v8_ipwm_r3_failed_result"]},
        ],
    }

    artifact: dict[str, Any] = {
        "schema": "lexigen-v8-fvd-experience-r1",
        "builder_version": "build_fvd_experience_r1.py:r1",
        "status": "development_seed_artifact_not_final_holdout_artifact",
        "source_manifest": {
            "manifest_id": manifest["manifest"],
            "verified_git_blob_sha1": dict(sorted(verified.items())),
        },
        "proposal_profiles": profiles,
        "global_guards": global_guards,
        "negative_lessons": negative_lessons,
        "historical_summary": {
            "v5_campaign_status": v5_audit.get("status"),
            "v5_clean_wins": int(v5_audit.get("task_win_counts", {}).get("v5_full", 0)),
            "v5_confirmed_causal_transfer_wins": len(v5_audit.get("causal_transfer_wins", [])),
            "v7_campaign_status": v7_audit.get("status"),
            "v7_confirmed_causal_transfer_wins": v7_causal_wins,
            "r3_classification": r3.get("classification"),
        },
        "official_holdout_data_accessed": False,
        "blocked_ipwm_real_intervention_data_accessed": False,
        "scientific_transfer_evidence_created_by_this_build": False,
        "artifact_sha256": "",
    }
    if excluded_evidence_tasks:
        artifact["development_leave_one_out"] = {
            "excluded_evidence_tasks": sorted(excluded_evidence_tasks),
            "claim_boundary": "Used only for historical development replay; target outcome-derived positive evidence is excluded before allocation.",
        }
    artifact_for_hash = dict(artifact)
    artifact_for_hash["artifact_sha256"] = ""
    artifact["artifact_sha256"] = hashlib.sha256(canonical_bytes(artifact_for_hash)).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact_sha256": artifact["artifact_sha256"],
        "proposal_profile_count": len(profiles),
        "negative_lesson_count": len(negative_lessons),
        "excluded_evidence_tasks": sorted(excluded_evidence_tasks),
        "v5_confirmed_causal_transfer_wins": artifact["historical_summary"]["v5_confirmed_causal_transfer_wins"],
        "v7_confirmed_causal_transfer_wins": artifact["historical_summary"]["v7_confirmed_causal_transfer_wins"],
        "official_holdout_data_accessed": False,
        "scientific_transfer_evidence_created_by_this_build": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
