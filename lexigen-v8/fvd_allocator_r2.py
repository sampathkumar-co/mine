from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fvd_r2_common import (
    allocate,
    canonical_sha256,
    context_coverage,
    predicted_success_probability,
    reachability_factor,
    shuffled_ledger,
    utility_multiplier,
)

VALID_ARMS = {
    "fvd_full",
    "fvd_no_experience",
    "fvd_shuffled_experience",
    "retrieval_only",
    "evolution_only",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def run_allocator(
    proposal_classes: dict[str, Any],
    controller: dict[str, Any],
    ledger: dict[str, Any],
    task: dict[str, Any],
    arm: str,
    budget: int,
) -> dict[str, Any]:
    if arm not in VALID_ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    if task.get("partition") != "calibration":
        raise RuntimeError("R2 calibration allocator accepts calibration source descriptors only")
    if ledger.get("calibration_outcomes_loaded") is not False:
        raise RuntimeError("ledger crossed calibration outcome boundary")
    if task.get("official_fvd_holdouts_accessed") is True:
        raise RuntimeError("official holdout descriptor is forbidden")

    profiles = [dict(row) for row in proposal_classes["proposal_classes"]]
    class_ids = [str(row["proposal_class_id"]) for row in profiles]
    active_ledger = shuffled_ledger(ledger, class_ids) if arm == "fvd_shuffled_experience" else ledger
    traits = set(map(str, task.get("traits", [])))

    details: list[dict[str, Any]] = []
    scores: list[tuple[str, float]] = []
    for profile in profiles:
        pid = str(profile["proposal_class_id"])
        coverage = context_coverage(list(profile.get("context_tags", [])), traits)
        context_score = 0.20 + 2.00 * coverage
        reach, reach_detail = reachability_factor(pid, traits, controller)
        probability: float | None = None
        multiplier: float | None = None
        neighbors: list[dict[str, Any]] = []

        if arm == "evolution_only":
            score = 1.0
        elif arm == "fvd_no_experience":
            score = reach
        elif arm == "retrieval_only":
            score = context_score * reach
        else:
            probability, neighbors = predicted_success_probability(pid, traits, active_ledger, controller)
            multiplier = utility_multiplier(probability, controller)
            score = context_score * reach * multiplier

        score = max(0.01, float(score))
        scores.append((pid, score))
        details.append({
            "proposal_class_id": pid,
            "score": score,
            "context_coverage": coverage,
            "context_score": context_score,
            "reachability_factor": reach,
            "reachability": reach_detail,
            "predicted_success_probability": probability,
            "utility_multiplier": multiplier,
            "outcome_neighbors": neighbors,
        })

    counts = allocate(scores, budget)
    ranking = sorted(details, key=lambda row: (-float(row["score"]), str(row["proposal_class_id"])))
    experience_view = {
        "arm": arm,
        "ledger_sha256": active_ledger.get("artifact_sha256") if arm not in {"fvd_no_experience", "retrieval_only", "evolution_only"} else None,
        "task": task["task"],
        "traits": sorted(traits),
    }
    return {
        "schema": "lexigen-v8-fvd-r2-allocation-r1",
        "arm": arm,
        "task": task["task"],
        "task_traits": sorted(traits),
        "proposal_budget": budget,
        "budget_used": sum(counts.values()),
        "allocation": dict(sorted(counts.items())),
        "ranking": ranking,
        "source_descriptor_sha256": task.get("descriptor_sha256"),
        "apprenticeship_ledger_sha256": ledger.get("artifact_sha256"),
        "experience_view_sha256": canonical_sha256(experience_view),
        "calibration_outcomes_loaded": False,
        "official_fvd_holdouts_accessed": False,
        "scientific_transfer_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-classes", type=Path, default=Path("lexigen-v8/FVD_PROPOSAL_CLASSES_R1.json"))
    parser.add_argument("--controller", type=Path, default=Path("lexigen-v8/FVD_R2_CONTROLLER_SPEC.json"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_allocator(
        load_json(args.proposal_classes),
        load_json(args.controller),
        load_json(args.ledger),
        load_json(args.task),
        args.arm,
        args.budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
