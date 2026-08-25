from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_POLICY = {
    "primary": "pairwise_concordance",
    "minimum_observed_covered_candidates": 4,
    "minimum_comparable_pairs": 3,
    "directional_pairwise_concordance_minimum": 0.70,
    "directional_spearman_minimum": 0.50,
    "require_top1_hit": True,
    "errored_or_infrastructure_rows": "excluded_not_negative",
    "actual_outcome_ties": "excluded_from_pairwise_denominator",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def actual_key(row: dict[str, Any]) -> tuple[int, float, float]:
    correct = bool(row.get("correct", False))
    harmonic = float(row.get("harmonic_speedup", 0.0) or 0.0) if correct else 0.0
    minimum = float(row.get("minimum_speedup", 0.0) or 0.0) if correct else 0.0
    return 1 if correct else 0, harmonic, minimum


def average_ranks(values: list[float], *, descending: bool) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=descending)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for pos in range(cursor, end):
            ranks[indexed[pos][0]] = average
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm = sum(left) / len(left)
    rm = sum(right) / len(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    ld = math.sqrt(sum((a - lm) ** 2 for a in left))
    rd = math.sqrt(sum((b - rm) ** 2 for b in right))
    if ld == 0 or rd == 0:
        return None
    return numerator / (ld * rd)


def score_prediction(prediction: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if prediction.get("status") != "frozen_before_R5_performance_spec_timing_or_outcome":
        raise ValueError("prediction is not an R5 prospective freeze")
    if prediction.get("R5_performance_spec_accessed") or prediction.get("R5_outcome_accessed") or prediction.get("R5_timing_accessed"):
        raise ValueError("prediction freeze records forbidden R5 access")
    if prediction.get("R5_ground_truth_message_or_diff_accessed"):
        raise ValueError("prediction freeze records ground-truth contamination")
    policy = prediction.get("scoring_policy")
    if policy != EXPECTED_POLICY:
        raise ValueError(f"scoring policy changed: {policy!r}")
    if str(result.get("instance_id")) != str(prediction.get("target_task")):
        raise ValueError("target identity mismatch")
    if str(result.get("status", "")).startswith("infrastructure_"):
        return {
            "project": "LEXIGEN OMEGA",
            "stage": "prospective_exact_sequence_transfer_score_R5",
            "status": "insufficient_evidence_infrastructure_failure",
            "target_task": prediction["target_task"],
            "observed_covered_candidates": 0,
            "gate_credit": False,
        }
    if not bool(result.get("scientific_evidence_eligible", False)):
        raise ValueError("result is not marked scientific-evidence eligible")

    actual = {
        str(row["candidate"]): row
        for row in result.get("candidate_results", ())
        if not row.get("error")
    }
    predicted = list(prediction.get("prospective_ranking", ()))
    observed = [row for row in predicted if row["candidate_id"] in actual]

    pairwise_total = pairwise_concordant = pairwise_discordant = 0
    for i, left in enumerate(observed):
        for right in observed[i + 1 :]:
            lk = actual_key(actual[left["candidate_id"]])
            rk = actual_key(actual[right["candidate_id"]])
            if lk == rk:
                continue
            pairwise_total += 1
            if lk > rk:
                pairwise_concordant += 1
            else:
                pairwise_discordant += 1
    concordance = pairwise_concordant / pairwise_total if pairwise_total else None

    predicted_scores = [float(row["transferred_prior_harmonic_speedup"]) for row in observed]
    actual_scores = [
        float(actual[row["candidate_id"]].get("harmonic_speedup", 0.0) or 0.0)
        if bool(actual[row["candidate_id"]].get("correct", False)) else 0.0
        for row in observed
    ]
    spearman = pearson(
        average_ranks(predicted_scores, descending=True),
        average_ranks(actual_scores, descending=True),
    )

    top1_hit = None
    if observed:
        predicted_top = observed[0]["candidate_id"]
        best = max(actual_key(actual[row["candidate_id"]]) for row in observed)
        top1_hit = actual_key(actual[predicted_top]) == best

    if len(observed) < policy["minimum_observed_covered_candidates"] or pairwise_total < policy["minimum_comparable_pairs"]:
        verdict = "insufficient_clean_covered_evidence"
    elif (
        concordance is not None
        and concordance >= policy["directional_pairwise_concordance_minimum"]
        and spearman is not None
        and spearman >= policy["directional_spearman_minimum"]
        and (bool(top1_hit) or not policy["require_top1_hit"])
    ):
        verdict = "sequence_only_transfer_directionally_supported"
    else:
        verdict = "sequence_only_transfer_not_supported"

    actual_order = sorted(
        observed,
        key=lambda row: (actual_key(actual[row["candidate_id"]]), row["candidate_id"]),
        reverse=True,
    )
    return {
        "project": "LEXIGEN OMEGA",
        "stage": "prospective_exact_sequence_transfer_score_R5",
        "status": verdict,
        "source_prediction_stage": prediction["stage"],
        "target_task": prediction["target_task"],
        "preregistered_scoring_policy": policy,
        "observed_covered_candidates": len(observed),
        "covered_candidate_ids_observed": [row["candidate_id"] for row in observed],
        "pairwise_comparable_pairs": pairwise_total,
        "pairwise_concordant": pairwise_concordant,
        "pairwise_discordant": pairwise_discordant,
        "pairwise_concordance": concordance,
        "spearman_rank_correlation": spearman,
        "predicted_top_candidate": observed[0]["candidate_id"] if observed else None,
        "top1_hit": top1_hit,
        "actual_order_over_observed_covered": [row["candidate_id"] for row in actual_order],
        "gate_credit": False,
        "claim_boundary": "This is prospective-development replication evidence only. Even a positive result does not establish general or causal transfer, AGI, or a breakthrough.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    score = score_prediction(load(args.prediction), load(args.result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(score, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
