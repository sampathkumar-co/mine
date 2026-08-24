from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MIN_OBSERVED_COVERED = 4
MIN_COMPARABLE_PAIRS = 3
DIRECTIONAL_PAIRWISE_MIN = 0.70
DIRECTIONAL_SPEARMAN_MIN = 0.50
REQUIRE_TOP1_HIT = True


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _actual_key(row: dict[str, Any]) -> tuple[int, float, float]:
    correct = bool(row.get("correct", False))
    harmonic = float(row.get("harmonic_speedup", 0.0) or 0.0) if correct else 0.0
    minimum = float(row.get("minimum_speedup", 0.0) or 0.0) if correct else 0.0
    return (1 if correct else 0, harmonic, minimum)


def _average_ranks(values: list[float], *, descending: bool) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=descending)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
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
    if prediction.get("stage") != "task2_prospective_exact_sequence_transfer_r1":
        raise ValueError("unsupported prospective prediction")
    if prediction.get("status") != "frozen_before_Task2_outcome":
        raise ValueError("prediction was not prospectively frozen")
    if prediction.get("Task2_outcome_accessed") or prediction.get("Task2_timing_accessed"):
        raise ValueError("prediction lock claims Task2 outcome/timing access")
    if result.get("stage") != "tasks2_6_preflight_r1" or int(result.get("task", -1)) != 2:
        raise ValueError("unsupported Task2 result")
    if str(result.get("instance_id")) != str(prediction.get("target_task")):
        raise ValueError("target identity mismatch")
    if not bool(result.get("campaign_credit_eligible", False)):
        raise ValueError("Task2 result is not campaign-credit eligible")
    if str(result.get("status", "")).startswith("infrastructure_"):
        return {
            "status": "insufficient_evidence_infrastructure_failure",
            "observed_covered_candidates": 0,
            "prediction_stage": prediction["stage"],
        }

    actual = {
        str(row["candidate"]): row
        for row in result.get("candidate_results", ())
        if not row.get("error")
    }
    predicted_rows = list(prediction.get("prospective_ranking", ()))
    observed = [row for row in predicted_rows if row["candidate_id"] in actual]

    pairwise_total = 0
    pairwise_concordant = 0
    pairwise_discordant = 0
    for i, left in enumerate(observed):
        for right in observed[i + 1 :]:
            lkey = _actual_key(actual[left["candidate_id"]])
            rkey = _actual_key(actual[right["candidate_id"]])
            if lkey == rkey:
                continue
            pairwise_total += 1
            # `observed` preserves predicted best-to-worst order, so the pair is
            # concordant exactly when the left candidate actually beats the right.
            if lkey > rkey:
                pairwise_concordant += 1
            else:
                pairwise_discordant += 1

    concordance = (
        pairwise_concordant / pairwise_total if pairwise_total else None
    )

    predicted_scores = [float(row["transferred_prior_harmonic_speedup"]) for row in observed]
    actual_scores = [
        float(actual[row["candidate_id"]].get("harmonic_speedup", 0.0) or 0.0)
        if bool(actual[row["candidate_id"]].get("correct", False))
        else 0.0
        for row in observed
    ]
    predicted_ranks = _average_ranks(predicted_scores, descending=True)
    actual_ranks = _average_ranks(actual_scores, descending=True)
    spearman = _pearson(predicted_ranks, actual_ranks)

    top1_hit = None
    if observed:
        predicted_top = observed[0]["candidate_id"]
        best_actual_key = max(_actual_key(actual[row["candidate_id"]]) for row in observed)
        top1_hit = _actual_key(actual[predicted_top]) == best_actual_key

    if len(observed) < MIN_OBSERVED_COVERED or pairwise_total < MIN_COMPARABLE_PAIRS:
        verdict = "insufficient_clean_covered_evidence"
    elif (
        concordance is not None
        and concordance >= DIRECTIONAL_PAIRWISE_MIN
        and spearman is not None
        and spearman >= DIRECTIONAL_SPEARMAN_MIN
        and (top1_hit or not REQUIRE_TOP1_HIT)
    ):
        verdict = "sequence_only_transfer_directionally_supported_on_Task2"
    else:
        verdict = "sequence_only_transfer_not_supported_on_Task2"

    actual_order = sorted(
        observed,
        key=lambda row: (_actual_key(actual[row["candidate_id"]]), row["candidate_id"]),
        reverse=True,
    )

    return {
        "project": "LEXIGEN OMEGA",
        "stage": "task2_prospective_exact_sequence_transfer_score_r1",
        "status": verdict,
        "source_prediction_stage": prediction["stage"],
        "target_task": prediction["target_task"],
        "preregistered_scoring_policy": {
            "minimum_observed_covered_candidates": MIN_OBSERVED_COVERED,
            "minimum_comparable_pairs": MIN_COMPARABLE_PAIRS,
            "directional_pairwise_concordance_minimum": DIRECTIONAL_PAIRWISE_MIN,
            "directional_spearman_minimum": DIRECTIONAL_SPEARMAN_MIN,
            "require_top1_hit": REQUIRE_TOP1_HIT,
            "errored_or_infrastructure_rows": "excluded_not_negative",
            "actual_outcome_ties": "excluded_from_pairwise_denominator",
        },
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
        "claim_boundary": "One prospective target can falsify the exact-sequence baseline but cannot establish general cross-repository transfer. Positive evidence still requires additional prospective repositories and causal controls.",
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
