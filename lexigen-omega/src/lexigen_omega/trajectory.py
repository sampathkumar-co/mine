from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluationObservation:
    task_id: str
    candidate_id: str
    artifact_fingerprint: str
    arm: str
    valid: bool
    score: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    credit_eligible: bool = True
    source: str = ""


@dataclass(frozen=True)
class HindsightPreference:
    task_id: str
    preferred_candidate: str
    rejected_candidate: str
    preferred_fingerprint: str
    rejected_fingerprint: str
    margin: float
    source: str


def ingest_v7_gso_preflight_result(
    payload: Mapping[str, Any],
    *,
    require_credit_eligible: bool = True,
) -> tuple[EvaluationObservation, ...]:
    """Convert an already-sealed V7 GSO result into Ω development observations.

    Ω does not rerun or reinterpret GSO here. The V7 harness remains the evaluator;
    this bridge only consumes its recorded outcomes. Diagnostic/contaminated tasks are
    excluded by default so they cannot silently become learning evidence.
    """

    if payload.get("stage") != "tasks2_6_preflight_r1":
        raise ValueError("unsupported V7 GSO stage")
    eligible = bool(payload.get("campaign_credit_eligible", False))
    if require_credit_eligible and not eligible:
        return ()

    task_id = str(payload.get("instance_id") or f"v7-task-{payload.get('task')}")
    observations: list[EvaluationObservation] = []
    for row in payload.get("candidate_results", []):
        candidate = str(row["candidate"])
        correct = bool(row.get("correct", False))
        harmonic = float(row.get("harmonic_speedup", 0.0) or 0.0)
        minimum = float(row.get("minimum_speedup", 0.0) or 0.0)
        fingerprint = str(row.get("patch_sha256") or f"unhashed:{candidate}")
        observations.append(
            EvaluationObservation(
                task_id=task_id,
                candidate_id=candidate,
                artifact_fingerprint=fingerprint,
                arm=str(row.get("arm", "unknown")),
                valid=correct,
                score=harmonic if correct else 0.0,
                metrics={
                    "harmonic_speedup": harmonic,
                    "minimum_speedup": minimum,
                    "tests_passed": float(row.get("tests_passed", 0) or 0),
                    "test_count": float(row.get("test_count", 0) or 0),
                },
                credit_eligible=eligible,
                source="lexigen-v7-gso-preflight-r1",
            )
        )
    return tuple(observations)


def _rank_key(observation: EvaluationObservation) -> tuple[int, float, float, str]:
    return (
        1 if observation.valid else 0,
        observation.score,
        float(observation.metrics.get("minimum_speedup", 0.0)),
        observation.candidate_id,
    )


def build_hindsight_preferences(
    observations: tuple[EvaluationObservation, ...],
    *,
    max_pairs: int = 64,
) -> tuple[HindsightPreference, ...]:
    """Create deterministic preference data from executable evaluator feedback.

    Correct candidates dominate incorrect candidates. Among correct candidates,
    evaluator score then minimum per-test score defines preference. This produces
    learning data without inventing labels or task-specific human judgements.
    """

    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    if not observations or max_pairs == 0:
        return ()
    task_ids = {item.task_id for item in observations}
    if len(task_ids) != 1:
        raise ValueError("preferences must be built one task at a time")
    if any(not item.credit_eligible for item in observations):
        raise ValueError("ineligible observations cannot enter hindsight learning")

    ranked = sorted(observations, key=_rank_key, reverse=True)
    pairs: list[HindsightPreference] = []
    for better_index, better in enumerate(ranked):
        for worse in ranked[better_index + 1 :]:
            if _rank_key(better) == _rank_key(worse):
                continue
            if better.valid and not worse.valid:
                margin = max(1.0, better.score)
            else:
                margin = max(0.0, better.score - worse.score)
            pairs.append(
                HindsightPreference(
                    task_id=better.task_id,
                    preferred_candidate=better.candidate_id,
                    rejected_candidate=worse.candidate_id,
                    preferred_fingerprint=better.artifact_fingerprint,
                    rejected_fingerprint=worse.artifact_fingerprint,
                    margin=margin,
                    source=better.source,
                )
            )
            if len(pairs) >= max_pairs:
                return tuple(pairs)
    return tuple(pairs)
