from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


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


@dataclass(frozen=True)
class ProposalProvenance:
    candidate_id: str
    arm: str
    primitive_sequence: tuple[str, ...]
    macro_ids: tuple[str, ...]
    mechanism: str
    source_visible_preconditions: str
    correctness_risk: str
    files_functions: tuple[str, ...]
    expected_performance_mechanism: str


@dataclass(frozen=True)
class MechanismObservation:
    evaluation: EvaluationObservation
    provenance: ProposalProvenance


@dataclass(frozen=True)
class MechanismPreference:
    task_id: str
    preferred: ProposalProvenance
    rejected: ProposalProvenance
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

    Infrastructure incidents and partially evaluated rows are intentionally excluded.
    They are useful engineering evidence, but must not teach Ω that a mechanism is
    scientifically bad. Only candidate rows that completed without runner errors enter
    hindsight learning.
    """

    if payload.get("stage") != "tasks2_6_preflight_r1":
        raise ValueError("unsupported V7 GSO stage")
    eligible = bool(payload.get("campaign_credit_eligible", False))
    if require_credit_eligible and not eligible:
        return ()

    status = str(payload.get("status", ""))
    if status.startswith("infrastructure_"):
        return ()

    task_id = str(payload.get("instance_id") or f"v7-task-{payload.get('task')}")
    observations: list[EvaluationObservation] = []
    for row in payload.get("candidate_results", []):
        # A row with a runner/install/apply/eqcheck exception has not completed the
        # same evaluation path as a successful row. Keep it out of learning until a
        # future explicit failure taxonomy can prove it is a scientific negative.
        if row.get("error"):
            continue
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


def parse_v7_gso_proposals(payload: Mapping[str, Any]) -> tuple[ProposalProvenance, ...]:
    """Parse the frozen pre-timing V7 proposal pool into mechanism provenance.

    The proposal file is intentionally accepted only when it asserts no timing or
    expert leakage. This prevents post-result descriptions from becoming Ω training
    labels under the guise of provenance.
    """

    if bool(payload.get("timing_feedback_used", True)):
        raise ValueError("proposal provenance used timing feedback")
    for forbidden in ("expert_opt_commit_accessed", "expert_diff_accessed", "hints_accessed"):
        if bool(payload.get(forbidden, True)):
            raise ValueError(f"proposal provenance crossed forbidden boundary: {forbidden}")

    schema = tuple(str(x) for x in payload.get("schema", ()))
    required = (
        "proposal_id",
        "arm",
        "primitive_sequence",
        "macro_ids",
        "mechanism",
        "source_visible_preconditions",
        "correctness_risk",
        "files_functions",
        "expected_performance_mechanism",
    )
    if schema != required:
        raise ValueError("unsupported V7 proposal schema")

    records: list[ProposalProvenance] = []
    seen: set[str] = set()
    for raw in payload.get("proposals", ()):
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != len(required):
            raise ValueError("malformed proposal row")
        candidate = str(raw[0])
        if candidate in seen:
            raise ValueError(f"duplicate proposal id: {candidate}")
        seen.add(candidate)
        records.append(
            ProposalProvenance(
                candidate_id=candidate,
                arm=str(raw[1]),
                primitive_sequence=tuple(str(x) for x in raw[2]),
                macro_ids=tuple(str(x) for x in raw[3]),
                mechanism=str(raw[4]),
                source_visible_preconditions=str(raw[5]),
                correctness_risk=str(raw[6]),
                files_functions=tuple(str(x) for x in raw[7]),
                expected_performance_mechanism=str(raw[8]),
            )
        )
    return tuple(records)


def attach_proposal_provenance(
    observations: tuple[EvaluationObservation, ...],
    proposals: tuple[ProposalProvenance, ...],
) -> tuple[MechanismObservation, ...]:
    """Join evaluator outcomes to mechanisms by frozen candidate ID and arm."""

    lookup = {item.candidate_id: item for item in proposals}
    joined: list[MechanismObservation] = []
    for observation in observations:
        provenance = lookup.get(observation.candidate_id)
        if provenance is None:
            raise ValueError(f"missing frozen proposal provenance for {observation.candidate_id}")
        if provenance.arm != observation.arm:
            raise ValueError(f"arm mismatch for {observation.candidate_id}")
        joined.append(MechanismObservation(observation, provenance))
    return tuple(joined)


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


def build_mechanism_preferences(
    observations: tuple[MechanismObservation, ...],
    *,
    max_pairs: int = 64,
) -> tuple[MechanismPreference, ...]:
    """Convert outcome preferences into mechanism-level learning records."""

    if not observations:
        return ()
    by_candidate = {item.evaluation.candidate_id: item.provenance for item in observations}
    raw = build_hindsight_preferences(
        tuple(item.evaluation for item in observations),
        max_pairs=max_pairs,
    )
    return tuple(
        MechanismPreference(
            task_id=item.task_id,
            preferred=by_candidate[item.preferred_candidate],
            rejected=by_candidate[item.rejected_candidate],
            margin=item.margin,
            source=item.source,
        )
        for item in raw
    )
