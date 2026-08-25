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
    """Convert an already-sealed V7 Tasks2-6 preflight result into Ω observations.

    Ω never reruns or reinterprets GSO here. Diagnostic/contaminated tasks,
    infrastructure incidents and partially evaluated candidate rows are excluded so
    they cannot silently become learning evidence.
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
        # Runner/install/apply/eqcheck exceptions are not automatically scientific
        # negatives. A later explicit taxonomy may distinguish them; until then they
        # stay out of hindsight learning.
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


def ingest_v7_gso_task1_revision_result(
    payload: Mapping[str, Any],
) -> tuple[EvaluationObservation, ...]:
    """Ingest the sealed equal-budget Task1 revision-slot feedback as dev evidence.

    This is *not* authoritative GSO success evidence and earns no causal credit by
    itself. It is safe hindsight data because all compared candidates were frozen
    before the feedback round, all arms had equal revision budget, all equivalence
    checks completed, and expert targets/diffs/hints remained unopened.
    """

    if payload.get("stage") != "revision_slot_1_14_test_result_sealed":
        raise ValueError("unsupported Task1 revision stage")
    if payload.get("status") != "success":
        return ()
    for forbidden in (
        "expert_target_timing_accessed",
        "expert_opt_commit_accessed",
        "expert_diff_accessed",
        "hints_accessed",
        "human_task_specific_solver_contribution",
    ):
        if bool(payload.get(forbidden, True)):
            raise ValueError(f"Task1 evidence crossed forbidden boundary: {forbidden}")
    if int(payload.get("revision_slots_consumed_per_arm", -1)) != 1:
        raise ValueError("Task1 revision did not preserve equal one-slot arm budget")

    task_id = str(payload.get("instance_id", ""))
    if not task_id:
        raise ValueError("Task1 evidence missing instance identity")

    observations: list[EvaluationObservation] = []
    candidates = payload.get("candidates", {})
    for arm in ("v7_full", "v7_no_library", "v7_random_library"):
        row = candidates.get(arm)
        if not isinstance(row, Mapping):
            raise ValueError(f"Task1 evidence missing arm {arm}")
        if not bool(row.get("all_14_equivalence_checks_passed", False)):
            continue
        candidate = str(row["candidate_id"])
        harmonic = float(row.get("harmonic_speedup_vs_base", 0.0) or 0.0)
        minimum = float(row.get("minimum_speedup_vs_base", 0.0) or 0.0)
        maximum = float(row.get("maximum_speedup_vs_base", 0.0) or 0.0)
        observations.append(
            EvaluationObservation(
                task_id=task_id,
                candidate_id=candidate,
                artifact_fingerprint=str(row["patch_sha256"]),
                arm=arm,
                valid=True,
                score=harmonic,
                metrics={
                    "harmonic_speedup": harmonic,
                    "minimum_speedup": minimum,
                    "maximum_speedup": maximum,
                    "tests_passed": 14.0,
                    "test_count": 14.0,
                },
                credit_eligible=True,
                source="lexigen-v7-gso-task1-revision1-sealed-dev-feedback",
            )
        )
    return tuple(observations)


def _common_proposal_boundary_ok(payload: Mapping[str, Any]) -> None:
    if bool(payload.get("timing_feedback_used", True)):
        raise ValueError("proposal provenance used timing feedback")

    # Tasks2-6 schema.
    if "schema" in payload:
        for forbidden in ("expert_opt_commit_accessed", "expert_diff_accessed", "hints_accessed"):
            if bool(payload.get(forbidden, True)):
                raise ValueError(f"proposal provenance crossed forbidden boundary: {forbidden}")
        return

    # Task1 nested-arm schema.
    if bool(payload.get("correctness_feedback_used", True)):
        raise ValueError("Task1 proposal provenance used correctness feedback")
    for forbidden in ("expert_patch_used", "hints_used"):
        if bool(payload.get(forbidden, True)):
            raise ValueError(f"Task1 proposal provenance crossed forbidden boundary: {forbidden}")


def parse_v7_gso_proposals(payload: Mapping[str, Any]) -> tuple[ProposalProvenance, ...]:
    """Parse either frozen V7 proposal schema into common mechanism provenance."""

    _common_proposal_boundary_ok(payload)
    records: list[ProposalProvenance] = []
    seen: set[str] = set()

    if "schema" in payload:
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
        rows = payload.get("proposals", ())
        for raw in rows:
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

    arms = payload.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("unsupported V7 proposal payload")
    for arm in ("v7_full", "v7_no_library", "v7_random_library"):
        rows = arms.get(arm)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"Task1 proposal payload missing arm {arm}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("malformed Task1 proposal row")
            candidate = str(row["candidate_id"])
            if candidate in seen:
                raise ValueError(f"duplicate proposal id: {candidate}")
            seen.add(candidate)
            preconditions = row.get("source_visible_preconditions", ())
            if isinstance(preconditions, str):
                precondition_text = preconditions
            else:
                precondition_text = "; ".join(str(x) for x in preconditions)
            records.append(
                ProposalProvenance(
                    candidate_id=candidate,
                    arm=arm,
                    primitive_sequence=tuple(str(x) for x in row.get("primitive_sequence", ())),
                    macro_ids=tuple(str(x) for x in row.get("macro_ids", ())),
                    mechanism=str(row.get("mechanism", "")),
                    source_visible_preconditions=precondition_text,
                    correctness_risk=str(row.get("correctness_risk", "")),
                    files_functions=tuple(str(x) for x in row.get("files", ())),
                    expected_performance_mechanism=str(row.get("expected_performance", "")),
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


def _outcome_key(observation: EvaluationObservation) -> tuple[int, float, float]:
    """Scientific outcome only; never include identifiers as preference evidence."""

    return (
        1 if observation.valid else 0,
        observation.score,
        float(observation.metrics.get("minimum_speedup", 0.0)),
    )


def _sort_key(observation: EvaluationObservation) -> tuple[int, float, float, str]:
    """Outcome ranking plus ID only for deterministic ordering of exact ties."""

    return (*_outcome_key(observation), observation.candidate_id)


def build_hindsight_preferences(
    observations: tuple[EvaluationObservation, ...],
    *,
    max_pairs: int = 64,
) -> tuple[HindsightPreference, ...]:
    """Create deterministic preference data from executable evaluator feedback."""

    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    if not observations or max_pairs == 0:
        return ()
    task_ids = {item.task_id for item in observations}
    if len(task_ids) != 1:
        raise ValueError("preferences must be built one task at a time")
    if any(not item.credit_eligible for item in observations):
        raise ValueError("ineligible observations cannot enter hindsight learning")

    ranked = sorted(observations, key=_sort_key, reverse=True)
    pairs: list[HindsightPreference] = []
    for better_index, better in enumerate(ranked):
        for worse in ranked[better_index + 1 :]:
            # Candidate IDs are allowed to make sorting reproducible, but an ID must
            # never manufacture a preference when executable outcomes are tied.
            if _outcome_key(better) == _outcome_key(worse):
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
