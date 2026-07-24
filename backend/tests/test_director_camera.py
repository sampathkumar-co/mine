from app.director.camera import (
    AcceptedPickup,
    PickupMissionSpec,
    audit_production_readiness,
    integrate_accepted_pickups,
    promote_accepted_pickup,
    validate_pickup_clip,
)
from app.director.edit_graph import EditSegment
from app.director.semantic_overlays import ProductionEditDecisionGraph
from app.sensory.models import (
    AnalysisBundle,
    ClipAnalysis,
    ContinuityProfile,
    SceneRange,
    SemanticTag,
    SubjectFraming,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)


def _transcript(text: str, duration: float = 6) -> TranscriptResult:
    words = text.split()
    step = duration / max(1, len(words))
    return TranscriptResult(
        text=text,
        provider="fixture",
        model="fixture",
        duration_seconds=duration,
        words=[
            TranscriptWord(
                word=word,
                start=index * step,
                end=(index + 0.8) * step,
            )
            for index, word in enumerate(words)
        ],
        segments=[TranscriptSegment(start=0, end=duration, text=text, confidence=0.95)],
    )


def _clip(
    asset_id: str,
    *,
    role: str = "primary_speech",
    text: str = "Why this workflow produces a better result",
    duration: float = 6,
    quality: float = 0.9,
    has_audio: bool | None = None,
    evidence_terms: list[str] | None = None,
    tags: list[SemanticTag] | None = None,
) -> ClipAnalysis:
    audio = role == "primary_speech" if has_audio is None else has_audio
    return ClipAnalysis(
        asset_id=asset_id,
        filename=f"{asset_id}.mp4",
        sha256=f"sha-{asset_id}",
        media={
            "duration_seconds": duration,
            "width": 1080,
            "height": 1920,
            "has_audio": audio,
        },
        transcript=_transcript(text, duration) if audio else None,
        scenes=[SceneRange(start=0, end=duration)],
        subject_framing=SubjectFraming(normalized_center_x=0.5, confidence=0.85),
        role=role,
        quality_score=quality,
        evidence_terms=evidence_terms or [],
        semantic_tags=tags or [],
        continuity=ContinuityProfile(
            brightness=0.52,
            saturation=0.5,
            motion_energy=0.25,
            subject_center_x=0.5,
        ),
    )


def _analysis(*clips: ClipAnalysis) -> AnalysisBundle:
    first = clips[0]
    return AnalysisBundle(
        media=first.media,
        transcript=first.transcript,
        scenes=first.scenes,
        subject_framing=first.subject_framing,
        source_clips=list(clips),
    )


def test_readiness_audit_requests_blocking_proof_pickup() -> None:
    speech = _clip(
        "speech",
        text="Why this process saves time and gives a better result",
        duration=16,
    )
    report = audit_production_readiness(
        _analysis(speech),
        {
            "objective": "Show dashboard proof of the result and ask viewers to book a call",
            "must_include": ["dashboard proof", "book a call"],
            "target_duration_seconds": 30,
        },
        threshold=0.72,
    )

    mission_types = {mission.mission_type for mission in report.missions}
    assert report.ready is False
    assert "evidence" in mission_types
    assert "cta" in mission_types
    assert any(mission.priority == "blocking" for mission in report.missions)
    assert any("proof" in reason.casefold() for reason in report.blocking_reasons)


def test_pickup_validation_accepts_matching_proof_and_rejects_duplicate() -> None:
    mission = PickupMissionSpec(
        mission_type="evidence",
        priority="blocking",
        title="Capture dashboard proof",
        reason="The claim requires proof",
        target_terms=["dashboard", "proof", "result"],
        minimum_duration_seconds=3,
        maximum_duration_seconds=12,
        requires_audio=False,
        capture_requirements={},
        insertion_strategy="overlay",
    )
    good = _clip(
        "pickup-proof",
        role="evidence",
        text="",
        duration=5,
        evidence_terms=["dashboard", "proof", "result"],
        tags=[
            SemanticTag(
                label="dashboard",
                confidence=0.95,
                source="filename",
                evidence="dashboard proof result",
            )
        ],
    )
    accepted = validate_pickup_clip(mission, good)
    assert accepted.accepted is True
    assert accepted.score >= 0.58
    assert {"dashboard", "proof", "result"} <= set(accepted.matched_terms)

    duplicate = good.model_copy(
        update={
            "asset_id": "pickup-duplicate",
            "duplicate_of_asset_id": "existing-proof",
            "quality_score": 0.2,
        }
    )
    rejected = validate_pickup_clip(mission, duplicate)
    assert rejected.accepted is False
    assert any("duplicates" in reason for reason in rejected.blocking_reasons)


def test_accepted_pickup_is_promoted_to_requested_editorial_role() -> None:
    raw = _clip("pickup", role="b_roll", text="", evidence_terms=["screen"])
    pickup = AcceptedPickup(
        asset_id="pickup",
        mission_type="evidence",
        target_terms=["dashboard", "proof"],
        insertion_strategy="overlay",
        validation_score=0.9,
    )

    promoted = promote_accepted_pickup(raw, pickup)

    assert promoted.role == "evidence"
    assert {"dashboard", "proof", "screen"} <= set(promoted.evidence_terms)
    assert any(tag.label == "director_camera_evidence" for tag in promoted.semantic_tags)


def test_accepted_hook_and_evidence_are_inserted_into_graph() -> None:
    speech = _clip("speech", duration=8)
    hook = _clip(
        "pickup-hook",
        text="Stop wasting time on the wrong workflow",
        duration=4,
    )
    proof = _clip(
        "pickup-proof",
        role="evidence",
        text="",
        duration=5,
        evidence_terms=["workflow", "result"],
    )
    analysis = _analysis(speech, hook, proof)
    graph = ProductionEditDecisionGraph(
        target_duration_seconds=12,
        selected_duration_seconds=8,
        segments=[
            EditSegment(
                source_asset_id="speech",
                source_index=0,
                source_start=0,
                source_end=8,
                output_start=0,
                output_end=8,
                score=0.8,
                confidence=0.9,
                reason="Existing narration",
                transcript_text="Why this workflow produces a better result",
            )
        ],
    )

    revised = integrate_accepted_pickups(
        graph,
        analysis,
        [
            AcceptedPickup(
                asset_id="pickup-hook",
                mission_type="hook",
                target_terms=["workflow"],
                insertion_strategy="prepend",
                validation_score=0.92,
            ),
            AcceptedPickup(
                asset_id="pickup-proof",
                mission_type="evidence",
                target_terms=["workflow", "result"],
                insertion_strategy="overlay",
                validation_score=0.88,
            ),
        ],
    )

    assert revised.segments[0].source_asset_id == "pickup-hook"
    assert revised.segments[1].output_start == revised.segments[0].output_end
    assert any(overlay.source_asset_id == "pickup-proof" for overlay in revised.overlays)
    assert revised.selected_duration_seconds > graph.selected_duration_seconds
