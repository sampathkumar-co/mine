from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.director.edit_graph import EditSegment
from app.director.semantic_overlays import ProductionEditDecisionGraph, VisualOverlay
from app.sensory.models import AnalysisBundle, ClipAnalysis, SemanticTag
from app.sensory.semantics import continuity_similarity, semantic_terms

MissionType = Literal["hook", "cta", "evidence", "b_roll", "audio_retake", "continuity_retake"]
MissionPriority = Literal["blocking", "high", "medium", "optional"]
MissionStatus = Literal["requested", "submitted", "accepted", "rejected", "cancelled"]

WORD_RE = re.compile(r"[a-z0-9']+")
HOOK_TERMS = {"avoid", "biggest", "mistake", "never", "secret", "stop", "why", "how"}
CTA_TERMS = {
    "book",
    "buy",
    "call",
    "click",
    "comment",
    "contact",
    "download",
    "follow",
    "join",
    "message",
    "save",
    "share",
    "subscribe",
    "visit",
}
EVIDENCE_TERMS = {
    "before",
    "after",
    "chart",
    "dashboard",
    "demo",
    "evidence",
    "metrics",
    "product",
    "proof",
    "result",
    "screen",
    "testimonial",
}


class ReadinessDimension(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    blocking: bool = False
    findings: list[str] = Field(default_factory=list)


class PickupMissionSpec(BaseModel):
    mission_type: MissionType
    priority: MissionPriority
    title: str
    reason: str
    target_terms: list[str] = Field(default_factory=list)
    minimum_duration_seconds: float = Field(default=2, ge=0.5, le=60)
    maximum_duration_seconds: float = Field(default=12, ge=1, le=120)
    requires_audio: bool = False
    capture_requirements: dict[str, Any] = Field(default_factory=dict)
    insertion_strategy: Literal["prepend", "append", "overlay", "replace_if_needed"]


class ProductionReadinessReport(BaseModel):
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    ready: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    dimensions: list[ReadinessDimension]
    missions: list[PickupMissionSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PickupValidationResult(BaseModel):
    accepted: bool
    score: float = Field(ge=0, le=1)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    continuity_score: float | None = Field(default=None, ge=0, le=1)


class AcceptedPickup(BaseModel):
    asset_id: str
    mission_type: MissionType
    target_terms: list[str] = Field(default_factory=list)
    insertion_strategy: str
    validation_score: float = Field(ge=0, le=1)


def _tokens(value: str) -> set[str]:
    return {token for token in WORD_RE.findall(value.casefold()) if len(token) > 2}


def _clip_terms(clip: ClipAnalysis) -> set[str]:
    terms = _tokens(clip.filename)
    terms.update(clip.evidence_terms)
    terms.update(semantic_terms(clip.semantic_tags))
    if clip.transcript is not None:
        terms.update(_tokens(clip.transcript.text))
    return terms


def _accepted_clips(analysis: AnalysisBundle) -> list[ClipAnalysis]:
    return [
        clip
        for clip in analysis.source_clips
        if clip.role != "rejected" and not clip.duplicate_of_asset_id
    ]


def _transcript_text(clips: Iterable[ClipAnalysis]) -> str:
    return " ".join(clip.transcript.text for clip in clips if clip.transcript is not None)


def _mission_requirements(mission_type: MissionType) -> dict[str, Any]:
    common = {
        "orientation": "vertical_9_16_preferred",
        "lighting": "Face or product evenly lit; avoid a bright window directly behind the subject.",
        "stability": "Use a tripod or brace the phone; hold the final frame for one second.",
        "safe_zones": "Keep essential faces, products, and text inside the central 70% of frame.",
        "continuity": "Match the existing location, wardrobe, colour temperature, and camera height where possible.",
    }
    if mission_type in {"hook", "cta", "audio_retake"}:
        common.update(
            {
                "framing": "Eye-level medium close-up with eyes near the upper third.",
                "audio": "Quiet room, phone 45–90 cm away, no fan/music, speak one clean sentence per take.",
                "handles": "Leave 0.5 seconds of silence before and after speaking.",
            }
        )
    else:
        common.update(
            {
                "framing": "Show the proof, product, action, or environment clearly without important details near edges.",
                "motion": "Use one slow deliberate move or a locked-off shot; avoid rapid handheld panning.",
                "handles": "Record at least two seconds longer than the requested usable duration.",
            }
        )
    return common


def audit_production_readiness(
    analysis: AnalysisBundle,
    contract: dict[str, Any],
    *,
    threshold: float = 0.72,
) -> ProductionReadinessReport:
    clips = _accepted_clips(analysis)
    primary = [clip for clip in clips if clip.role == "primary_speech"]
    visuals = [clip for clip in clips if clip.role in {"b_roll", "evidence"}]
    text = _transcript_text(clips)
    transcript_terms = _tokens(text)
    objective = str(contract.get("objective") or "")
    objective_terms = _tokens(objective)
    required_terms = {
        token
        for item in contract.get("must_include", [])
        for token in _tokens(str(item))
    }
    target_duration = float(contract.get("target_duration_seconds", 45) or 45)
    available_seconds = sum(float(clip.media.get("duration_seconds", 0) or 0) for clip in clips)

    has_hook = bool(transcript_terms & HOOK_TERMS) or any(
        clip.transcript and clip.transcript.segments and clip.transcript.segments[0].start <= 2
        for clip in primary
    )
    wants_cta = bool(objective_terms & CTA_TERMS) or bool(required_terms & CTA_TERMS)
    has_cta = bool(transcript_terms & CTA_TERMS)
    narrative_score = 0.0
    narrative_findings: list[str] = []
    if primary:
        narrative_score += 0.45
    else:
        narrative_findings.append("No usable primary speaking take was identified.")
    if text.strip():
        narrative_score += 0.25
    else:
        narrative_findings.append("No timestamped spoken narrative is available.")
    if has_hook:
        narrative_score += 0.2
    else:
        narrative_findings.append("The opening lacks a clearly detectable hook.")
    if not wants_cta or has_cta:
        narrative_score += 0.1
    else:
        narrative_findings.append("The objective implies an action, but no clear call to action was detected.")

    audio_clips = [clip for clip in primary if bool(clip.media.get("has_audio"))]
    audio_score = min(1.0, len(audio_clips) / max(1, len(primary))) if primary else 0.0
    audio_findings: list[str] = []
    if not audio_clips:
        audio_findings.append("No accepted primary take contains usable audio.")
    elif any(clip.transcript is None for clip in audio_clips):
        audio_score *= 0.75
        audio_findings.append("Some primary audio could not be verified with a transcript.")

    mean_quality = sum(clip.quality_score for clip in clips) / max(1, len(clips))
    framing_confidence = sum(clip.subject_framing.confidence for clip in primary) / max(1, len(primary))
    visual_score = min(1.0, mean_quality * 0.72 + framing_confidence * 0.28)
    visual_findings: list[str] = []
    if mean_quality < 0.42:
        visual_findings.append("Overall footage quality is below the stable production target.")
    if primary and framing_confidence < 0.25:
        visual_findings.append("Subject framing is uncertain; keep the face inside the central safe zone.")

    coverage_ratio = available_seconds / max(1.0, target_duration)
    coverage_score = min(1.0, coverage_ratio / 1.15)
    coverage_findings: list[str] = []
    if coverage_ratio < 0.8:
        coverage_findings.append(
            f"Only {available_seconds:.1f}s of accepted footage is available for a {target_duration:.0f}s target."
        )
    if target_duration >= 20 and not visuals:
        coverage_score *= 0.72
        coverage_findings.append("No dedicated B-roll or evidence clip is available for visual coverage.")

    evidence_requested = bool((objective_terms | required_terms) & EVIDENCE_TERMS)
    evidence_clips = [clip for clip in visuals if clip.role == "evidence"]
    evidence_score = 1.0 if not evidence_requested else min(1.0, len(evidence_clips) / 1)
    evidence_findings: list[str] = []
    if evidence_requested and not evidence_clips:
        evidence_findings.append("The Director Contract requests proof, but no evidence clip was identified.")

    continuity_scores: list[float] = []
    for left, right in zip(clips, clips[1:], strict=False):
        continuity_scores.append(continuity_similarity(left.continuity, right.continuity))
    continuity_score = sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0.8
    continuity_findings = []
    if continuity_scores and continuity_score < 0.45:
        continuity_findings.append("Lighting, framing, or motion changes sharply across usable clips.")

    dimensions = [
        ReadinessDimension(
            name="narrative",
            score=min(1.0, narrative_score),
            weight=0.28,
            blocking=not primary,
            findings=narrative_findings,
        ),
        ReadinessDimension(
            name="audio",
            score=audio_score,
            weight=0.2,
            blocking=not audio_clips,
            findings=audio_findings,
        ),
        ReadinessDimension(
            name="visual_quality",
            score=visual_score,
            weight=0.17,
            findings=visual_findings,
        ),
        ReadinessDimension(
            name="coverage",
            score=coverage_score,
            weight=0.17,
            findings=coverage_findings,
        ),
        ReadinessDimension(
            name="evidence",
            score=evidence_score,
            weight=0.12,
            blocking=evidence_requested and not evidence_clips,
            findings=evidence_findings,
        ),
        ReadinessDimension(
            name="continuity",
            score=continuity_score,
            weight=0.06,
            findings=continuity_findings,
        ),
    ]
    score = round(sum(item.score * item.weight for item in dimensions), 3)
    blocking = [finding for item in dimensions if item.blocking for finding in item.findings]

    missions: list[PickupMissionSpec] = []
    if not primary or not audio_clips:
        missions.append(
            PickupMissionSpec(
                mission_type="audio_retake",
                priority="blocking",
                title="Record a clean primary explanation",
                reason="A publishable spoken narrative with usable audio is required.",
                target_terms=sorted((objective_terms | required_terms) - CTA_TERMS)[:10],
                minimum_duration_seconds=max(5, min(20, target_duration * 0.35)),
                maximum_duration_seconds=max(15, min(60, target_duration * 1.25)),
                requires_audio=True,
                capture_requirements=_mission_requirements("audio_retake"),
                insertion_strategy="replace_if_needed",
            )
        )
    elif not has_hook:
        missions.append(
            PickupMissionSpec(
                mission_type="hook",
                priority="high",
                title="Record a direct opening hook",
                reason="The current opening does not state a sharp problem, result, or curiosity gap.",
                target_terms=sorted(objective_terms | required_terms)[:8],
                minimum_duration_seconds=2,
                maximum_duration_seconds=7,
                requires_audio=True,
                capture_requirements={
                    **_mission_requirements("hook"),
                    "script_shape": "One sentence: name the painful problem or promised result immediately; no greeting.",
                },
                insertion_strategy="prepend",
            )
        )
    if wants_cta and not has_cta:
        missions.append(
            PickupMissionSpec(
                mission_type="cta",
                priority="high",
                title="Record one clear call to action",
                reason="The objective asks the viewer to act, but the existing footage has no explicit next step.",
                target_terms=sorted((objective_terms | required_terms) & CTA_TERMS)[:6],
                minimum_duration_seconds=2,
                maximum_duration_seconds=8,
                requires_audio=True,
                capture_requirements={
                    **_mission_requirements("cta"),
                    "script_shape": "Use one action, one destination, and one reason to act now.",
                },
                insertion_strategy="append",
            )
        )
    if evidence_requested and not evidence_clips:
        missions.append(
            PickupMissionSpec(
                mission_type="evidence",
                priority="blocking",
                title="Capture the proof shot",
                reason="The claim needs visible support such as a dashboard, result, product detail, or demonstration.",
                target_terms=sorted((objective_terms | required_terms) & EVIDENCE_TERMS)[:10],
                minimum_duration_seconds=3,
                maximum_duration_seconds=12,
                requires_audio=False,
                capture_requirements={
                    **_mission_requirements("evidence"),
                    "shot_list": "Show the exact metric, result, product action, or before/after state mentioned in the narration.",
                    "privacy": "Hide personal data, account numbers, notifications, and unrelated customer information.",
                },
                insertion_strategy="overlay",
            )
        )
    if target_duration >= 20 and not visuals:
        missions.append(
            PickupMissionSpec(
                mission_type="b_roll",
                priority="medium",
                title="Capture one supporting action shot",
                reason="A longer vertical edit needs visual relief from the talking head.",
                target_terms=sorted(objective_terms | required_terms)[:8],
                minimum_duration_seconds=4,
                maximum_duration_seconds=15,
                requires_audio=False,
                capture_requirements={
                    **_mission_requirements("b_roll"),
                    "shot_list": "Record the subject doing the exact action being discussed, or a clean environmental detail that supports it.",
                },
                insertion_strategy="overlay",
            )
        )
    if continuity_score < 0.45:
        missions.append(
            PickupMissionSpec(
                mission_type="continuity_retake",
                priority="medium",
                title="Record a continuity bridge shot",
                reason="The usable clips change sharply in lighting, framing, or motion.",
                target_terms=sorted(objective_terms)[:6],
                minimum_duration_seconds=3,
                maximum_duration_seconds=10,
                requires_audio=False,
                capture_requirements={
                    **_mission_requirements("continuity_retake"),
                    "bridge": "Use a neutral close-up or environmental detail that can hide the jump between locations or takes.",
                },
                insertion_strategy="overlay",
            )
        )

    ready = score >= threshold and not blocking
    return ProductionReadinessReport(
        score=score,
        threshold=threshold,
        ready=ready,
        blocking_reasons=blocking,
        dimensions=dimensions,
        missions=missions,
        notes=[
            "Director Camera uses explainable local media, transcript, semantic, framing, and continuity signals.",
            "A user may override advisory missions, but required-mode blocking missions stop rendering until accepted footage is available.",
        ],
    )


def validate_pickup_clip(
    mission: PickupMissionSpec,
    clip: ClipAnalysis,
    *,
    anchor_clip: ClipAnalysis | None = None,
) -> PickupValidationResult:
    blocking: list[str] = []
    warnings: list[str] = []
    duration = float(clip.media.get("duration_seconds", 0) or 0)
    if duration < mission.minimum_duration_seconds * 0.8:
        blocking.append(
            f"Pickup is {duration:.1f}s; mission needs about {mission.minimum_duration_seconds:.1f}s or more."
        )
    if duration > mission.maximum_duration_seconds * 1.8:
        warnings.append("Pickup is much longer than requested; only the strongest section will be used.")
    if clip.duplicate_of_asset_id:
        blocking.append("Pickup duplicates footage already submitted.")
    if clip.quality_score < 0.3:
        blocking.append("Pickup quality is below the minimum usable threshold.")
    elif clip.quality_score < 0.48:
        warnings.append("Pickup is usable but has limited visual quality headroom.")
    if mission.requires_audio and not bool(clip.media.get("has_audio")):
        blocking.append("This mission requires spoken audio, but no audio stream was detected.")
    if mission.requires_audio and clip.transcript is None:
        warnings.append("Spoken content could not be text-verified; review the take manually before publishing.")

    clip_terms = _clip_terms(clip)
    target_terms = set(mission.target_terms)
    matched = sorted(clip_terms & target_terms)
    semantic_score = len(matched) / max(1, min(6, len(target_terms))) if target_terms else 0.65
    if mission.mission_type == "evidence" and target_terms and semantic_score == 0:
        blocking.append("The submitted pickup does not visibly or textually match the requested proof terms.")
    elif target_terms and semantic_score < 0.2:
        warnings.append("The pickup has only a weak semantic match to the mission.")

    continuity_score = None
    if anchor_clip is not None:
        continuity_score = continuity_similarity(anchor_clip.continuity, clip.continuity)
        if continuity_score < 0.28:
            warnings.append("Lighting or framing differs materially from the existing footage.")

    audio_component = 1.0 if not mission.requires_audio or bool(clip.media.get("has_audio")) else 0.0
    duration_component = min(1.0, duration / max(0.5, mission.minimum_duration_seconds))
    continuity_component = continuity_score if continuity_score is not None else 0.65
    score = round(
        min(
            1.0,
            clip.quality_score * 0.38
            + semantic_score * 0.25
            + audio_component * 0.17
            + duration_component * 0.12
            + continuity_component * 0.08,
        ),
        3,
    )
    return PickupValidationResult(
        accepted=not blocking and score >= 0.58,
        score=score,
        blocking_reasons=blocking,
        warnings=warnings,
        matched_terms=matched[:12],
        continuity_score=continuity_score,
    )


def promote_accepted_pickup(
    clip: ClipAnalysis,
    pickup: AcceptedPickup,
) -> ClipAnalysis:
    role = "primary_speech" if pickup.mission_type in {"hook", "cta", "audio_retake"} else "b_roll"
    if pickup.mission_type == "evidence":
        role = "evidence"
    evidence_terms = sorted(set(clip.evidence_terms) | set(pickup.target_terms))
    tags = [
        *clip.semantic_tags,
        SemanticTag(
            label=f"director_camera_{pickup.mission_type}",
            confidence=pickup.validation_score,
            source="filename",
            evidence="Accepted Director Camera pickup mission.",
        ),
    ]
    return clip.model_copy(
        update={
            "role": role,
            "rejection_reasons": [],
            "evidence_terms": evidence_terms,
            "semantic_tags": tags,
            "quality_score": max(clip.quality_score, pickup.validation_score * 0.8),
        }
    )


def _pickup_segment(
    clip: ClipAnalysis,
    *,
    source_index: int,
    at_start: bool,
) -> EditSegment | None:
    duration = float(clip.media.get("duration_seconds", 0) or 0)
    if duration <= 0.2:
        return None
    transcript_segment = None
    if clip.transcript and clip.transcript.segments:
        transcript_segment = clip.transcript.segments[0] if at_start else clip.transcript.segments[-1]
    if transcript_segment is not None:
        start = transcript_segment.start
        end = min(transcript_segment.end, start + 8)
        text = transcript_segment.text
        confidence = transcript_segment.confidence
    elif clip.scenes:
        scene = clip.scenes[0] if at_start else clip.scenes[-1]
        start = scene.start
        end = min(scene.end, start + 6)
        text = None
        confidence = scene.confidence
    else:
        start = 0.0
        end = min(duration, 6)
        text = None
        confidence = 0.6
    if end - start < 0.3:
        return None
    return EditSegment(
        source_asset_id=clip.asset_id,
        source_index=source_index,
        clip_role="primary_speech",
        source_start=round(start, 3),
        source_end=round(end, 3),
        output_start=0,
        output_end=round(end - start, 3),
        score=0.9,
        confidence=confidence,
        transition="cut",
        reason="Accepted Director Camera pickup inserted for a required story beat.",
        transcript_text=text,
    )


def _reflow(segments: list[EditSegment]) -> list[EditSegment]:
    cursor = 0.0
    result: list[EditSegment] = []
    for segment in segments:
        duration = segment.source_end - segment.source_start
        if duration <= 0.12:
            continue
        result.append(
            segment.model_copy(
                update={
                    "output_start": round(cursor, 3),
                    "output_end": round(cursor + duration, 3),
                }
            )
        )
        cursor += duration
    return result


def integrate_accepted_pickups(
    graph: ProductionEditDecisionGraph,
    analysis: AnalysisBundle,
    pickups: list[AcceptedPickup],
) -> ProductionEditDecisionGraph:
    if not pickups:
        return graph
    clips_by_id = {clip.asset_id: clip for clip in analysis.source_clips}
    source_index_by_id = {clip.asset_id: index for index, clip in enumerate(analysis.source_clips)}
    segments = list(graph.segments)
    overlays = list(graph.overlays)
    notes = list(graph.notes)

    for pickup in pickups:
        clip = clips_by_id.get(pickup.asset_id)
        if clip is None:
            continue
        already_used = any(item.source_asset_id == pickup.asset_id for item in segments) or any(
            item.source_asset_id == pickup.asset_id for item in overlays
        )
        if already_used:
            notes.append(
                f"Director Camera pickup {pickup.asset_id} was selected naturally by the planner."
            )
            continue
        source_index = source_index_by_id[pickup.asset_id]
        if pickup.mission_type in {"hook", "cta"}:
            segment = _pickup_segment(
                clip,
                source_index=source_index,
                at_start=pickup.mission_type == "hook",
            )
            if segment is not None:
                segments = [segment, *segments] if pickup.mission_type == "hook" else [*segments, segment]
                segments = _reflow(segments)
                notes.append(
                    f"Inserted accepted {pickup.mission_type} pickup {pickup.asset_id} into the narration story."
                )
        elif pickup.mission_type in {"evidence", "b_roll", "continuity_retake"} and segments:
            target = next(
                (
                    item
                    for item in segments
                    if _tokens(item.transcript_text or "") & set(pickup.target_terms)
                ),
                segments[len(segments) // 2],
            )
            available = float(clip.media.get("duration_seconds", 0) or 0)
            overlay_duration = min(3.5, available, max(1.2, target.output_end - target.output_start))
            if overlay_duration >= 0.6:
                output_start = min(
                    max(target.output_start + 0.15, 0),
                    max(0, graph.selected_duration_seconds - overlay_duration),
                )
                overlays.append(
                    VisualOverlay(
                        source_asset_id=pickup.asset_id,
                        source_index=source_index,
                        source_start=0,
                        source_end=round(overlay_duration, 3),
                        output_start=round(output_start, 3),
                        output_end=round(output_start + overlay_duration, 3),
                        match_score=pickup.validation_score,
                        continuity_score=0.65,
                        transition="overlay_fade",
                        matched_terms=pickup.target_terms[:8],
                        reason="Accepted Director Camera pickup inserted for missing visual coverage.",
                    )
                )
                notes.append(
                    f"Inserted accepted {pickup.mission_type} pickup {pickup.asset_id} as a visual overlay."
                )

    selected_duration = segments[-1].output_end if segments else 0.0
    return graph.model_copy(
        update={
            "segments": segments,
            "overlays": sorted(overlays, key=lambda item: item.output_start),
            "selected_duration_seconds": round(selected_duration, 3),
            "notes": notes,
        }
    )
