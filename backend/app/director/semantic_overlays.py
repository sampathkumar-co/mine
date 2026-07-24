from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.director.edit_graph import (
    EditDecisionGraph,
    EditSegment,
    build_tier1_edit_graph,
)
from app.sensory.models import AnalysisBundle, ClipAnalysis
from app.sensory.semantics import continuity_similarity, semantic_terms

STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


class VisualOverlay(BaseModel):
    source_asset_id: str
    source_index: int = Field(ge=0)
    source_start: float = Field(ge=0)
    source_end: float = Field(gt=0)
    output_start: float = Field(ge=0)
    output_end: float = Field(gt=0)
    match_score: float = Field(ge=0, le=1)
    continuity_score: float = Field(default=0.5, ge=0, le=1)
    transition: str = "overlay_cut"
    matched_terms: list[str] = Field(default_factory=list)
    reason: str


class ContinuityDecision(BaseModel):
    left_asset_id: str | None = None
    right_asset_id: str | None = None
    score: float = Field(ge=0, le=1)
    transition: str
    reason: str


class ProductionEditDecisionGraph(EditDecisionGraph):
    strategy: str = "tier1_semantic_evidence_overlays"
    overlays: list[VisualOverlay] = Field(default_factory=list)
    continuity_decisions: list[ContinuityDecision] = Field(default_factory=list)
    critic_report: dict[str, object] = Field(default_factory=dict)


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", value.casefold())
        if len(token) > 2 and token not in STOP_TERMS
    }


def _clip_bundle(clip: ClipAnalysis) -> AnalysisBundle:
    return AnalysisBundle(
        media=clip.media,
        transcript=clip.transcript,
        scenes=clip.scenes,
        subject_framing=clip.subject_framing,
    )


def _speech_story(
    fallback: EditDecisionGraph,
    analysis: AnalysisBundle,
    *,
    objective: str,
    target_duration_seconds: float,
) -> list[EditSegment]:
    candidates: list[EditSegment] = []
    for source_index, clip in enumerate(analysis.source_clips):
        if clip.role != "primary_speech" or clip.duplicate_of_asset_id:
            continue
        duration = float(clip.media.get("duration_seconds", 0) or 0)
        clip_graph = build_tier1_edit_graph(
            _clip_bundle(clip),
            objective=objective,
            target_duration_seconds=min(target_duration_seconds, max(duration, 0.1)),
            source_asset_id=clip.asset_id,
            source_index=source_index,
            clip_role=clip.role,
        )
        for segment in clip_graph.segments:
            candidates.append(
                segment.model_copy(
                    update={
                        "score": round(min(1.0, segment.score * 0.78 + clip.quality_score * 0.22), 3),
                        "reason": (
                            f"{segment.reason}; retained as narration from {clip.filename} "
                            f"(quality {clip.quality_score:.2f})"
                        ),
                    }
                )
            )

    if not candidates:
        candidates = [
            segment
            for segment in fallback.segments
            if segment.clip_role == "primary_speech"
        ] or list(fallback.segments)

    candidates.sort(key=lambda item: (-item.score, item.source_index, item.source_start))
    ordered: list[EditSegment] = []
    while candidates:
        last_asset = ordered[-1].source_asset_id if ordered else None
        different_source = next(
            (item for item in candidates if item.source_asset_id != last_asset),
            None,
        )
        selected = different_source or candidates[0]
        candidates.remove(selected)
        ordered.append(selected)

    output_cursor = 0.0
    selected_story: list[EditSegment] = []
    for item in ordered:
        remaining = target_duration_seconds - output_cursor
        if remaining <= 0.2:
            break
        duration = item.source_end - item.source_start
        if duration > remaining:
            item = item.model_copy(update={"source_end": item.source_start + remaining})
            duration = remaining
        selected_story.append(
            item.model_copy(
                update={
                    "output_start": round(output_cursor, 3),
                    "output_end": round(output_cursor + duration, 3),
                }
            )
        )
        output_cursor += duration
    return selected_story


def _apply_continuity(
    segments: list[EditSegment],
    clips_by_id: dict[str, ClipAnalysis],
) -> tuple[list[EditSegment], list[ContinuityDecision]]:
    if not segments:
        return [], []

    updated = [segments[0].model_copy(update={"transition": "cut"})]
    decisions: list[ContinuityDecision] = []
    for previous, current in zip(segments, segments[1:], strict=False):
        left = clips_by_id.get(previous.source_asset_id or "")
        right = clips_by_id.get(current.source_asset_id or "")
        score = continuity_similarity(left.continuity, right.continuity) if left and right else 0.5
        if previous.source_asset_id == current.source_asset_id:
            transition = "intra_clip_cut"
            reason = "The adjacent selections come from the same source clip."
        elif score >= 0.78:
            transition = "match_cut"
            reason = "Framing, light, colour, and motion are sufficiently similar for a match cut."
        elif score >= 0.5:
            transition = "continuity_cut"
            reason = "The sources are compatible enough for a direct continuity cut."
        else:
            transition = "soft_dissolve"
            reason = "The sources differ materially, so a softer visual transition is preferred."
        updated.append(current.model_copy(update={"transition": transition}))
        decisions.append(
            ContinuityDecision(
                left_asset_id=previous.source_asset_id,
                right_asset_id=current.source_asset_id,
                score=score,
                transition=transition,
                reason=reason,
            )
        )
    return updated, decisions


def _overlay_terms(clip: ClipAnalysis) -> set[str]:
    terms = set(clip.evidence_terms)
    terms.update(_tokens(clip.filename))
    terms.update(semantic_terms(clip.semantic_tags))
    if clip.transcript:
        terms.update(_tokens(clip.transcript.text))
    return terms


def _build_overlays(
    story: list[EditSegment],
    clips: list[ClipAnalysis],
    *,
    objective: str,
    max_overlays: int,
    minimum_match_score: float,
) -> list[VisualOverlay]:
    visual_clips = [
        (index, clip)
        for index, clip in enumerate(clips)
        if clip.role in {"b_roll", "evidence"}
        and not clip.duplicate_of_asset_id
        and clip.quality_score >= 0.25
    ]
    if not visual_clips or not story or max_overlays <= 0:
        return []

    objective_terms = _tokens(objective)
    used_assets: set[str] = set()
    overlays: list[VisualOverlay] = []
    for segment in story:
        segment_duration = segment.output_end - segment.output_start
        if segment_duration < 1.2 or len(overlays) >= max_overlays:
            continue
        claim_terms = _tokens(segment.transcript_text or "") | objective_terms
        best: tuple[float, int, ClipAnalysis, list[str]] | None = None
        for source_index, clip in visual_clips:
            if clip.asset_id in used_assets and len(visual_clips) > 1:
                continue
            terms = _overlay_terms(clip)
            matched = sorted(claim_terms & terms)
            overlap = len(matched) / max(1, min(len(claim_terms), 8))
            role_bonus = 0.22 if clip.role == "evidence" else 0.1
            semantic_bonus = min(0.16, len(clip.semantic_tags) * 0.025)
            score = min(1.0, overlap * 0.55 + clip.quality_score * 0.22 + role_bonus + semantic_bonus)
            candidate = (score, source_index, clip, matched)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or best[0] < minimum_match_score:
            continue

        score, source_index, clip, matched_terms = best
        clip_duration = float(clip.media.get("duration_seconds", 0) or 0)
        overlay_duration = min(3.2, max(1.0, segment_duration * 0.55), clip_duration)
        if overlay_duration <= 0.2:
            continue
        output_start = segment.output_start + min(0.35, max(0.08, segment_duration * 0.12))
        output_end = min(segment.output_end - 0.05, output_start + overlay_duration)
        if output_end - output_start < 0.6:
            continue
        source_start = clip.scenes[0].start if clip.scenes else 0.0
        source_end = min(clip_duration, source_start + (output_end - output_start))
        if source_end - source_start < 0.6:
            source_start = 0.0
            source_end = min(clip_duration, output_end - output_start)
        narration_clip = next(
            (item for item in clips if item.asset_id == segment.source_asset_id),
            None,
        )
        continuity = (
            continuity_similarity(narration_clip.continuity, clip.continuity)
            if narration_clip is not None
            else 0.5
        )
        transition = "overlay_match_cut" if continuity >= 0.72 else "overlay_fade"
        overlays.append(
            VisualOverlay(
                source_asset_id=clip.asset_id,
                source_index=source_index,
                source_start=round(source_start, 3),
                source_end=round(source_end, 3),
                output_start=round(output_start, 3),
                output_end=round(output_end, 3),
                match_score=round(score, 3),
                continuity_score=continuity,
                transition=transition,
                matched_terms=matched_terms[:8],
                reason=(
                    f"Matched {clip.filename} to the narration claim using "
                    f"{', '.join(matched_terms[:5]) or 'semantic role and quality evidence'}."
                ),
            )
        )
        used_assets.add(clip.asset_id)
    return overlays


def _style_overlay_limit(analysis: AnalysisBundle, default: int) -> int:
    style = analysis.production_style
    if not isinstance(style, dict):
        return default
    value = style.get("max_visual_overlays")
    if value is None:
        return default
    try:
        return min(default, max(0, int(value)))
    except (TypeError, ValueError):
        return default


def enhance_graph_with_semantic_overlays(
    graph: EditDecisionGraph,
    analysis: AnalysisBundle,
    *,
    objective: str,
    target_duration_seconds: float,
    max_overlays: int = 4,
    minimum_match_score: float = 0.3,
) -> ProductionEditDecisionGraph:
    max_overlays = _style_overlay_limit(analysis, max_overlays)
    clips_by_id = {clip.asset_id: clip for clip in analysis.source_clips}
    story = _speech_story(
        graph,
        analysis,
        objective=objective,
        target_duration_seconds=target_duration_seconds,
    )
    story, continuity_decisions = _apply_continuity(story, clips_by_id)
    overlays = _build_overlays(
        story,
        analysis.source_clips,
        objective=objective,
        max_overlays=max_overlays,
        minimum_match_score=minimum_match_score,
    )
    selected_duration = story[-1].output_end if story else 0.0
    notes = [
        *graph.notes,
        f"Preserved narration while scheduling {len(overlays)} semantic evidence/B-roll overlay(s).",
        f"Applied a maximum of {max_overlays} visual overlay(s) for this production style.",
        f"Scored {len(continuity_decisions)} cross-segment continuity decision(s).",
    ]
    return ProductionEditDecisionGraph(
        version=graph.version,
        target_duration_seconds=target_duration_seconds,
        selected_duration_seconds=round(selected_duration, 3),
        segments=story,
        overlays=overlays,
        continuity_decisions=continuity_decisions,
        notes=notes,
    )
