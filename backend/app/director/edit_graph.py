from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from app.sensory.models import AnalysisBundle, ClipAnalysis, TranscriptSegment

FILLER_WORDS = {
    "ah",
    "basically",
    "erm",
    "hmm",
    "kind of",
    "like",
    "literally",
    "sort of",
    "uh",
    "um",
    "you know",
}
HOOK_TERMS = {
    "avoid",
    "best",
    "biggest",
    "how",
    "mistake",
    "never",
    "secret",
    "stop",
    "why",
}


class EditSegment(BaseModel):
    source_asset_id: str | None = None
    source_index: int = Field(default=0, ge=0)
    clip_role: str = "primary_speech"
    source_start: float = Field(ge=0)
    source_end: float = Field(ge=0)
    output_start: float = Field(ge=0)
    output_end: float = Field(ge=0)
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    transition: str = "cut"
    reason: str
    transcript_text: str | None = None


class EditDecisionGraph(BaseModel):
    version: int = 1
    strategy: str = "tier1_retention_cleanup"
    target_duration_seconds: float = Field(gt=0)
    selected_duration_seconds: float = Field(ge=0)
    segments: list[EditSegment]
    notes: list[str] = Field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def _segment_score(
    segment: TranscriptSegment,
    objective_terms: set[str],
    source_duration: float,
) -> float:
    tokens = _tokens(segment.text)
    if not tokens:
        return 0.05

    duration = max(0.01, segment.end - segment.start)
    filler_hits = sum(1 for token in tokens if token in FILLER_WORDS)
    filler_ratio = filler_hits / len(tokens)
    objective_hits = len(set(tokens) & objective_terms)
    hook_hits = len(set(tokens) & HOOK_TERMS)

    score = 0.48
    if 1.2 <= duration <= 10:
        score += 0.12
    elif duration > 16:
        score -= 0.12
    if source_duration > 0 and segment.start <= min(15.0, source_duration * 0.2):
        score += 0.08
    score += min(objective_hits * 0.06, 0.18)
    score += min(hook_hits * 0.07, 0.14)
    score -= min(filler_ratio * 0.5, 0.3)
    if len(tokens) < 3:
        score -= 0.12
    return min(1.0, max(0.05, score))


def _fallback_candidates(analysis: AnalysisBundle) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            start=scene.start,
            end=scene.end,
            text="",
            confidence=scene.confidence,
        )
        for scene in analysis.scenes
    ]


def _merge_adjacent(segments: Iterable[TranscriptSegment]) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for segment in sorted(segments, key=lambda item: item.start):
        if not merged or segment.start - merged[-1].end > 0.2:
            merged.append(segment.model_copy())
            continue
        previous = merged[-1]
        previous.end = max(previous.end, segment.end)
        previous.text = " ".join(part for part in [previous.text, segment.text] if part).strip()
        previous.confidence = min(previous.confidence, segment.confidence)
    return merged


def build_tier1_edit_graph(
    analysis: AnalysisBundle,
    *,
    objective: str,
    target_duration_seconds: float,
    source_asset_id: str | None = None,
    source_index: int = 0,
    clip_role: str = "primary_speech",
) -> EditDecisionGraph:
    source_duration = float(analysis.media.get("duration_seconds", 0) or 0)
    candidates = list(analysis.transcript.segments) if analysis.transcript else []
    uses_transcript = bool(candidates)
    notes: list[str] = []
    if not candidates:
        candidates = _fallback_candidates(analysis)
        notes.append("Transcript unavailable; scene ranges were used as conservative candidates.")
    if not candidates and source_duration > 0:
        candidates = [
            TranscriptSegment(start=0, end=source_duration, text="", confidence=0.4)
        ]

    objective_terms = set(_tokens(objective))
    scored = [
        (candidate, _segment_score(candidate, objective_terms, source_duration))
        for candidate in candidates
        if candidate.end > candidate.start
    ]
    scored.sort(key=lambda item: (-item[1], item[0].start))

    chosen: list[TranscriptSegment] = []
    remaining = target_duration_seconds
    for candidate, score in scored:
        if remaining <= 0.2:
            break
        duration = candidate.end - candidate.start
        if uses_transcript and score < 0.34 and chosen:
            continue
        if duration > remaining:
            candidate = candidate.model_copy(update={"end": candidate.start + remaining})
            duration = remaining
        chosen.append(candidate)
        remaining -= duration

    if not chosen and scored:
        candidate = scored[0][0]
        chosen = [
            candidate.model_copy(
                update={"end": min(candidate.end, candidate.start + target_duration_seconds)}
            )
        ]

    merged = _merge_adjacent(chosen)
    output_cursor = 0.0
    graph_segments: list[EditSegment] = []
    score_lookup = {(item.start, item.end): score for item, score in scored}
    for segment in merged:
        duration = segment.end - segment.start
        score = score_lookup.get((segment.start, segment.end), 0.62)
        reason = "Selected for clarity and information density"
        if not uses_transcript:
            reason = "Selected as conservative visual coverage without transcript evidence"
        elif segment.start <= 15:
            reason = "Selected as an early high-value hook or setup"
        graph_segments.append(
            EditSegment(
                source_asset_id=source_asset_id,
                source_index=source_index,
                clip_role=clip_role,
                source_start=round(segment.start, 3),
                source_end=round(segment.end, 3),
                output_start=round(output_cursor, 3),
                output_end=round(output_cursor + duration, 3),
                score=round(score, 3),
                confidence=round(segment.confidence, 3),
                reason=reason,
                transcript_text=segment.text or None,
            )
        )
        output_cursor += duration

    return EditDecisionGraph(
        target_duration_seconds=target_duration_seconds,
        selected_duration_seconds=round(output_cursor, 3),
        segments=graph_segments,
        notes=notes,
    )


def _clip_bundle(clip: ClipAnalysis) -> AnalysisBundle:
    return AnalysisBundle(
        media=clip.media,
        transcript=clip.transcript,
        scenes=clip.scenes,
        subject_framing=clip.subject_framing,
    )


def _candidate_segments(
    clips: list[ClipAnalysis],
    *,
    objective: str,
    target_duration_seconds: float,
) -> list[EditSegment]:
    candidates: list[EditSegment] = []
    for source_index, clip in enumerate(clips):
        if clip.role == "rejected" or clip.duplicate_of_asset_id:
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
            role_bonus = 0.04 if clip.role == "evidence" else 0.02 if clip.role == "b_roll" else 0
            adjusted = min(1.0, segment.score * 0.76 + clip.quality_score * 0.24 + role_bonus)
            candidates.append(
                segment.model_copy(
                    update={
                        "score": round(adjusted, 3),
                        "confidence": round(
                            min(segment.confidence, max(0.25, clip.quality_score)),
                            3,
                        ),
                        "reason": (
                            f"{segment.reason}; selected from {clip.filename} "
                            f"({clip.role}, clip quality {clip.quality_score:.2f})"
                        ),
                    }
                )
            )
    return candidates


def build_multiclip_edit_graph(
    analysis: AnalysisBundle,
    *,
    objective: str,
    target_duration_seconds: float,
) -> EditDecisionGraph:
    clips = analysis.source_clips
    if not clips:
        return build_tier1_edit_graph(
            analysis,
            objective=objective,
            target_duration_seconds=target_duration_seconds,
        )

    candidates = _candidate_segments(
        clips,
        objective=objective,
        target_duration_seconds=target_duration_seconds,
    )
    if not candidates:
        return EditDecisionGraph(
            strategy="tier1_multiclip_story",
            target_duration_seconds=target_duration_seconds,
            selected_duration_seconds=0,
            segments=[],
            notes=["All uploaded source clips were rejected or identified as duplicates."],
        )

    speech = [item for item in candidates if item.clip_role == "primary_speech"]
    visual = [item for item in candidates if item.clip_role in {"b_roll", "evidence"}]
    speech.sort(key=lambda item: (-item.score, item.source_index, item.source_start))
    visual.sort(key=lambda item: (-item.score, item.source_index, item.source_start))

    ordered: list[EditSegment] = []
    if speech:
        ordered.append(speech.pop(0))

    while speech or visual:
        use_visual = bool(visual) and bool(ordered) and len(ordered) % 3 == 2
        pool = visual if use_visual else speech or visual
        last_asset = ordered[-1].source_asset_id if ordered else None
        different_source = next(
            (item for item in pool if item.source_asset_id != last_asset),
            None,
        )
        selected = different_source or pool[0]
        pool.remove(selected)
        ordered.append(selected)

    output_cursor = 0.0
    selected_segments: list[EditSegment] = []
    for item in ordered:
        remaining = target_duration_seconds - output_cursor
        if remaining <= 0.2:
            break
        duration = item.source_end - item.source_start
        if duration > remaining:
            item = item.model_copy(update={"source_end": item.source_start + remaining})
            duration = remaining
        transition = "cut"
        if selected_segments and selected_segments[-1].source_asset_id != item.source_asset_id:
            transition = "source_change_cut"
        selected_segments.append(
            item.model_copy(
                update={
                    "output_start": round(output_cursor, 3),
                    "output_end": round(output_cursor + duration, 3),
                    "transition": transition,
                }
            )
        )
        output_cursor += duration

    rejected = sum(1 for clip in clips if clip.role == "rejected")
    duplicates = sum(1 for clip in clips if clip.duplicate_of_asset_id)
    notes = [
        f"Built one story from {len(clips) - rejected} accepted source clip(s).",
        f"Rejected {rejected} clip(s), including {duplicates} duplicate or near-duplicate clip(s).",
    ]
    if any(segment.clip_role == "evidence" for segment in selected_segments):
        notes.append("Inserted evidence-oriented footage where it strengthened the selected story.")
    if any(segment.clip_role == "b_roll" for segment in selected_segments):
        notes.append("Inserted B-roll candidates to add visual coverage and pacing contrast.")

    return EditDecisionGraph(
        strategy="tier1_multiclip_story",
        target_duration_seconds=target_duration_seconds,
        selected_duration_seconds=round(output_cursor, 3),
        segments=selected_segments,
        notes=notes,
    )
