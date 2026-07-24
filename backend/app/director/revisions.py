from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from app.director.edit_graph import EditSegment
from app.director.semantic_overlays import ProductionEditDecisionGraph, VisualOverlay


class LockedRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    label: str | None = Field(default=None, max_length=200)


class ChangedRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    reason: str


class RevisionIntent(BaseModel):
    instruction: str
    target_duration_seconds: float | None = Field(default=None, gt=0)
    trim_intro_seconds: float = Field(default=0, ge=0)
    trim_outro_seconds: float = Field(default=0, ge=0)
    remove_terms: list[str] = Field(default_factory=list)
    overlay_action: Literal["keep", "remove", "less", "more"] = "keep"
    captions_enabled: bool | None = None
    caption_all_caps: bool | None = None
    caption_size_delta: int = 0
    music_enabled: bool | None = None
    warnings: list[str] = Field(default_factory=list)


class RevisionRenderPlan(BaseModel):
    scope: Literal["metadata_only", "component_partial", "full_master"]
    changed_components: list[str] = Field(default_factory=list)
    changed_ranges: list[ChangedRange] = Field(default_factory=list)
    reused_ranges: list[ChangedRange] = Field(default_factory=list)
    reuse_narration_master: bool = False
    base_graph_fingerprint: str
    revised_graph_fingerprint: str
    notes: list[str] = Field(default_factory=list)


class RevisionApplication(BaseModel):
    graph: ProductionEditDecisionGraph
    intent: RevisionIntent
    render_plan: RevisionRenderPlan


def _seconds_near(text: str, term: str, default: float) -> float:
    match = re.search(
        rf"(?:{term})[^\d]{{0,24}}(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
        text,
    )
    return float(match.group(1)) if match else default


def parse_revision_intent(
    instruction: str,
    *,
    base_duration_seconds: float,
) -> RevisionIntent:
    text = " ".join(instruction.casefold().split())
    warnings: list[str] = []

    target_duration = None
    duration_match = re.search(
        r"(?:shorten|trim|make|cut)[^\d]{0,30}(?:to|around|about)?\s*(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
        text,
    )
    if duration_match:
        target_duration = float(duration_match.group(1))
    elif any(term in text for term in ("make it faster", "make it tighter", "tighten the edit")):
        target_duration = max(5.0, round(base_duration_seconds * 0.85, 2))

    trim_intro = 0.0
    if any(term in text for term in ("remove intro", "remove the intro", "cut intro", "trim intro")):
        trim_intro = _seconds_near(text, r"(?:intro|opening|beginning|start)", 1.5)

    trim_outro = 0.0
    if any(term in text for term in ("remove outro", "remove the outro", "cut ending", "trim outro")):
        trim_outro = _seconds_near(text, r"(?:outro|ending|end)", 1.5)

    remove_terms = [
        match.strip()
        for match in re.findall(r"(?:remove|omit|delete)\s+[\"']([^\"']+)[\"']", instruction, re.I)
        if match.strip()
    ]

    overlay_action: Literal["keep", "remove", "less", "more"] = "keep"
    if any(term in text for term in ("remove b-roll", "remove broll", "no b-roll", "no broll", "remove overlays")):
        overlay_action = "remove"
    elif any(term in text for term in ("less b-roll", "less broll", "fewer overlays")):
        overlay_action = "less"
    elif any(term in text for term in ("more b-roll", "more broll", "more evidence", "more overlays")):
        overlay_action = "more"
        warnings.append(
            "Additional overlay discovery requires re-running semantic planning; the revision worker will attempt it when source analysis is available."
        )

    captions_enabled = None
    if any(term in text for term in ("remove captions", "no captions", "hide captions")):
        captions_enabled = False
    elif any(term in text for term in ("add captions", "show captions", "enable captions")):
        captions_enabled = True

    caption_all_caps = None
    if any(term in text for term in ("all caps captions", "uppercase captions", "captions uppercase")):
        caption_all_caps = True
    elif any(term in text for term in ("normal case captions", "sentence case captions", "not all caps")):
        caption_all_caps = False

    caption_size_delta = 0
    if any(term in text for term in ("larger captions", "bigger captions", "increase caption size")):
        caption_size_delta = 8
    elif any(term in text for term in ("smaller captions", "reduce caption size")):
        caption_size_delta = -8

    music_enabled = None
    if any(term in text for term in ("remove music", "no music", "mute music")):
        music_enabled = False
    elif any(term in text for term in ("add music", "enable music", "restore music")):
        music_enabled = True

    supported = any(
        [
            target_duration is not None,
            trim_intro > 0,
            trim_outro > 0,
            bool(remove_terms),
            overlay_action != "keep",
            captions_enabled is not None,
            caption_all_caps is not None,
            caption_size_delta != 0,
            music_enabled is not None,
        ]
    )
    if not supported:
        warnings.append(
            "No deterministic revision operation was recognized. The request is preserved for a future model-backed revision interpreter."
        )

    return RevisionIntent(
        instruction=instruction,
        target_duration_seconds=target_duration,
        trim_intro_seconds=trim_intro,
        trim_outro_seconds=trim_outro,
        remove_terms=remove_terms,
        overlay_action=overlay_action,
        captions_enabled=captions_enabled,
        caption_all_caps=caption_all_caps,
        caption_size_delta=caption_size_delta,
        music_enabled=music_enabled,
        warnings=warnings,
    )


def _overlaps(start: float, end: float, locked: list[LockedRange]) -> bool:
    return any(end > item.start and start < item.end for item in locked)


def _reflow_segments(segments: list[EditSegment]) -> list[EditSegment]:
    cursor = 0.0
    result: list[EditSegment] = []
    for segment in segments:
        duration = max(0.0, segment.source_end - segment.source_start)
        if duration < 0.12:
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


def _trim_intro(
    segments: list[EditSegment],
    seconds: float,
    locked: list[LockedRange],
) -> list[EditSegment]:
    if seconds <= 0:
        return segments
    result: list[EditSegment] = []
    for segment in segments:
        if _overlaps(segment.output_start, segment.output_end, locked):
            result.append(segment)
            continue
        if segment.output_end <= seconds:
            continue
        if segment.output_start < seconds:
            removed = seconds - segment.output_start
            segment = segment.model_copy(
                update={"source_start": min(segment.source_end, segment.source_start + removed)}
            )
        result.append(segment)
    return result


def _trim_outro(
    segments: list[EditSegment],
    seconds: float,
    total_duration: float,
    locked: list[LockedRange],
) -> list[EditSegment]:
    if seconds <= 0:
        return segments
    cutoff = max(0.0, total_duration - seconds)
    result: list[EditSegment] = []
    for segment in segments:
        if _overlaps(segment.output_start, segment.output_end, locked):
            result.append(segment)
            continue
        if segment.output_start >= cutoff:
            continue
        if segment.output_end > cutoff:
            kept = cutoff - segment.output_start
            segment = segment.model_copy(
                update={"source_end": max(segment.source_start, segment.source_start + kept)}
            )
        result.append(segment)
    return result


def _limit_duration(
    segments: list[EditSegment],
    target: float,
    locked: list[LockedRange],
) -> list[EditSegment]:
    result: list[EditSegment] = []
    cursor = 0.0
    for segment in segments:
        duration = segment.source_end - segment.source_start
        if cursor >= target and not _overlaps(segment.output_start, segment.output_end, locked):
            continue
        if cursor + duration > target and not _overlaps(
            segment.output_start, segment.output_end, locked
        ):
            duration = max(0.0, target - cursor)
            if duration < 0.12:
                continue
            segment = segment.model_copy(update={"source_end": segment.source_start + duration})
        result.append(segment)
        cursor += max(0.0, segment.source_end - segment.source_start)
    return result


def _fingerprint(graph: ProductionEditDecisionGraph) -> str:
    payload = graph.model_dump(mode="json", exclude={"notes", "critic_report"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _segment_signature(segment: EditSegment) -> tuple[object, ...]:
    return (
        segment.source_asset_id,
        round(segment.source_start, 3),
        round(segment.source_end, 3),
        segment.transcript_text,
    )


def _overlay_signature(overlay: VisualOverlay) -> tuple[object, ...]:
    return (
        overlay.source_asset_id,
        round(overlay.source_start, 3),
        round(overlay.source_end, 3),
        round(overlay.output_start, 3),
        round(overlay.output_end, 3),
    )


def compare_graphs(
    base: ProductionEditDecisionGraph,
    revised: ProductionEditDecisionGraph,
) -> RevisionRenderPlan:
    changed_components: list[str] = []
    changed_ranges: list[ChangedRange] = []

    base_segments = [_segment_signature(item) for item in base.segments]
    revised_segments = [_segment_signature(item) for item in revised.segments]
    if base_segments != revised_segments:
        changed_components.append("narration")
        end = max(base.selected_duration_seconds, revised.selected_duration_seconds)
        changed_ranges.append(ChangedRange(start=0, end=end, reason="Narration graph changed"))

    base_overlays = [_overlay_signature(item) for item in base.overlays]
    revised_overlays = [_overlay_signature(item) for item in revised.overlays]
    if base_overlays != revised_overlays:
        changed_components.append("overlays")
        ranges = [
            ChangedRange(start=item.output_start, end=item.output_end, reason="Visual overlay changed")
            for item in revised.overlays
            if _overlay_signature(item) not in set(base_overlays)
        ]
        changed_ranges.extend(ranges)

    base_overrides = base.render_overrides
    revised_overrides = revised.render_overrides
    for component, keys in {
        "captions": {"captions_enabled", "caption_all_caps", "caption_size_delta"},
        "music": {"music_enabled"},
    }.items():
        if any(base_overrides.get(key) != revised_overrides.get(key) for key in keys):
            changed_components.append(component)

    changed_components = sorted(set(changed_components))
    if not changed_components:
        scope: Literal["metadata_only", "component_partial", "full_master"] = "metadata_only"
    elif set(changed_components) <= {"overlays", "captions"}:
        scope = "component_partial"
    else:
        scope = "full_master"

    reuse_narration = scope in {"metadata_only", "component_partial"}
    reused_ranges = []
    if reuse_narration and revised.selected_duration_seconds > 0:
        reused_ranges.append(
            ChangedRange(
                start=0,
                end=revised.selected_duration_seconds,
                reason="Narration master is unchanged and can be reused",
            )
        )

    return RevisionRenderPlan(
        scope=scope,
        changed_components=changed_components,
        changed_ranges=changed_ranges,
        reused_ranges=reused_ranges,
        reuse_narration_master=reuse_narration,
        base_graph_fingerprint=_fingerprint(base),
        revised_graph_fingerprint=_fingerprint(revised),
        notes=[],
    )


def apply_revision(
    base_graph: ProductionEditDecisionGraph,
    instruction: str,
    *,
    next_version: int,
    locked_ranges: list[LockedRange] | None = None,
) -> RevisionApplication:
    locked = locked_ranges or []
    intent = parse_revision_intent(
        instruction,
        base_duration_seconds=base_graph.selected_duration_seconds,
    )
    segments = list(base_graph.segments)

    segments = _trim_intro(segments, intent.trim_intro_seconds, locked)
    segments = _trim_outro(
        segments,
        intent.trim_outro_seconds,
        base_graph.selected_duration_seconds,
        locked,
    )
    if intent.remove_terms:
        terms = [term.casefold() for term in intent.remove_terms]
        segments = [
            segment
            for segment in segments
            if _overlaps(segment.output_start, segment.output_end, locked)
            or not any(term in (segment.transcript_text or "").casefold() for term in terms)
        ]
    if intent.target_duration_seconds is not None:
        segments = _limit_duration(segments, intent.target_duration_seconds, locked)
    segments = _reflow_segments(segments)

    overlays = list(base_graph.overlays)
    if intent.overlay_action == "remove":
        overlays = []
    elif intent.overlay_action == "less":
        overlays = overlays[: max(0, (len(overlays) + 1) // 2)]

    duration = segments[-1].output_end if segments else 0.0
    overlays = [
        overlay
        for overlay in overlays
        if overlay.output_start < duration and overlay.output_end <= duration
    ]

    overrides = dict(base_graph.render_overrides)
    if intent.captions_enabled is not None:
        overrides["captions_enabled"] = intent.captions_enabled
    if intent.caption_all_caps is not None:
        overrides["caption_all_caps"] = intent.caption_all_caps
    if intent.caption_size_delta:
        overrides["caption_size_delta"] = intent.caption_size_delta
    if intent.music_enabled is not None:
        overrides["music_enabled"] = intent.music_enabled

    notes = [
        *base_graph.notes,
        f"Revision v{next_version}: {instruction.strip()}",
        *intent.warnings,
    ]
    revised = base_graph.model_copy(
        update={
            "version": next_version,
            "selected_duration_seconds": round(duration, 3),
            "target_duration_seconds": (
                intent.target_duration_seconds
                if intent.target_duration_seconds is not None
                else base_graph.target_duration_seconds
            ),
            "segments": segments,
            "overlays": overlays,
            "render_overrides": overrides,
            "notes": notes,
            "critic_report": {},
        }
    )
    render_plan = compare_graphs(base_graph, revised)
    render_plan.notes.extend(intent.warnings)
    return RevisionApplication(graph=revised, intent=intent, render_plan=render_plan)
