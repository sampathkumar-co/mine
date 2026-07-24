from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import Field

from app.director.revisions import (
    ChangedRange,
    LockedRange,
    RevisionApplication,
    RevisionRenderPlan,
    _limit_duration,
    _overlaps,
    _overlay_signature,
    _reflow_segments,
    _segment_signature,
    _trim_intro,
    _trim_outro,
    parse_revision_intent,
)
from app.director.semantic_overlays import ProductionEditDecisionGraph


class RevisionEditDecisionGraph(ProductionEditDecisionGraph):
    render_overrides: dict[str, object] = Field(default_factory=dict)


def normalize_revision_graph(payload: dict[str, object]) -> RevisionEditDecisionGraph:
    return RevisionEditDecisionGraph.model_validate(payload)


def _fingerprint(graph: RevisionEditDecisionGraph) -> str:
    payload = graph.model_dump(mode="json", exclude={"notes", "critic_report"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_revision_phrasing(instruction: str) -> str:
    normalized = instruction
    replacements = {
        r"\bmake\s+(?:the\s+)?captions?\s+larger\b": "larger captions",
        r"\bmake\s+(?:the\s+)?captions?\s+bigger\b": "bigger captions",
        r"\bmake\s+(?:the\s+)?captions?\s+smaller\b": "smaller captions",
    }
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def compare_revision_graphs(
    base: RevisionEditDecisionGraph,
    revised: RevisionEditDecisionGraph,
) -> RevisionRenderPlan:
    changed_components: list[str] = []
    changed_ranges: list[ChangedRange] = []

    base_segments = [_segment_signature(item) for item in base.segments]
    revised_segments = [_segment_signature(item) for item in revised.segments]
    if base_segments != revised_segments:
        changed_components.append("narration")
        changed_ranges.append(
            ChangedRange(
                start=0,
                end=max(base.selected_duration_seconds, revised.selected_duration_seconds),
                reason="Narration graph changed",
            )
        )

    base_overlay_signatures = [_overlay_signature(item) for item in base.overlays]
    revised_overlay_signatures = [_overlay_signature(item) for item in revised.overlays]
    if base_overlay_signatures != revised_overlay_signatures:
        changed_components.append("overlays")
        known = set(base_overlay_signatures)
        changed_ranges.extend(
            ChangedRange(
                start=item.output_start,
                end=item.output_end,
                reason="Visual overlay changed",
            )
            for item in revised.overlays
            if _overlay_signature(item) not in known
        )

    for component, keys in {
        "captions": {"captions_enabled", "caption_all_caps", "caption_size_delta"},
        "music": {"music_enabled"},
    }.items():
        if any(
            base.render_overrides.get(key) != revised.render_overrides.get(key)
            for key in keys
        ):
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


def apply_graph_revision(
    base_graph: RevisionEditDecisionGraph,
    instruction: str,
    *,
    next_version: int,
    locked_ranges: list[LockedRange] | None = None,
) -> RevisionApplication:
    locked = locked_ranges or []
    parser_instruction = _normalize_revision_phrasing(instruction)
    intent = parse_revision_intent(
        parser_instruction,
        base_duration_seconds=base_graph.selected_duration_seconds,
    )
    intent.instruction = instruction
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
            "notes": [
                *base_graph.notes,
                f"Revision v{next_version}: {instruction.strip()}",
                *intent.warnings,
            ],
            "critic_report": {},
        }
    )
    render_plan = compare_revision_graphs(base_graph, revised)
    render_plan.notes.extend(intent.warnings)
    return RevisionApplication(graph=revised, intent=intent, render_plan=render_plan)
