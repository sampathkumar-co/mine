from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.director.semantic_overlays import (
    ProductionEditDecisionGraph,
    VisualOverlay,
)
from app.sensory.models import AnalysisBundle


class CriticIssue(BaseModel):
    code: str
    severity: Literal["warning", "blocking"]
    message: str


class EditorialCriticReport(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=1)
    issues: list[CriticIssue] = Field(default_factory=list)
    repairs_applied: list[str] = Field(default_factory=list)
    checked_rules: list[str] = Field(default_factory=list)


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.casefold()))


def _selected_content(graph: ProductionEditDecisionGraph, analysis: AnalysisBundle) -> str:
    clips_by_id = {clip.asset_id: clip for clip in analysis.source_clips}
    parts = [segment.transcript_text or "" for segment in graph.segments]
    for overlay in graph.overlays:
        clip = clips_by_id.get(overlay.source_asset_id)
        if clip is None:
            continue
        parts.extend(clip.evidence_terms)
        parts.extend(tag.label.replace("_", " ") for tag in clip.semantic_tags)
        if clip.transcript:
            parts.append(clip.transcript.text)
    return _normalise(" ".join(parts))


def _clean_overlays(
    graph: ProductionEditDecisionGraph,
    analysis: AnalysisBundle,
) -> tuple[list[VisualOverlay], list[str], list[CriticIssue]]:
    clips_by_id = {clip.asset_id: clip for clip in analysis.source_clips}
    accepted: list[VisualOverlay] = []
    repairs: list[str] = []
    issues: list[CriticIssue] = []

    for overlay in sorted(graph.overlays, key=lambda item: (-item.match_score, item.output_start)):
        clip = clips_by_id.get(overlay.source_asset_id)
        if clip is None or clip.role == "rejected" or clip.duplicate_of_asset_id:
            repairs.append(f"Removed overlay from unavailable or rejected asset {overlay.source_asset_id}.")
            continue
        if overlay.output_start < 0 or overlay.output_end > graph.selected_duration_seconds + 0.05:
            repairs.append(f"Removed out-of-bounds overlay from {clip.filename}.")
            continue
        if overlay.source_end <= overlay.source_start or overlay.output_end <= overlay.output_start:
            repairs.append(f"Removed zero-duration overlay from {clip.filename}.")
            continue
        if any(
            overlay.output_start < existing.output_end
            and overlay.output_end > existing.output_start
            for existing in accepted
        ):
            repairs.append(f"Removed overlapping lower-priority overlay from {clip.filename}.")
            continue
        if overlay.match_score < 0.2:
            repairs.append(f"Removed weakly matched overlay from {clip.filename}.")
            continue
        accepted.append(overlay)

    accepted.sort(key=lambda item: item.output_start)
    if graph.overlays and not accepted:
        issues.append(
            CriticIssue(
                code="all_overlays_removed",
                severity="warning",
                message="All planned visual overlays were removed by validation.",
            )
        )
    return accepted, repairs, issues


def review_and_repair_edit_graph(
    graph: ProductionEditDecisionGraph,
    analysis: AnalysisBundle,
    contract: dict[str, Any],
) -> tuple[ProductionEditDecisionGraph, EditorialCriticReport]:
    issues: list[CriticIssue] = []
    checked_rules = [
        "renderable_segments",
        "source_integrity",
        "overlay_bounds",
        "overlay_overlap",
        "duration_contract",
        "must_include",
        "must_avoid",
        "continuity_risk",
        "evidence_coverage",
    ]
    clips_by_id = {clip.asset_id: clip for clip in analysis.source_clips}

    if not graph.segments:
        issues.append(
            CriticIssue(
                code="no_renderable_segments",
                severity="blocking",
                message="The edit contains no renderable narration or visual segment.",
            )
        )

    for segment in graph.segments:
        clip = clips_by_id.get(segment.source_asset_id or "")
        if clip is None:
            issues.append(
                CriticIssue(
                    code="unknown_source_asset",
                    severity="blocking",
                    message=f"Segment references unknown source asset {segment.source_asset_id}.",
                )
            )
        elif clip.role == "rejected" or clip.duplicate_of_asset_id:
            issues.append(
                CriticIssue(
                    code="rejected_source_used",
                    severity="blocking",
                    message=f"Segment uses rejected or duplicate source asset {clip.asset_id}.",
                )
            )

    cleaned_overlays, repairs, overlay_issues = _clean_overlays(graph, analysis)
    issues.extend(overlay_issues)
    repaired_graph = graph.model_copy(update={"overlays": cleaned_overlays})

    target_duration = float(contract.get("target_duration_seconds", graph.target_duration_seconds) or 0)
    if target_duration > 0 and repaired_graph.selected_duration_seconds > target_duration * 1.08 + 0.25:
        issues.append(
            CriticIssue(
                code="duration_contract_exceeded",
                severity="blocking",
                message="The planned edit materially exceeds the Director Contract duration.",
            )
        )

    selected_content = _selected_content(repaired_graph, analysis)
    must_include = contract.get("must_include") or []
    must_avoid = contract.get("must_avoid") or []
    for requirement in must_include:
        normalised = _normalise(str(requirement))
        if normalised and normalised not in selected_content:
            issues.append(
                CriticIssue(
                    code="missing_required_content",
                    severity="blocking",
                    message=f"Required content is not represented in the selected edit: {requirement}",
                )
            )
    for prohibition in must_avoid:
        normalised = _normalise(str(prohibition))
        if normalised and normalised in selected_content:
            issues.append(
                CriticIssue(
                    code="prohibited_content_selected",
                    severity="blocking",
                    message=f"Prohibited content appears in the selected edit: {prohibition}",
                )
            )

    low_continuity = [
        decision for decision in repaired_graph.continuity_decisions if decision.score < 0.35
    ]
    if low_continuity:
        issues.append(
            CriticIssue(
                code="low_continuity_transitions",
                severity="warning",
                message=(
                    f"{len(low_continuity)} transition(s) have low visual continuity and should use "
                    "a cutaway or soft transition."
                ),
            )
        )

    objective = _normalise(str(contract.get("objective", "")))
    evidence_requested = any(
        term in objective
        for term in {"proof", "prove", "evidence", "demo", "demonstrate", "result", "results"}
    )
    if evidence_requested and not repaired_graph.overlays:
        issues.append(
            CriticIssue(
                code="missing_evidence_overlay",
                severity="warning",
                message="The objective asks for proof or demonstration, but no matched evidence overlay was available.",
            )
        )

    blocking_count = sum(issue.severity == "blocking" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    score = max(0.0, 1.0 - blocking_count * 0.35 - warning_count * 0.08)
    report = EditorialCriticReport(
        passed=blocking_count == 0,
        score=round(score, 3),
        issues=issues,
        repairs_applied=repairs,
        checked_rules=checked_rules,
    )
    repaired_graph = repaired_graph.model_copy(
        update={
            "critic_report": report.model_dump(mode="json"),
            "notes": [
                *repaired_graph.notes,
                f"Editorial critic score: {report.score:.2f}; applied {len(repairs)} repair(s).",
            ],
        }
    )
    return repaired_graph, report
