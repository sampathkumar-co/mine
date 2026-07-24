from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.director.memory import (
    ALLOWED_EXPLICIT_PREFERENCES,
    MemorySignal,
    calculate_performance_score,
    explicit_preference_signals,
    extract_text_memory_signals,
    graph_memory_signals,
    normalise_profile_key,
    performance_memory_signals,
)
from app.director.memory_store import (
    apply_memory_evidence,
    get_or_create_memory_profile,
    memory_profile_key,
)
from app.director.revision_engine import RevisionEditDecisionGraph, normalize_revision_graph
from app.models.analysis import EditDecisionGraphRecord, EditGraphRevision, ProjectAnalysis
from app.models.memory import ProjectPerformanceSignal
from app.models.project import Project
from app.schemas.memory import (
    DirectorMemoryPreferenceUpdate,
    DirectorMemoryPreferenceUpdated,
    DirectorMemoryProfileRead,
    ProjectFeedbackCreate,
    ProjectFeedbackRecorded,
    ProjectPerformanceCreate,
    ProjectPerformanceRecorded,
)

router = APIRouter()


def _get_project(db: Session, project_id: UUID) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


def _resolve_graph(
    db: Session,
    project: Project,
    revision_version: int | None,
) -> tuple[RevisionEditDecisionGraph, int, dict[str, Any]]:
    active = db.scalar(
        select(EditDecisionGraphRecord).where(EditDecisionGraphRecord.project_id == project.id)
    )
    if active is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project has no Edit Decision Graph to evaluate",
        )
    version = revision_version or active.version
    revision = db.scalar(
        select(EditGraphRevision).where(
            EditGraphRevision.project_id == project.id,
            EditGraphRevision.version == version,
        )
    )
    if revision is not None:
        if revision.status != "ready" or not revision.graph_payload:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only ready revisions can receive feedback or performance signals",
            )
        return normalize_revision_graph(revision.graph_payload), version, revision.render_plan
    if version != active.version:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Revision not found")
    return normalize_revision_graph(active.payload), version, {}


def _production_style(db: Session, project_id: UUID) -> dict[str, Any]:
    analysis = db.scalar(select(ProjectAnalysis).where(ProjectAnalysis.project_id == project_id))
    if analysis is None:
        return {}
    style = analysis.payload.get("production_style")
    return style if isinstance(style, dict) else {}


def _feedback_weight(payload: ProjectFeedbackCreate) -> float:
    if payload.verdict == "accepted":
        return {1: 0.5, 2: 0.65, 3: 0.85, 4: 1.2, 5: 1.5}.get(payload.rating or 4, 1.2)
    if payload.verdict == "rejected":
        return {1: 1.7, 2: 1.5, 3: 1.2, 4: 0.9, 5: 0.7}.get(payload.rating or 1, 1.5)
    return 0.8


def _dimension_rating_signals(
    graph: RevisionEditDecisionGraph,
    style: dict[str, Any],
    ratings: dict[str, int],
) -> list[MemorySignal]:
    if not ratings:
        return []
    current = graph_memory_signals(
        graph,
        style,
        sentiment="positive",
        weight=1.0,
        source="dimension_rating_fixture",
    )
    by_dimension = {signal.dimension: signal for signal in current}
    signals: list[MemorySignal] = []
    for key, rating in ratings.items():
        dimension = ALLOWED_EXPLICIT_PREFERENCES.get(key, key)
        existing = by_dimension.get(dimension)
        if existing is None or rating == 3:
            continue
        signals.append(
            existing.model_copy(
                update={
                    "sentiment": "positive" if rating >= 4 else "negative",
                    "weight": 1.2 if rating in {1, 5} else 0.8,
                    "source": "dimension_rating",
                    "reason": f"User rated {dimension} {rating}/5.",
                    "explicit": True,
                }
            )
        )
    return signals


@router.get(
    "/users/{user_id}/director-memory/{profile_key}",
    response_model=DirectorMemoryProfileRead,
    tags=["director-memory"],
)
def get_director_memory_profile(
    user_id: UUID,
    profile_key: str,
    db: Session = Depends(get_db),
):
    profile = get_or_create_memory_profile(
        db,
        user_id=user_id,
        profile_key=profile_key,
    )
    db.commit()
    db.refresh(profile)
    return profile


@router.post(
    "/users/{user_id}/director-memory/{profile_key}/preferences",
    response_model=DirectorMemoryPreferenceUpdated,
    tags=["director-memory"],
)
def update_director_memory_preferences(
    user_id: UUID,
    profile_key: str,
    payload: DirectorMemoryPreferenceUpdate,
    db: Session = Depends(get_db),
) -> DirectorMemoryPreferenceUpdated:
    profile = get_or_create_memory_profile(
        db,
        user_id=user_id,
        profile_key=profile_key,
    )
    signals = explicit_preference_signals(payload.preferences)
    for key, value in payload.avoid_preferences.items():
        dimension = ALLOWED_EXPLICIT_PREFERENCES.get(key)
        if dimension is None:
            continue
        signals.append(
            MemorySignal(
                dimension=dimension,
                value=value,
                sentiment="negative",
                weight=2.3,
                source="explicit_avoid_preference",
                reason=f"User explicitly asked to avoid {key}={value!r}.",
                explicit=True,
            )
        )
    if not signals:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No supported Director Memory preference keys were supplied",
        )
    apply_memory_evidence(
        db,
        profile=profile,
        signals=signals,
        project_id=None,
        revision_version=None,
        event_type="manual_preference",
        verdict="preferred",
        feedback_text=payload.note,
        payload={
            "preferences": payload.preferences,
            "avoid_preferences": payload.avoid_preferences,
        },
        weight=2.3,
    )
    return DirectorMemoryPreferenceUpdated(
        user_id=user_id,
        profile_key=profile.profile_key,
        profile_version=profile.version,
        signal_count=len(signals),
        updated_dimensions=sorted({signal.dimension for signal in signals}),
    )


@router.get(
    "/projects/{project_id}/director-memory",
    response_model=DirectorMemoryProfileRead,
    tags=["director-memory"],
)
def get_project_director_memory(
    project_id: UUID,
    db: Session = Depends(get_db),
):
    project = _get_project(db, project_id)
    profile = get_or_create_memory_profile(
        db,
        user_id=project.user_id,
        profile_key=memory_profile_key(project.contract),
    )
    db.commit()
    db.refresh(profile)
    return profile


@router.post(
    "/projects/{project_id}/feedback",
    response_model=ProjectFeedbackRecorded,
    status_code=status.HTTP_201_CREATED,
    tags=["director-memory"],
)
def record_project_feedback(
    project_id: UUID,
    payload: ProjectFeedbackCreate,
    db: Session = Depends(get_db),
) -> ProjectFeedbackRecorded:
    project = _get_project(db, project_id)
    graph, version, render_plan = _resolve_graph(db, project, payload.revision_version)
    style = _production_style(db, project_id)
    profile = get_or_create_memory_profile(
        db,
        user_id=project.user_id,
        profile_key=memory_profile_key(project.contract),
    )
    sentiment = "positive" if payload.verdict == "accepted" else "negative"
    changed_components = {
        str(item) for item in render_plan.get("changed_components", []) if str(item)
    }
    signals = graph_memory_signals(
        graph,
        style,
        sentiment=sentiment,
        weight=_feedback_weight(payload),
        source=f"revision_{payload.verdict}",
        changed_components=changed_components or None,
    )
    signals.extend(extract_text_memory_signals(payload.feedback_text))
    signals.extend(explicit_preference_signals(payload.explicit_preferences))
    signals.extend(_dimension_rating_signals(graph, style, payload.dimension_ratings))

    evidence = apply_memory_evidence(
        db,
        profile=profile,
        signals=signals,
        project_id=project_id,
        revision_version=version,
        event_type="project_feedback",
        verdict=payload.verdict,
        feedback_text=payload.feedback_text,
        payload={
            "rating": payload.rating,
            "explicit_preferences": payload.explicit_preferences,
            "dimension_ratings": payload.dimension_ratings,
            "render_plan": render_plan,
        },
        weight=_feedback_weight(payload),
    )
    return ProjectFeedbackRecorded(
        evidence_id=evidence.id,
        project_id=project_id,
        revision_version=version,
        profile_key=profile.profile_key,
        profile_version=profile.version,
        signal_count=len(signals),
        updated_dimensions=sorted({signal.dimension for signal in signals}),
    )


@router.post(
    "/projects/{project_id}/performance",
    response_model=ProjectPerformanceRecorded,
    status_code=status.HTTP_201_CREATED,
    tags=["director-memory"],
)
def record_project_performance(
    project_id: UUID,
    payload: ProjectPerformanceCreate,
    db: Session = Depends(get_db),
) -> ProjectPerformanceRecorded:
    project = _get_project(db, project_id)
    graph, version, _ = _resolve_graph(db, project, payload.revision_version)
    style = _production_style(db, project_id)
    profile = get_or_create_memory_profile(
        db,
        user_id=project.user_id,
        profile_key=memory_profile_key(project.contract),
    )
    metrics = payload.metrics_payload()
    score = calculate_performance_score(
        metrics,
        video_duration_seconds=graph.selected_duration_seconds,
    )
    signals = performance_memory_signals(graph, style, score=score)
    performance = ProjectPerformanceSignal(
        profile_id=profile.id,
        project_id=project_id,
        revision_version=version,
        platform=normalise_profile_key(payload.platform),
        normalized_score=score,
        metrics=metrics,
        published_at=payload.published_at,
    )
    apply_memory_evidence(
        db,
        profile=profile,
        signals=signals,
        project_id=project_id,
        revision_version=version,
        event_type="performance",
        verdict="strong" if score >= 0.68 else "weak" if score <= 0.3 else "neutral",
        feedback_text=None,
        payload={"platform": payload.platform, "score": score, "metrics": metrics},
        weight=0.4,
        performance_sample=performance,
    )
    return ProjectPerformanceRecorded(
        signal_id=performance.id,
        project_id=project_id,
        revision_version=version,
        profile_key=profile.profile_key,
        normalized_score=score,
        memory_signal_count=len(signals),
        profile_version=profile.version,
    )
