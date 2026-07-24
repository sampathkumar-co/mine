from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.director.memory import MemorySignal, normalise_profile_key, update_memory_state
from app.models.memory import (
    DirectorMemoryEvidence,
    DirectorMemoryProfile,
    ProjectPerformanceSignal,
)


def get_or_create_memory_profile(
    db: Session,
    *,
    user_id: UUID,
    profile_key: str | None,
) -> DirectorMemoryProfile:
    normalized = normalise_profile_key(profile_key)
    profile = db.scalar(
        select(DirectorMemoryProfile).where(
            DirectorMemoryProfile.user_id == user_id,
            DirectorMemoryProfile.profile_key == normalized,
        )
    )
    if profile is None:
        profile = DirectorMemoryProfile(user_id=user_id, profile_key=normalized)
        db.add(profile)
        db.flush()
    return profile


def apply_memory_evidence(
    db: Session,
    *,
    profile: DirectorMemoryProfile,
    signals: list[MemorySignal],
    project_id: UUID | None,
    revision_version: int | None,
    event_type: str,
    verdict: str | None,
    feedback_text: str | None,
    payload: dict[str, Any],
    weight: float,
    performance_sample: ProjectPerformanceSignal | None = None,
) -> DirectorMemoryEvidence:
    preferences, negative = update_memory_state(
        profile.preferences,
        profile.negative_preferences,
        signals,
    )
    profile.preferences = preferences
    profile.negative_preferences = negative
    profile.evidence_count += 1
    if performance_sample is not None:
        profile.performance_sample_count += 1
        db.add(performance_sample)
    profile.version += 1

    evidence = DirectorMemoryEvidence(
        profile_id=profile.id,
        project_id=project_id,
        revision_version=revision_version,
        event_type=event_type,
        verdict=verdict,
        weight=weight,
        feedback_text=feedback_text,
        payload={
            **payload,
            "signals": [signal.model_dump(mode="json") for signal in signals],
        },
    )
    db.add(evidence)
    db.commit()
    db.refresh(profile)
    db.refresh(evidence)
    if performance_sample is not None:
        db.refresh(performance_sample)
    return evidence


def memory_profile_key(contract: dict[str, Any]) -> str:
    return normalise_profile_key(str(contract.get("director_profile_key") or "default"))
