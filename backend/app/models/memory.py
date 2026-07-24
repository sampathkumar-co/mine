from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
    select,
)
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.project import Project


def utc_now() -> datetime:
    return datetime.now(UTC)


class DirectorMemoryProfile(Base):
    __tablename__ = "director_memory_profiles"
    __table_args__ = (
        UniqueConstraint("user_id", "profile_key", name="uq_director_memory_user_profile"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), index=True)
    profile_key: Mapped[str] = mapped_column(String(100), default="default", index=True)
    preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    negative_preferences: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    performance_sample_count: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class DirectorMemoryEvidence(Base):
    __tablename__ = "director_memory_evidence"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(
        ForeignKey("director_memory_profiles.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    revision_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ProjectPerformanceSignal(Base):
    __tablename__ = "project_performance_signals"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(
        ForeignKey("director_memory_profiles.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    revision_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    platform: Mapped[str] = mapped_column(String(80), index=True)
    normalized_score: Mapped[float] = mapped_column(Float, default=0.0)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


@event.listens_for(Project, "before_insert")
def compile_memory_for_new_project(
    _: Any,
    connection: Connection,
    project: Project,
) -> None:
    contract = dict(project.contract or {})
    if not bool(contract.get("use_director_memory", True)):
        return

    from app.director.memory import apply_director_memory, normalise_profile_key

    profile_key = normalise_profile_key(str(contract.get("director_profile_key") or "default"))
    profile_row = connection.execute(
        select(
            DirectorMemoryProfile.preferences,
            DirectorMemoryProfile.version,
        ).where(
            DirectorMemoryProfile.user_id == project.user_id,
            DirectorMemoryProfile.profile_key == profile_key,
        )
    ).first()
    if profile_row is None:
        return

    application = apply_director_memory(contract, profile_row.preferences)
    effective_contract = application.contract
    brand_rules = dict(effective_contract.get("brand_rules") or {})
    if application.captions_enabled is not None:
        brand_rules["captions_enabled"] = application.captions_enabled
    if (
        application.max_visual_overlays is not None
        and "max_visual_overlays" not in brand_rules
    ):
        brand_rules["max_visual_overlays"] = application.max_visual_overlays
    effective_contract["brand_rules"] = brand_rules
    effective_contract["_director_memory_application"] = {
        "profile_key": profile_key,
        "profile_version": profile_row.version,
        "applied": application.applied,
        "skipped": application.skipped,
    }
    project.contract = effective_contract
