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
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class DirectorCameraAudit(Base):
    __tablename__ = "director_camera_audits"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_director_camera_project_version"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    mode: Mapped[str] = mapped_column(String(24), default="advisory")
    readiness_score: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.72)
    ready: Mapped[bool] = mapped_column(default=False)
    report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PickupMission(Base):
    __tablename__ = "pickup_missions"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    audit_id: Mapped[UUID] = mapped_column(
        ForeignKey("director_camera_audits.id", ondelete="CASCADE"), index=True
    )
    mission_type: Mapped[str] = mapped_column(String(40), index=True)
    priority: Mapped[str] = mapped_column(String(24), index=True)
    title: Mapped[str] = mapped_column(String(240))
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="requested", index=True)
    specification: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    target_terms: Mapped[list[str]] = mapped_column(JSON, default=list)
    submitted_asset_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("project_assets.id", ondelete="SET NULL"), nullable=True, index=True
    )
    accepted_asset_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("project_assets.id", ondelete="SET NULL"), nullable=True, index=True
    )
    validation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
