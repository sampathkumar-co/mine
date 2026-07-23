from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ProjectStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    RENDERING = "rendering"
    QUALITY_CHECK = "quality_check"
    READY = "ready"
    FAILED = "failed"


class DirectorContract(BaseModel):
    objective: str = Field(min_length=3, max_length=500)
    target_platform: str = "instagram_reels"
    target_duration_seconds: int = Field(default=45, ge=5, le=600)
    tier: int = Field(default=1, ge=1, le=6)
    instructions: str | None = Field(default=None, max_length=10_000)
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    creative_freedom: float = Field(default=0.5, ge=0, le=1)


class ProjectCreate(BaseModel):
    user_id: UUID
    contract: DirectorContract


class ProjectAccepted(BaseModel):
    project_id: UUID = Field(default_factory=uuid4)
    status: ProjectStatus = ProjectStatus.QUEUED
    message: str = "Project accepted for autonomous production."
