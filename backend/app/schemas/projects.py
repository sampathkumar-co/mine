from datetime import datetime
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.enums import AssetKind, ProjectStatus


class DirectorContract(BaseModel):
    objective: str = Field(min_length=3, max_length=500)
    target_audience: str | None = Field(default=None, max_length=500)
    target_platform: str = Field(default="instagram_reels", min_length=2, max_length=100)
    target_duration_seconds: int = Field(default=45, ge=5, le=600)
    tier: int = Field(default=1, ge=1, le=6)
    instructions: str | None = Field(default=None, max_length=10_000)
    must_include: list[str] = Field(default_factory=list, max_length=100)
    must_avoid: list[str] = Field(default_factory=list, max_length=100)
    reference_rules: dict[str, str] = Field(default_factory=dict)
    brand_rules: dict[str, Any] = Field(default_factory=dict)
    creative_freedom: float = Field(default=0.5, ge=0, le=1)
    director_profile_key: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    use_director_memory: bool = True

    @model_validator(mode="after")
    def requirements_must_not_conflict(self) -> Self:
        required = {item.strip().casefold() for item in self.must_include if item.strip()}
        prohibited = {item.strip().casefold() for item in self.must_avoid if item.strip()}
        conflicts = sorted(required & prohibited)
        if conflicts:
            raise ValueError(f"Director Contract has conflicting rules: {', '.join(conflicts)}")
        return self


class ProjectCreate(BaseModel):
    user_id: UUID
    contract: DirectorContract


class ProjectAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kind: AssetKind
    original_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    created_at: datetime


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    status: ProjectStatus
    contract: DirectorContract
    task_id: str | None
    output_available: bool
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    assets: list[ProjectAssetRead]


class ProjectAccepted(BaseModel):
    project_id: UUID
    status: ProjectStatus
    task_id: str
    message: str = "Project queued for autonomous production."


class ProjectIntelligenceRead(BaseModel):
    project_id: UUID
    analysis: dict[str, Any] | None
    edit_decision_graph: dict[str, Any] | None
    graph_version: int | None


class RevisionLockedRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    label: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("Locked range end must be after start")
        return self


class RevisionCreate(BaseModel):
    instruction: str = Field(min_length=3, max_length=5_000)
    base_version: int | None = Field(default=None, ge=1)
    locked_ranges: list[RevisionLockedRange] = Field(default_factory=list, max_length=50)


class RevisionAccepted(BaseModel):
    project_id: UUID
    version: int
    base_version: int
    status: str
    task_id: str
    message: str = "Revision queued for isolated rendering."


class RevisionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: int
    base_version: int | None
    instruction: str | None
    status: str
    task_id: str | None
    is_active: bool
    output_available: bool = False
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def derive_output_available(cls, value: Any) -> Any:
        if hasattr(value, "output_path"):
            return {
                "version": value.version,
                "base_version": value.base_version,
                "instruction": value.instruction,
                "status": value.status,
                "task_id": value.task_id,
                "is_active": value.is_active,
                "output_available": bool(value.output_path),
                "error_message": value.error_message,
                "created_at": value.created_at,
                "updated_at": value.updated_at,
            }
        return value


class RevisionDetail(RevisionSummary):
    graph: dict[str, Any]
    render_plan: dict[str, Any]
    critic_report: dict[str, Any]
    locked_ranges: list[dict[str, Any]]

    @model_validator(mode="before")
    @classmethod
    def expand_revision(cls, value: Any) -> Any:
        if hasattr(value, "graph_payload"):
            return {
                "version": value.version,
                "base_version": value.base_version,
                "instruction": value.instruction,
                "status": value.status,
                "task_id": value.task_id,
                "is_active": value.is_active,
                "output_available": bool(value.output_path),
                "error_message": value.error_message,
                "created_at": value.created_at,
                "updated_at": value.updated_at,
                "graph": value.graph_payload,
                "render_plan": value.render_plan,
                "critic_report": value.critic_report,
                "locked_ranges": value.locked_ranges,
            }
        return value


class RevisionComparison(BaseModel):
    project_id: UUID
    left_version: int
    right_version: int
    render_plan: dict[str, Any]


class RevisionActivated(BaseModel):
    project_id: UUID
    version: int
    status: str = "active"
    message: str = "Revision activated. Previous versions remain available for undo or redo."
