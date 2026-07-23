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
