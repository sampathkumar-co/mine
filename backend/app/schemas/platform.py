from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.core.enums import AssetKind, ProjectStatus


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=256)
    display_name: str = Field(min_length=2, max_length=160)
    workspace_name: str = Field(default="My Workspace", min_length=2, max_length=180)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    display_name: str
    created_at: datetime


class WorkspaceRead(BaseModel):
    id: UUID
    name: str
    slug: str
    role: str
    created_at: datetime


class AuthSession(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserRead
    workspaces: list[WorkspaceRead]


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=2, max_length=180)


class WorkspaceProjectRead(BaseModel):
    id: UUID
    workspace_id: UUID
    status: ProjectStatus
    objective: str
    target_platform: str
    target_duration_seconds: int
    output_available: bool
    asset_count: int
    created_at: datetime
    updated_at: datetime


class ResumableUploadCreate(BaseModel):
    kind: AssetKind = AssetKind.SOURCE_VIDEO
    original_filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=3, max_length=255)
    total_bytes: int = Field(gt=0)


class ResumableUploadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    asset_id: UUID | None
    kind: str
    original_filename: str
    content_type: str
    total_bytes: int
    received_bytes: int
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DeliveryLinkRead(BaseModel):
    project_id: UUID
    revision_version: int | None
    url: str
    expires_at: datetime
    download: bool
