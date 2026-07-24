from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

from app.core.enums import AssetKind
from app.schemas.platform import UserRead, WorkspaceRead

WorkspaceRole = Literal["owner", "admin", "editor", "viewer"]


class VerificationConfirm(BaseModel):
    token: str = Field(min_length=20, max_length=512)


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str = Field(min_length=20, max_length=512)
    new_password: str = Field(min_length=10, max_length=256)


class SimpleMessage(BaseModel):
    message: str


class AccountContextRead(BaseModel):
    user: UserRead
    workspaces: list[WorkspaceRead]


class WorkspaceMemberRead(BaseModel):
    id: UUID
    user_id: UUID
    email: str
    display_name: str
    role: WorkspaceRole
    created_at: datetime


class WorkspaceMemberUpdate(BaseModel):
    role: WorkspaceRole


class WorkspaceInvitationCreate(BaseModel):
    email: EmailStr
    role: WorkspaceRole = "editor"

    @model_validator(mode="after")
    def owner_invites_are_not_allowed(self) -> Self:
        if self.role == "owner":
            raise ValueError("Ownership must be transferred explicitly, not invited")
        return self


class WorkspaceInvitationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    workspace_id: UUID
    email: str
    role: WorkspaceRole
    expires_at: datetime
    accepted_at: datetime | None
    revoked_at: datetime | None
    created_at: datetime


class WorkspaceInvitationAccepted(BaseModel):
    workspace_id: UUID
    membership_id: UUID
    role: WorkspaceRole
    message: str = "Workspace invitation accepted."


class InvitationAcceptRequest(BaseModel):
    token: str = Field(min_length=20, max_length=512)


class AuditEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    workspace_id: UUID | None
    actor_user_id: UUID | None
    action: str
    resource_type: str
    resource_id: str | None
    request_id: str | None
    ip_address: str | None
    user_agent: str | None
    payload: dict[str, object]
    created_at: datetime


class BillingAccountRead(BaseModel):
    workspace_id: UUID
    plan: str
    balance_credits: Decimal
    reserved_credits: Decimal
    available_credits: Decimal
    updated_at: datetime


class BillingEntryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    workspace_id: UUID
    project_id: UUID | None
    actor_user_id: UUID | None
    kind: str
    amount_credits: Decimal
    idempotency_key: str
    description: str
    entry_metadata: dict[str, object]
    created_at: datetime


class MultipartUploadCreate(BaseModel):
    kind: AssetKind = AssetKind.SOURCE_VIDEO
    original_filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=3, max_length=255)
    total_bytes: int = Field(gt=0)
    part_size: int | None = Field(default=None, ge=5_242_880, le=67_108_864)


class MultipartUploadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    asset_id: UUID | None
    provider: str
    object_key: str
    kind: str
    original_filename: str
    content_type: str
    total_bytes: int
    part_size: int
    status: str
    error_message: str | None
    expires_at: datetime
    created_at: datetime
    updated_at: datetime


class MultipartPartTargetRead(BaseModel):
    upload_id: UUID
    part_number: int
    expected_size: int
    method: str
    url: str
    headers: dict[str, str]


class MultipartPartRegister(BaseModel):
    part_number: int = Field(ge=1, le=10_000)
    etag: str = Field(min_length=1, max_length=300)
    size_bytes: int = Field(gt=0)


class MultipartCompleteRequest(BaseModel):
    parts: list[MultipartPartRegister] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def parts_must_be_unique(self) -> Self:
        numbers = [part.part_number for part in self.parts]
        if len(numbers) != len(set(numbers)):
            raise ValueError("Multipart part numbers must be unique")
        return self


class MultipartPartRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    part_number: int
    etag: str
    size_bytes: int


class MultipartUploadDetail(MultipartUploadRead):
    parts: list[MultipartPartRead]
