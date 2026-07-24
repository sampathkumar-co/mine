from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

PrivacyRequestKind = Literal["export", "deletion"]
PrivacyRequestStatus = Literal[
    "queued",
    "processing",
    "ready",
    "scheduled",
    "completed",
    "failed",
    "cancelled",
]


class PrivacyRequestRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    workspace_id: UUID
    requested_by_user_id: UUID | None
    kind: PrivacyRequestKind
    status: PrivacyRequestStatus
    result_sha256: str | None
    result_size_bytes: int | None
    available_until: datetime | None
    execute_after: datetime | None
    completed_at: datetime | None
    error_message: str | None
    request_metadata: dict[str, object]
    created_at: datetime
    updated_at: datetime


class PrivacyDeletionCreate(BaseModel):
    confirmation: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=3, max_length=500)


class PrivacyDeliveryRead(BaseModel):
    request_id: UUID
    url: str
    expires_at: datetime
    sha256: str
    size_bytes: int


class ServiceComponentRead(BaseModel):
    status: Literal["ok", "degraded", "failed"]
    detail: str | None = None


class ReadinessRead(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    version: str
    components: dict[str, ServiceComponentRead]
