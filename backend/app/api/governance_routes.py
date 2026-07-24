from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import InvalidTokenError, sign_payload, verify_signed_payload
from app.core.time import as_utc, is_expired_at
from app.models.governance import PrivacyRequest
from app.models.platform import Workspace
from app.schemas.governance import (
    PrivacyDeletionCreate,
    PrivacyDeliveryRead,
    PrivacyRequestRead,
)
from app.services.governance import generate_workspace_export, workspace_deletion_blockers
from app.services.permissions import can_manage_members, membership_for
from app.worker.governance import generate_workspace_export_task

router = APIRouter(tags=["privacy"])
settings = get_settings()


def _current_user_id(request: Request) -> UUID:
    user_id = getattr(request.state, "user_id", None)
    if not isinstance(user_id, UUID):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user_id


def _manager(db: Session, workspace_id: UUID, request: Request):
    membership = membership_for(db, workspace_id, _current_user_id(request))
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    if not can_manage_members(membership.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace administrator permission is required",
        )
    return membership


def _owner(db: Session, workspace_id: UUID, request: Request):
    membership = _manager(db, workspace_id, request)
    if membership.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace owner permission is required",
        )
    return membership


@router.get(
    "/workspaces/{workspace_id}/privacy/requests",
    response_model=list[PrivacyRequestRead],
)
def list_privacy_requests(
    workspace_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> list[PrivacyRequest]:
    _manager(db, workspace_id, request)
    return list(
        db.scalars(
            select(PrivacyRequest)
            .where(PrivacyRequest.workspace_id == workspace_id)
            .order_by(PrivacyRequest.created_at.desc())
            .limit(100)
        ).all()
    )


@router.post(
    "/workspaces/{workspace_id}/privacy/exports",
    response_model=PrivacyRequestRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_workspace_export(
    workspace_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> PrivacyRequest:
    membership = _manager(db, workspace_id, request)
    existing = db.scalar(
        select(PrivacyRequest).where(
            PrivacyRequest.workspace_id == workspace_id,
            PrivacyRequest.kind == "export",
            PrivacyRequest.status.in_(["queued", "processing"]),
        )
    )
    if existing is not None:
        return existing
    privacy_request = PrivacyRequest(
        workspace_id=workspace_id,
        requested_by_user_id=membership.user_id,
        kind="export",
        status="queued",
        request_metadata={"format": "zip+json", "includes_raw_media": False},
    )
    db.add(privacy_request)
    db.commit()
    db.refresh(privacy_request)
    if settings.environment.casefold() == "test":
        generate_workspace_export(db, privacy_request, settings)
        db.commit()
        db.refresh(privacy_request)
        return privacy_request
    try:
        generate_workspace_export_task.delay(str(privacy_request.id))
    except Exception as exc:
        privacy_request.status = "failed"
        privacy_request.error_message = "Export could not be queued"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workspace export could not be queued",
        ) from exc
    return privacy_request


@router.get(
    "/workspaces/{workspace_id}/privacy/requests/{request_id}/delivery",
    response_model=PrivacyDeliveryRead,
)
def create_privacy_delivery(
    workspace_id: UUID,
    request_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> PrivacyDeliveryRead:
    _manager(db, workspace_id, request)
    privacy_request = db.get(PrivacyRequest, request_id)
    if privacy_request is None or privacy_request.workspace_id != workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Privacy request not found")
    now = datetime.now(UTC)
    available_until = (
        as_utc(privacy_request.available_until) if privacy_request.available_until else None
    )
    if (
        privacy_request.status != "ready"
        or not privacy_request.result_path
        or not privacy_request.result_sha256
        or privacy_request.result_size_bytes is None
        or available_until is None
        or is_expired_at(available_until, now=now)
        or not Path(privacy_request.result_path).is_file()
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Export is not available")
    expires_at = min(
        available_until,
        now + timedelta(minutes=settings.delivery_link_minutes),
    )
    token = sign_payload(
        {
            "purpose": "privacy_export",
            "request_id": str(privacy_request.id),
            "exp": int(expires_at.timestamp()),
        },
        settings,
    )
    return PrivacyDeliveryRead(
        request_id=privacy_request.id,
        url=f"{settings.api_v1_prefix}/privacy-deliveries/{token}",
        expires_at=expires_at,
        sha256=privacy_request.result_sha256,
        size_bytes=privacy_request.result_size_bytes,
    )


@router.get("/privacy-deliveries/{token}", include_in_schema=False)
def download_privacy_export(token: str, db: Session = Depends(get_db)) -> FileResponse:
    try:
        payload = verify_signed_payload(token, settings)
        if payload.get("purpose") != "privacy_export":
            raise InvalidTokenError("Token has the wrong purpose")
        request_id = UUID(str(payload["request_id"]))
    except (InvalidTokenError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export not found") from exc
    privacy_request = db.get(PrivacyRequest, request_id)
    if (
        privacy_request is None
        or privacy_request.status != "ready"
        or not privacy_request.result_path
        or not privacy_request.available_until
        or is_expired_at(privacy_request.available_until)
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export not found")
    path = Path(privacy_request.result_path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export not found")
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"director-os-workspace-{privacy_request.workspace_id}.zip",
    )


@router.post(
    "/workspaces/{workspace_id}/privacy/deletion",
    response_model=PrivacyRequestRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def schedule_workspace_deletion(
    workspace_id: UUID,
    payload: PrivacyDeletionCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> PrivacyRequest:
    membership = _owner(db, workspace_id, request)
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    if payload.confirmation.strip() != workspace.slug:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Type the workspace slug '{workspace.slug}' to confirm deletion",
        )
    existing = db.scalar(
        select(PrivacyRequest).where(
            PrivacyRequest.workspace_id == workspace_id,
            PrivacyRequest.kind == "deletion",
            PrivacyRequest.status.in_(["scheduled", "processing"]),
        )
    )
    if existing is not None:
        return existing
    blockers = workspace_deletion_blockers(db, workspace_id)
    if blockers:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=" ".join(blockers))
    execute_after = datetime.now(UTC) + timedelta(days=settings.workspace_deletion_grace_days)
    privacy_request = PrivacyRequest(
        workspace_id=workspace_id,
        requested_by_user_id=membership.user_id,
        kind="deletion",
        status="scheduled",
        execute_after=execute_after,
        request_metadata={"reason": payload.reason, "confirmation": workspace.slug},
    )
    db.add(privacy_request)
    db.commit()
    db.refresh(privacy_request)
    return privacy_request


@router.delete(
    "/workspaces/{workspace_id}/privacy/deletion/{request_id}",
    response_model=PrivacyRequestRead,
)
def cancel_workspace_deletion(
    workspace_id: UUID,
    request_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> PrivacyRequest:
    _owner(db, workspace_id, request)
    privacy_request = db.get(PrivacyRequest, request_id)
    if (
        privacy_request is None
        or privacy_request.workspace_id != workspace_id
        or privacy_request.kind != "deletion"
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deletion request not found")
    if privacy_request.status != "scheduled":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only a scheduled deletion can be cancelled",
        )
    privacy_request.status = "cancelled"
    privacy_request.completed_at = datetime.now(UTC)
    db.commit()
    db.refresh(privacy_request)
    return privacy_request
