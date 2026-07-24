from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import hash_password
from app.models.operations import (
    AuditEvent,
    BillingAccount,
    BillingEntry,
    MultipartUpload,
    MultipartUploadPart,
    UserEmailStatus,
    WorkspaceInvitation,
)
from app.models.platform import User, WorkspaceMembership
from app.models.project import Project
from app.schemas.operations import (
    AuditEventRead,
    BillingAccountRead,
    BillingAdjustmentCreate,
    BillingEntryRead,
    InvitationAcceptRequest,
    LogoutRequest,
    MultipartCompleteRequest,
    MultipartPartRegister,
    MultipartPartTargetRead,
    MultipartUploadCreate,
    MultipartUploadDetail,
    MultipartUploadRead,
    PasswordResetConfirm,
    PasswordResetRequest,
    RefreshRequest,
    SimpleMessage,
    VerificationConfirm,
    WorkspaceInvitationAccepted,
    WorkspaceInvitationCreate,
    WorkspaceInvitationRead,
    WorkspaceMemberRead,
    WorkspaceMemberUpdate,
)
from app.schemas.platform import AuthSession, UserRead, WorkspaceRead
from app.services.audit import record_audit
from app.services.billing import (
    InsufficientCreditsError,
    credits,
    ensure_billing_account,
    post_adjustment,
)
from app.services.email import (
    AccountTokenError,
    consume_account_token,
    email_is_verified,
    ensure_email_status,
    mark_email_verified,
    queue_email,
    queue_password_reset_email,
    queue_verification_email,
)
from app.services.permissions import can_manage_members, membership_for, normalize_role
from app.services.sessions import (
    SessionError,
    issue_session,
    revoke_all_sessions,
    revoke_session,
    rotate_session,
    token_hash,
)
from app.services.storage import (
    StorageError,
    cleanup_expired_multipart_upload,
    expected_part_size,
    finalize_upload_asset,
    is_expired,
    safe_object_key,
    storage_adapter,
    upsert_part,
    validate_part_number,
)
from app.storage.uploads import UnsupportedAssetError, _validate_content_type

router = APIRouter()
settings = get_settings()
EDITABLE_PROJECT_STATUSES = {"created", "uploading", "ready_to_queue", "failed"}


def _current_user_id(request: Request) -> UUID:
    user_id = getattr(request.state, "user_id", None)
    if not isinstance(user_id, UUID):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user_id


def _current_session_id(request: Request) -> UUID | None:
    session_id = getattr(request.state, "session_id", None)
    return session_id if isinstance(session_id, UUID) else None


def _workspace_read(membership: WorkspaceMembership) -> WorkspaceRead:
    return WorkspaceRead(
        id=membership.workspace.id,
        name=membership.workspace.name,
        slug=membership.workspace.slug,
        role=membership.role,
        created_at=membership.workspace.created_at,
    )


def _session_response(db: Session, user: User, issued) -> AuthSession:
    memberships = list(
        db.scalars(
            select(WorkspaceMembership)
            .where(WorkspaceMembership.user_id == user.id)
            .order_by(WorkspaceMembership.created_at)
        ).all()
    )
    return AuthSession(
        access_token=issued.access_token,
        refresh_token=issued.refresh_token,
        expires_at=issued.access_expires_at,
        refresh_expires_at=issued.refresh_expires_at,
        session_id=issued.record.id,
        user=UserRead(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            email_verified=email_is_verified(db, user.id),
            created_at=user.created_at,
        ),
        workspaces=[_workspace_read(item) for item in memberships],
    )


def _require_workspace_manager(db: Session, request: Request, workspace_id: UUID) -> WorkspaceMembership:
    membership = membership_for(db, workspace_id, _current_user_id(request))
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    if not can_manage_members(membership.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace administrator permission is required",
        )
    return membership


def _last_owner(db: Session, workspace_id: UUID, membership: WorkspaceMembership) -> bool:
    if membership.role != "owner":
        return False
    owner_count = db.scalar(
        select(func.count(WorkspaceMembership.id)).where(
            WorkspaceMembership.workspace_id == workspace_id,
            WorkspaceMembership.role == "owner",
        )
    )
    return int(owner_count or 0) <= 1


@router.post("/auth/session", response_model=AuthSession, tags=["authentication"])
def create_refreshable_session(
    request: Request,
    db: Session = Depends(get_db),
) -> AuthSession:
    user = db.get(User, _current_user_id(request))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    email_status = ensure_email_status(db, user.id)
    if email_status.verified_at is None and email_status.verification_sent_at is None:
        queue_verification_email(db, user, settings)
    issued = issue_session(db, user, settings, request=request)
    db.commit()
    return _session_response(db, user, issued)


@router.post("/auth/refresh", response_model=AuthSession, tags=["authentication"])
def refresh_session(
    payload: RefreshRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> AuthSession:
    try:
        issued = rotate_session(db, payload.refresh_token, settings, request=request)
    except SessionError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    user = db.get(User, issued.record.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    db.commit()
    return _session_response(db, user, issued)


@router.post("/auth/logout", response_model=SimpleMessage, tags=["authentication"])
def logout(
    payload: LogoutRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    user_id = _current_user_id(request)
    session_id = _current_session_id(request)
    if session_id is not None:
        revoke_session(db, session_id, user_id)
    elif payload.refresh_token:
        digest = token_hash(payload.refresh_token)
        from app.models.operations import AuthSessionRecord

        record = db.scalar(
            select(AuthSessionRecord).where(AuthSessionRecord.refresh_token_hash == digest)
        )
        if record and record.user_id == user_id:
            revoke_session(db, record.id, user_id)
    db.commit()
    return SimpleMessage(message="Session revoked.")


@router.post("/auth/logout-all", response_model=SimpleMessage, tags=["authentication"])
def logout_all(request: Request, db: Session = Depends(get_db)) -> SimpleMessage:
    count = revoke_all_sessions(db, _current_user_id(request))
    db.commit()
    return SimpleMessage(message=f"Revoked {count} session(s).")


@router.post(
    "/auth/email-verification/request",
    response_model=SimpleMessage,
    tags=["authentication"],
)
def request_email_verification(
    request: Request,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    user = db.get(User, _current_user_id(request))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if email_is_verified(db, user.id):
        return SimpleMessage(message="Email is already verified.")
    queue_verification_email(db, user, settings)
    db.commit()
    return SimpleMessage(message="Verification email queued.")


@router.post(
    "/auth/email-verification/confirm",
    response_model=SimpleMessage,
    tags=["authentication"],
)
def confirm_email_verification(
    payload: VerificationConfirm,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    try:
        user = consume_account_token(db, payload.token, "verify_email")
    except AccountTokenError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    mark_email_verified(db, user.id)
    db.commit()
    return SimpleMessage(message="Email verified.")


@router.post(
    "/auth/password-reset/request",
    response_model=SimpleMessage,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["authentication"],
)
def request_password_reset(
    payload: PasswordResetRequest,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    user = db.scalar(select(User).where(User.email == payload.email.strip().casefold()))
    if user is not None:
        queue_password_reset_email(db, user, settings)
        db.commit()
    return SimpleMessage(message="If the account exists, a reset email has been queued.")


@router.post(
    "/auth/password-reset/confirm",
    response_model=SimpleMessage,
    tags=["authentication"],
)
def confirm_password_reset(
    payload: PasswordResetConfirm,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    try:
        user = consume_account_token(db, payload.token, "reset_password")
    except AccountTokenError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    user.password_hash = hash_password(payload.new_password)
    mark_email_verified(db, user.id)
    revoke_all_sessions(db, user.id)
    db.commit()
    return SimpleMessage(message="Password changed and existing sessions revoked.")


@router.get(
    "/workspaces/{workspace_id}/members",
    response_model=list[WorkspaceMemberRead],
    tags=["workspaces"],
)
def list_workspace_members(
    workspace_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> list[WorkspaceMemberRead]:
    _require_workspace_manager(db, request, workspace_id)
    rows = db.execute(
        select(WorkspaceMembership, User)
        .join(User, User.id == WorkspaceMembership.user_id)
        .where(WorkspaceMembership.workspace_id == workspace_id)
        .order_by(WorkspaceMembership.created_at)
    ).all()
    return [
        WorkspaceMemberRead(
            id=membership.id,
            user_id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=membership.role,
            created_at=membership.created_at,
        )
        for membership, user in rows
    ]


@router.patch(
    "/workspaces/{workspace_id}/members/{membership_id}",
    response_model=WorkspaceMemberRead,
    tags=["workspaces"],
)
def update_workspace_member(
    workspace_id: UUID,
    membership_id: UUID,
    payload: WorkspaceMemberUpdate,
    request: Request,
    db: Session = Depends(get_db),
) -> WorkspaceMemberRead:
    actor = _require_workspace_manager(db, request, workspace_id)
    target = db.get(WorkspaceMembership, membership_id)
    if target is None or target.workspace_id != workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace member not found")
    role = normalize_role(payload.role)
    if (target.role == "owner" or role == "owner") and actor.role != "owner":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only an owner can change ownership")
    if target.role == "owner" and role != "owner" and _last_owner(db, workspace_id, target):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Workspace must keep an owner")
    target.role = role
    user = db.get(User, target.user_id)
    db.commit()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace user not found")
    return WorkspaceMemberRead(
        id=target.id,
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        role=target.role,
        created_at=target.created_at,
    )


@router.delete(
    "/workspaces/{workspace_id}/members/{membership_id}",
    response_model=SimpleMessage,
    tags=["workspaces"],
)
def remove_workspace_member(
    workspace_id: UUID,
    membership_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    actor = _require_workspace_manager(db, request, workspace_id)
    target = db.get(WorkspaceMembership, membership_id)
    if target is None or target.workspace_id != workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace member not found")
    if target.role == "owner" and actor.role != "owner":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only an owner can remove an owner")
    if _last_owner(db, workspace_id, target):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Workspace must keep an owner")
    db.delete(target)
    db.commit()
    return SimpleMessage(message="Workspace member removed.")


@router.post(
    "/workspaces/{workspace_id}/invitations",
    response_model=WorkspaceInvitationRead,
    status_code=status.HTTP_201_CREATED,
    tags=["workspaces"],
)
def create_workspace_invitation(
    workspace_id: UUID,
    payload: WorkspaceInvitationCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> WorkspaceInvitation:
    _require_workspace_manager(db, request, workspace_id)
    email = payload.email.strip().casefold()
    existing_user = db.scalar(select(User).where(User.email == email))
    if existing_user and membership_for(db, workspace_id, existing_user.id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User is already a member")
    now = datetime.now(UTC)
    open_invites = list(
        db.scalars(
            select(WorkspaceInvitation).where(
                WorkspaceInvitation.workspace_id == workspace_id,
                WorkspaceInvitation.email == email,
                WorkspaceInvitation.accepted_at.is_(None),
                WorkspaceInvitation.revoked_at.is_(None),
            )
        ).all()
    )
    for invitation in open_invites:
        invitation.revoked_at = now
    raw_token = secrets.token_urlsafe(40)
    invitation = WorkspaceInvitation(
        workspace_id=workspace_id,
        email=email,
        role=payload.role,
        token_hash=token_hash(raw_token),
        invited_by_user_id=_current_user_id(request),
        expires_at=now + timedelta(days=settings.invitation_days),
    )
    db.add(invitation)
    invite_url = f"{settings.public_app_url.rstrip('/')}/accept-invitation?token={raw_token}"
    queue_email(
        db,
        recipient=email,
        subject="You were invited to a Director OS workspace",
        body_text=f"Accept your Director OS workspace invitation:\n{invite_url}",
    )
    db.commit()
    db.refresh(invitation)
    return invitation


@router.get(
    "/workspaces/{workspace_id}/invitations",
    response_model=list[WorkspaceInvitationRead],
    tags=["workspaces"],
)
def list_workspace_invitations(
    workspace_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> list[WorkspaceInvitation]:
    _require_workspace_manager(db, request, workspace_id)
    return list(
        db.scalars(
            select(WorkspaceInvitation)
            .where(WorkspaceInvitation.workspace_id == workspace_id)
            .order_by(WorkspaceInvitation.created_at.desc())
        ).all()
    )


@router.delete(
    "/workspaces/{workspace_id}/invitations/{invitation_id}",
    response_model=SimpleMessage,
    tags=["workspaces"],
)
def revoke_workspace_invitation(
    workspace_id: UUID,
    invitation_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    _require_workspace_manager(db, request, workspace_id)
    invitation = db.get(WorkspaceInvitation, invitation_id)
    if invitation is None or invitation.workspace_id != workspace_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")
    if invitation.accepted_at is None:
        invitation.revoked_at = datetime.now(UTC)
        db.commit()
    return SimpleMessage(message="Invitation revoked.")


@router.post(
    "/invitations/accept",
    response_model=WorkspaceInvitationAccepted,
    tags=["workspaces"],
)
def accept_workspace_invitation(
    payload: InvitationAcceptRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> WorkspaceInvitationAccepted:
    user = db.get(User, _current_user_id(request))
    invitation = db.scalar(
        select(WorkspaceInvitation).where(
            WorkspaceInvitation.token_hash == token_hash(payload.token)
        )
    )
    now = datetime.now(UTC)
    if user is None or invitation is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invitation is invalid")
    if invitation.revoked_at or invitation.accepted_at or invitation.expires_at <= now:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invitation is no longer valid")
    if invitation.email != user.email:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invitation belongs to another email")
    membership = membership_for(db, invitation.workspace_id, user.id)
    if membership is None:
        membership = WorkspaceMembership(
            workspace_id=invitation.workspace_id,
            user_id=user.id,
            role=invitation.role,
        )
        db.add(membership)
        db.flush()
    invitation.accepted_at = now
    mark_email_verified(db, user.id)
    db.commit()
    return WorkspaceInvitationAccepted(
        workspace_id=invitation.workspace_id,
        membership_id=membership.id,
        role=membership.role,
    )


@router.get(
    "/workspaces/{workspace_id}/audit-events",
    response_model=list[AuditEventRead],
    tags=["audit"],
)
def list_audit_events(
    workspace_id: UUID,
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    action: str | None = Query(default=None, max_length=120),
    db: Session = Depends(get_db),
) -> list[AuditEvent]:
    _require_workspace_manager(db, request, workspace_id)
    query = select(AuditEvent).where(AuditEvent.workspace_id == workspace_id)
    if action:
        query = query.where(AuditEvent.action == action)
    return list(db.scalars(query.order_by(AuditEvent.created_at.desc()).limit(limit)).all())


@router.get(
    "/workspaces/{workspace_id}/billing",
    response_model=BillingAccountRead,
    tags=["billing"],
)
def get_billing_account(
    workspace_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> BillingAccountRead:
    membership = membership_for(db, workspace_id, _current_user_id(request))
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    account = ensure_billing_account(db, workspace_id)
    db.commit()
    return BillingAccountRead(
        workspace_id=workspace_id,
        plan=account.plan,
        balance_credits=account.balance_credits,
        reserved_credits=account.reserved_credits,
        available_credits=credits(account.balance_credits - account.reserved_credits),
        updated_at=account.updated_at,
    )


@router.get(
    "/workspaces/{workspace_id}/billing/entries",
    response_model=list[BillingEntryRead],
    tags=["billing"],
)
def list_billing_entries(
    workspace_id: UUID,
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[BillingEntry]:
    membership = membership_for(db, workspace_id, _current_user_id(request))
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    return list(
        db.scalars(
            select(BillingEntry)
            .where(BillingEntry.workspace_id == workspace_id)
            .order_by(BillingEntry.created_at.desc())
            .limit(limit)
        ).all()
    )


@router.post(
    "/workspaces/{workspace_id}/billing/adjustments",
    response_model=BillingEntryRead,
    status_code=status.HTTP_201_CREATED,
    tags=["billing"],
)
def create_billing_adjustment(
    workspace_id: UUID,
    payload: BillingAdjustmentCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> BillingEntry:
    membership = membership_for(db, workspace_id, _current_user_id(request))
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    if membership.role != "owner":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workspace owner permission is required")
    try:
        entry = post_adjustment(
            db,
            workspace_id,
            payload.amount_credits,
            idempotency_key=payload.idempotency_key,
            description=payload.description,
            actor_user_id=membership.user_id,
        )
    except InsufficientCreditsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    db.commit()
    db.refresh(entry)
    return entry


@router.post(
    "/projects/{project_id}/multipart-uploads",
    response_model=MultipartUploadRead,
    status_code=status.HTTP_201_CREATED,
    tags=["uploads"],
)
def create_multipart_upload(
    project_id: UUID,
    payload: MultipartUploadCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> MultipartUpload:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    if project.status.value not in EDITABLE_PROJECT_STATUSES:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Project assets are not editable")
    if payload.total_bytes > settings.max_upload_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Upload exceeds project limit")
    try:
        _validate_content_type(payload.kind, payload.content_type)
    except UnsupportedAssetError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    part_size = payload.part_size or settings.multipart_part_bytes
    upload = MultipartUpload(
        project_id=project.id,
        created_by_user_id=_current_user_id(request),
        provider=settings.object_storage_provider.casefold(),
        object_key=safe_object_key(project.id, payload.original_filename),
        kind=payload.kind.value,
        original_filename=Path(payload.original_filename).name[:512],
        content_type=payload.content_type,
        total_bytes=payload.total_bytes,
        part_size=part_size,
        expires_at=datetime.now(UTC) + timedelta(hours=settings.upload_session_hours),
    )
    db.add(upload)
    db.flush()
    try:
        upload.provider_upload_id = storage_adapter(settings, upload.provider).initiate(upload)
        project.status = project.status.__class__.UPLOADING
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Multipart upload could not be initialized") from exc
    db.refresh(upload)
    return upload


@router.get(
    "/multipart-uploads/{upload_id}",
    response_model=MultipartUploadDetail,
    tags=["uploads"],
)
def get_multipart_upload(
    upload_id: UUID,
    db: Session = Depends(get_db),
) -> MultipartUploadDetail:
    upload = db.get(MultipartUpload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Multipart upload not found")
    parts = list(
        db.scalars(
            select(MultipartUploadPart)
            .where(MultipartUploadPart.upload_id == upload.id)
            .order_by(MultipartUploadPart.part_number)
        ).all()
    )
    return MultipartUploadDetail.model_validate(
        {
            **MultipartUploadRead.model_validate(upload).model_dump(),
            "parts": parts,
        }
    )


@router.post(
    "/multipart-uploads/{upload_id}/parts/{part_number}/target",
    response_model=MultipartPartTargetRead,
    tags=["uploads"],
)
def create_part_target(
    upload_id: UUID,
    part_number: int,
    db: Session = Depends(get_db),
) -> MultipartPartTargetRead:
    upload = db.get(MultipartUpload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Multipart upload not found")
    if upload.status != "uploading" or is_expired(upload):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Multipart upload is not writable")
    try:
        validate_part_number(upload, part_number)
        target = storage_adapter(settings, upload.provider).part_target(upload, part_number)
    except StorageError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return MultipartPartTargetRead(
        upload_id=upload.id,
        part_number=part_number,
        expected_size=expected_part_size(upload, part_number),
        method=target.method,
        url=target.url,
        headers=target.headers,
    )


@router.put(
    "/multipart-uploads/{upload_id}/parts/{part_number}",
    response_model=MultipartPartRegister,
    tags=["uploads"],
)
def upload_multipart_part(
    upload_id: UUID,
    part_number: int,
    payload: bytes = Body(media_type="application/octet-stream"),
    db: Session = Depends(get_db),
) -> MultipartPartRegister:
    upload = db.get(MultipartUpload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Multipart upload not found")
    if upload.status != "uploading" or is_expired(upload):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Multipart upload is not writable")
    expected = expected_part_size(upload, part_number)
    if len(payload) != expected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Part has {len(payload)} bytes; expected {expected}",
        )
    try:
        etag, size_bytes = storage_adapter(settings, upload.provider).store_local_part(
            upload, part_number, payload
        )
    except StorageError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    part = upsert_part(
        db,
        upload,
        part_number=part_number,
        etag=etag,
        size_bytes=size_bytes,
    )
    db.commit()
    return MultipartPartRegister(
        part_number=part.part_number,
        etag=part.etag,
        size_bytes=part.size_bytes,
    )


@router.post(
    "/multipart-uploads/{upload_id}/parts",
    response_model=MultipartPartRegister,
    tags=["uploads"],
)
def register_remote_part(
    upload_id: UUID,
    payload: MultipartPartRegister,
    db: Session = Depends(get_db),
) -> MultipartPartRegister:
    upload = db.get(MultipartUpload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Multipart upload not found")
    expected = expected_part_size(upload, payload.part_number)
    if payload.size_bytes != expected:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Part size is invalid")
    upsert_part(
        db,
        upload,
        part_number=payload.part_number,
        etag=payload.etag,
        size_bytes=payload.size_bytes,
    )
    db.commit()
    return payload


@router.post(
    "/multipart-uploads/{upload_id}/complete",
    response_model=MultipartUploadRead,
    tags=["uploads"],
)
def complete_multipart_upload(
    upload_id: UUID,
    payload: MultipartCompleteRequest,
    db: Session = Depends(get_db),
) -> MultipartUpload:
    upload = db.get(MultipartUpload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Multipart upload not found")
    if upload.status == "completed":
        return upload
    if upload.status != "uploading" or is_expired(upload):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Multipart upload cannot be completed")
    for submitted in payload.parts:
        upsert_part(
            db,
            upload,
            part_number=submitted.part_number,
            etag=submitted.etag,
            size_bytes=submitted.size_bytes,
        )
    parts = list(
        db.scalars(
            select(MultipartUploadPart)
            .where(MultipartUploadPart.upload_id == upload.id)
            .order_by(MultipartUploadPart.part_number)
        ).all()
    )
    expected_count = (upload.total_bytes + upload.part_size - 1) // upload.part_size
    if len(parts) != expected_count or sum(part.size_bytes for part in parts) != upload.total_bytes:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Multipart upload is incomplete")
    try:
        completed = storage_adapter(settings, upload.provider).complete(upload, parts)
        finalize_upload_asset(db, upload, completed)
        db.commit()
    except Exception as exc:
        db.rollback()
        upload = db.get(MultipartUpload, upload_id)
        if upload is not None:
            upload.status = "failed"
            upload.error_message = str(exc)[:2_000]
            db.commit()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Multipart upload could not be finalized") from exc
    db.refresh(upload)
    return upload


@router.delete(
    "/multipart-uploads/{upload_id}",
    response_model=SimpleMessage,
    tags=["uploads"],
)
def abort_multipart_upload(
    upload_id: UUID,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    upload = db.get(MultipartUpload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Multipart upload not found")
    if upload.status in {"completed", "aborted"}:
        return SimpleMessage(message=f"Multipart upload is already {upload.status}.")
    try:
        storage_adapter(settings, upload.provider).abort(upload)
    finally:
        upload.status = "aborted"
        upload.error_message = "Upload was aborted by the user"
        db.commit()
    return SimpleMessage(message="Multipart upload aborted.")


@router.post(
    "/operations/uploads/cleanup",
    response_model=SimpleMessage,
    tags=["operations"],
)
def cleanup_uploads(
    request: Request,
    db: Session = Depends(get_db),
) -> SimpleMessage:
    user_id = _current_user_id(request)
    managed = db.scalar(
        select(func.count(WorkspaceMembership.id)).where(
            WorkspaceMembership.user_id == user_id,
            WorkspaceMembership.role.in_(["owner", "admin"]),
        )
    )
    if not managed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator permission is required")
    expired = list(
        db.scalars(
            select(MultipartUpload).where(
                MultipartUpload.expires_at <= datetime.now(UTC),
                MultipartUpload.status.in_(["uploading", "failed"]),
            )
        ).all()
    )
    for upload in expired:
        cleanup_expired_multipart_upload(db, upload, settings)
    record_audit(
        db,
        action="operations.upload_cleanup",
        resource_type="multipart_upload",
        actor_user_id=user_id,
        payload={"expired_uploads": len(expired)},
    )
    db.commit()
    return SimpleMessage(message=f"Expired {len(expired)} multipart upload(s).")
