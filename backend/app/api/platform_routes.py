from __future__ import annotations

import hashlib
import os
import re
import subprocess
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.enums import AssetKind, ProjectStatus
from app.core.security import (
    InvalidTokenError,
    create_access_token,
    hash_password,
    sign_payload,
    verify_password,
    verify_signed_payload,
)
from app.models.analysis import EditGraphRevision
from app.models.camera import PickupMission
from app.models.platform import ResumableUpload, User, Workspace, WorkspaceMembership
from app.models.project import Project, ProjectAsset
from app.schemas.platform import (
    AuthSession,
    DeliveryLinkRead,
    LoginRequest,
    RegisterRequest,
    ResumableUploadCreate,
    ResumableUploadRead,
    WorkspaceCreate,
    WorkspaceProjectRead,
    WorkspaceRead,
)
from app.storage.uploads import UnsupportedAssetError, _validate_content_type

router = APIRouter()
settings = get_settings()
EDITABLE_STATUSES = {
    ProjectStatus.CREATED,
    ProjectStatus.UPLOADING,
    ProjectStatus.READY_TO_QUEUE,
    ProjectStatus.FAILED,
}


def _current_user_id(request: Request) -> UUID:
    user_id = getattr(request.state, "user_id", None)
    if not isinstance(user_id, UUID):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user_id


def _workspace_read(membership: WorkspaceMembership) -> WorkspaceRead:
    return WorkspaceRead(
        id=membership.workspace.id,
        name=membership.workspace.name,
        slug=membership.workspace.slug,
        role=membership.role,
        created_at=membership.workspace.created_at,
    )


def _memberships(db: Session, user_id: UUID) -> list[WorkspaceMembership]:
    return list(
        db.scalars(
            select(WorkspaceMembership)
            .where(WorkspaceMembership.user_id == user_id)
            .order_by(WorkspaceMembership.created_at)
        ).all()
    )


def _slug(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:80] or "workspace"
    return f"{base}-{uuid4().hex[:8]}"


def _auth_session(db: Session, user: User) -> AuthSession:
    token, expires_at = create_access_token(user.id, settings)
    return AuthSession(
        access_token=token,
        expires_at=expires_at,
        user=user,
        workspaces=[_workspace_read(item) for item in _memberships(db, user.id)],
    )


@router.post(
    "/auth/register",
    response_model=AuthSession,
    status_code=status.HTTP_201_CREATED,
    tags=["authentication"],
)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> AuthSession:
    email = payload.email.strip().casefold()
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered")

    user = User(
        email=email,
        display_name=payload.display_name.strip(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.flush()
    workspace = Workspace(
        name=payload.workspace_name.strip(),
        slug=_slug(payload.workspace_name),
        created_by_user_id=user.id,
    )
    db.add(workspace)
    db.flush()
    db.add(
        WorkspaceMembership(
            workspace_id=workspace.id,
            user_id=user.id,
            role="owner",
        )
    )
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account could not be created with those details",
        ) from exc
    db.refresh(user)
    return _auth_session(db, user)


@router.post("/auth/login", response_model=AuthSession, tags=["authentication"])
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthSession:
    user = db.scalar(select(User).where(User.email == payload.email.strip().casefold()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password is incorrect",
        )
    return _auth_session(db, user)


@router.get("/auth/me", response_model=AuthSession, tags=["authentication"])
def me(request: Request, db: Session = Depends(get_db)) -> AuthSession:
    user = db.get(User, _current_user_id(request))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return _auth_session(db, user)


@router.get("/workspaces", response_model=list[WorkspaceRead], tags=["workspaces"])
def list_workspaces(request: Request, db: Session = Depends(get_db)) -> list[WorkspaceRead]:
    return [_workspace_read(item) for item in _memberships(db, _current_user_id(request))]


@router.post(
    "/workspaces",
    response_model=WorkspaceRead,
    status_code=status.HTTP_201_CREATED,
    tags=["workspaces"],
)
def create_workspace(
    payload: WorkspaceCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> WorkspaceRead:
    user_id = _current_user_id(request)
    workspace = Workspace(
        name=payload.name.strip(),
        slug=_slug(payload.name),
        created_by_user_id=user_id,
    )
    db.add(workspace)
    db.flush()
    membership = WorkspaceMembership(
        workspace_id=workspace.id,
        user_id=user_id,
        role="owner",
    )
    db.add(membership)
    db.commit()
    db.refresh(membership)
    return _workspace_read(membership)


@router.get(
    "/workspaces/{workspace_id}/projects",
    response_model=list[WorkspaceProjectRead],
    tags=["workspaces"],
)
def list_workspace_projects(
    workspace_id: UUID,
    db: Session = Depends(get_db),
) -> list[WorkspaceProjectRead]:
    projects = list(
        db.scalars(
            select(Project)
            .where(Project.workspace_id == workspace_id)
            .order_by(Project.updated_at.desc())
        ).all()
    )
    return [
        WorkspaceProjectRead(
            id=project.id,
            workspace_id=workspace_id,
            status=project.status,
            objective=str(project.contract.get("objective", "Untitled production")),
            target_platform=str(project.contract.get("target_platform", "unknown")),
            target_duration_seconds=int(project.contract.get("target_duration_seconds", 0)),
            output_available=project.output_available,
            asset_count=len(project.assets),
            created_at=project.created_at,
            updated_at=project.updated_at,
        )
        for project in projects
    ]


@router.post(
    "/projects/{project_id}/uploads",
    response_model=ResumableUploadRead,
    status_code=status.HTTP_201_CREATED,
    tags=["uploads"],
)
def create_resumable_upload(
    project_id: UUID,
    payload: ResumableUploadCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> ResumableUpload:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    if project.status not in EDITABLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Assets cannot be changed while project is {project.status.value}",
        )
    if payload.total_bytes > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload exceeds {settings.max_upload_bytes} bytes",
        )
    try:
        _validate_content_type(payload.kind, payload.content_type)
    except UnsupportedAssetError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    original_filename = Path(payload.original_filename).name[:512]
    upload_dir = Path(settings.upload_dir) / str(project.id) / "resumable"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload = ResumableUpload(
        project_id=project.id,
        created_by_user_id=_current_user_id(request),
        kind=payload.kind.value,
        original_filename=original_filename,
        content_type=payload.content_type,
        total_bytes=payload.total_bytes,
        storage_path=str(upload_dir / f"{uuid4().hex}.part"),
    )
    partial = Path(upload.storage_path)
    try:
        partial.touch(exist_ok=False)
        project.status = ProjectStatus.UPLOADING
        project.error_message = None
        db.add(upload)
        db.commit()
    except (OSError, SQLAlchemyError) as exc:
        db.rollback()
        partial.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upload session could not be created",
        ) from exc
    db.refresh(upload)
    return upload


@router.get("/uploads/{upload_id}", response_model=ResumableUploadRead, tags=["uploads"])
def get_resumable_upload(upload_id: UUID, db: Session = Depends(get_db)) -> ResumableUpload:
    upload = db.get(ResumableUpload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload session not found")
    return upload


def _complete_upload(db: Session, upload: ResumableUpload) -> None:
    project = db.get(Project, upload.project_id)
    if project is None:
        raise ValueError("Upload project no longer exists")
    temporary = Path(upload.storage_path)
    suffix = Path(upload.original_filename).suffix.lower()
    if len(suffix) > 12 or not suffix.replace(".", "").isalnum():
        suffix = ""
    stored_filename = f"{uuid4().hex}{suffix}"
    destination = Path(settings.upload_dir) / str(project.id) / stored_filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    with temporary.open("rb") as source:
        while chunk := source.read(settings.upload_chunk_bytes):
            digest.update(chunk)
    os.replace(temporary, destination)

    asset = ProjectAsset(
        project_id=project.id,
        kind=AssetKind(upload.kind),
        original_filename=upload.original_filename,
        stored_filename=stored_filename,
        content_type=upload.content_type,
        size_bytes=upload.total_bytes,
        sha256=digest.hexdigest(),
        storage_path=str(destination),
    )
    db.add(asset)
    db.flush()
    upload.asset_id = asset.id
    upload.storage_path = str(destination)
    upload.status = "completed"
    project.status = ProjectStatus.READY_TO_QUEUE
    project.error_message = None


@router.patch("/uploads/{upload_id}", response_model=ResumableUploadRead, tags=["uploads"])
async def append_resumable_upload(
    upload_id: UUID,
    request: Request,
    upload_offset: int = Header(alias="Upload-Offset", ge=0),
    db: Session = Depends(get_db),
) -> ResumableUpload:
    upload = db.get(ResumableUpload, upload_id)
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload session not found")
    if upload.status == "completed":
        return upload
    if upload.status != "uploading":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Upload is not writable")
    if upload_offset != upload.received_bytes:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Upload offset does not match the server",
            headers={"Upload-Offset": str(upload.received_bytes)},
        )

    payload = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(payload) + len(chunk) > settings.resumable_request_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Upload chunk exceeds the configured request limit",
            )
        if upload.received_bytes + len(payload) + len(chunk) > upload.total_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Upload exceeds its declared total size",
            )
        payload.extend(chunk)

    storage_path = Path(upload.storage_path)
    original_size = storage_path.stat().st_size
    try:
        with storage_path.open("ab") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        with suppress(OSError):
            with storage_path.open("r+b") as output:
                output.truncate(original_size)
        upload.status = "failed"
        upload.error_message = "Upload storage is unavailable"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upload storage is unavailable",
        ) from exc

    upload.received_bytes += len(payload)
    if upload.received_bytes == upload.total_bytes:
        try:
            _complete_upload(db, upload)
        except OSError as exc:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Upload could not be finalized",
            ) from exc
    db.commit()
    db.refresh(upload)
    return upload


def _delivery_path(db: Session, project_id: UUID, version: int | None) -> tuple[Path, int | None]:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    output_path = project.output_path
    resolved_version = version
    if version is not None:
        revision = db.scalar(
            select(EditGraphRevision).where(
                EditGraphRevision.project_id == project_id,
                EditGraphRevision.version == version,
                EditGraphRevision.status == "ready",
            )
        )
        if revision is None or not revision.output_path:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ready revision not found")
        output_path = revision.output_path
    if not output_path:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Project has no ready output")
    path = Path(output_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Output is no longer available")
    return path, resolved_version


@router.post(
    "/projects/{project_id}/delivery",
    response_model=DeliveryLinkRead,
    tags=["delivery"],
)
def create_delivery_link(
    project_id: UUID,
    version: int | None = Query(default=None, ge=1),
    download: bool = False,
    db: Session = Depends(get_db),
) -> DeliveryLinkRead:
    _delivery_path(db, project_id, version)
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.delivery_link_minutes)
    token = sign_payload(
        {
            "purpose": "delivery",
            "pid": str(project_id),
            "version": version,
            "download": download,
            "exp": int(expires_at.timestamp()),
        },
        settings,
    )
    return DeliveryLinkRead(
        project_id=project_id,
        revision_version=version,
        url=f"{settings.api_v1_prefix}/deliveries/{token}",
        expires_at=expires_at,
        download=download,
    )


@router.get("/deliveries/{token}", tags=["delivery"])
def deliver_output(token: str, db: Session = Depends(get_db)) -> FileResponse:
    try:
        payload = verify_signed_payload(token, settings)
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    if payload.get("purpose") != "delivery":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid delivery token")
    try:
        project_id = UUID(str(payload["pid"]))
        version_value = payload.get("version")
        version = int(version_value) if version_value is not None else None
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid delivery token") from exc
    path, resolved_version = _delivery_path(db, project_id, version)
    filename = f"director-os-{project_id}"
    if resolved_version is not None:
        filename += f"-v{resolved_version}"
    filename += path.suffix or ".mp4"
    disposition = "attachment" if bool(payload.get("download")) else "inline"
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        content_disposition_type=disposition,
        headers={"Cache-Control": "private, max-age=60"},
    )


@router.get(
    "/projects/{project_id}/director-camera/missions/{mission_id}/ghost-frame",
    tags=["director-camera"],
)
def get_mission_ghost_frame(
    project_id: UUID,
    mission_id: UUID,
    db: Session = Depends(get_db),
) -> FileResponse:
    mission = db.scalar(
        select(PickupMission).where(
            PickupMission.id == mission_id,
            PickupMission.project_id == project_id,
        )
    )
    if mission is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pickup mission not found")
    asset = None
    candidate_id = mission.accepted_asset_id or mission.submitted_asset_id
    if candidate_id is not None:
        asset = db.get(ProjectAsset, candidate_id)
    if asset is None:
        asset = db.scalar(
            select(ProjectAsset)
            .where(
                ProjectAsset.project_id == project_id,
                ProjectAsset.kind.in_([AssetKind.SOURCE_VIDEO, AssetKind.PICKUP_VIDEO]),
            )
            .order_by(ProjectAsset.created_at)
        )
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No continuity source exists")

    frame_dir = Path(settings.output_dir) / str(project_id) / "camera" / "ghost-frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / f"{mission_id}-{asset.id}.jpg"
    if not frame_path.exists():
        command = [
            settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0.5",
            "-i",
            asset.storage_path,
            "-frames:v",
            "1",
            "-vf",
            "scale=720:-2",
            "-q:v",
            "3",
            "-y",
            str(frame_path),
        ]
        try:
            subprocess.run(command, check=True, timeout=45, capture_output=True)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            frame_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Continuity ghost frame could not be generated",
            ) from exc
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=300"},
    )
