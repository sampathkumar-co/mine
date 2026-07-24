from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.enums import AssetKind, ProjectStatus
from app.models.camera import DirectorCameraAudit, PickupMission
from app.models.project import Project, ProjectAsset
from app.schemas.projects import (
    CameraOverrideRequest,
    DirectorCameraRead,
    PickupSubmissionAccepted,
    ProjectAccepted,
)
from app.storage.uploads import (
    EmptyUploadError,
    UnsupportedAssetError,
    UploadTooLargeError,
    store_upload,
)
from app.worker.tasks import run_project_pipeline

router = APIRouter(tags=["director-camera"])
settings = get_settings()


def _project(db: Session, project_id: UUID) -> Project:
    project = db.scalar(select(Project).where(Project.id == project_id))
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


def _mission(db: Session, project_id: UUID, mission_id: UUID) -> PickupMission:
    mission = db.scalar(
        select(PickupMission).where(
            PickupMission.id == mission_id,
            PickupMission.project_id == project_id,
        )
    )
    if mission is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pickup mission not found")
    return mission


def _preserve_ready_output(project: Project, mission_id: UUID) -> None:
    if project.status != ProjectStatus.READY or not project.output_path:
        return
    source = Path(project.output_path)
    if not source.exists():
        return
    backup_dir = Path(settings.output_dir) / str(project.id) / "camera-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = source.suffix or ".mp4"
    backup = backup_dir / f"before-{mission_id}-{timestamp}{suffix}"
    shutil.copy2(source, backup)
    project.output_path = str(backup)


def _queue_project(db: Session, project: Project) -> ProjectAccepted:
    previous_status = project.status
    project.status = ProjectStatus.QUEUED
    project.error_message = None
    db.commit()
    try:
        task_result = run_project_pipeline.delay(str(project.id))
    except Exception as exc:
        project.status = previous_status
        project.error_message = "Director Camera queue unavailable"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Director Camera processing queue is unavailable",
        ) from exc
    task_id = str(getattr(task_result, "id", ""))
    if not task_id:
        project.status = previous_status
        project.error_message = "Queue did not return a task identifier"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Director Camera queue did not accept the project",
        )
    project.task_id = task_id
    db.commit()
    return ProjectAccepted(
        project_id=project.id,
        status=project.status,
        task_id=task_id,
        message="Director Camera validation and production resumed.",
    )


@router.get(
    "/projects/{project_id}/director-camera",
    response_model=DirectorCameraRead,
)
def get_director_camera(
    project_id: UUID,
    db: Session = Depends(get_db),
) -> DirectorCameraRead:
    project = _project(db, project_id)
    audit = db.scalar(
        select(DirectorCameraAudit)
        .where(DirectorCameraAudit.project_id == project_id)
        .order_by(DirectorCameraAudit.version.desc())
    )
    if audit is None:
        return DirectorCameraRead(
            project_id=project_id,
            project_status=project.status,
            mode=str(project.contract.get("director_camera_mode", "off")),
        )
    missions = list(
        db.scalars(
            select(PickupMission)
            .where(PickupMission.project_id == project_id)
            .order_by(PickupMission.created_at.desc())
        ).all()
    )
    return DirectorCameraRead(
        project_id=project_id,
        project_status=project.status,
        audit_id=audit.id,
        audit_version=audit.version,
        mode=audit.mode,
        readiness_score=audit.readiness_score,
        threshold=audit.threshold,
        ready=audit.ready,
        report=audit.report,
        missions=missions,
    )


@router.post(
    "/projects/{project_id}/director-camera/missions/{mission_id}/submit",
    response_model=PickupSubmissionAccepted,
    status_code=status.HTTP_201_CREATED,
)
async def submit_pickup(
    project_id: UUID,
    mission_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> PickupSubmissionAccepted:
    project = _project(db, project_id)
    mission = _mission(db, project_id, mission_id)
    if project.status not in {ProjectStatus.NEEDS_PICKUPS, ProjectStatus.READY}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pickup footage can only be submitted for a waiting or completed project",
        )
    if mission.status in {"accepted", "cancelled"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Mission is already {mission.status}",
        )

    try:
        _preserve_ready_output(project, mission.id)
        db.commit()
    except OSError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The current ready output could not be preserved for the improvement pass",
        ) from exc

    try:
        stored = await store_upload(
            file,
            project_id=project.id,
            kind=AssetKind.PICKUP_VIDEO,
            settings=settings,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except EmptyUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except UnsupportedAssetError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upload storage is unavailable",
        ) from exc

    asset = ProjectAsset(
        project_id=project.id,
        kind=AssetKind.PICKUP_VIDEO,
        original_filename=stored.original_filename,
        stored_filename=stored.stored_filename,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_path=stored.storage_path,
    )
    db.add(asset)
    try:
        db.flush()
        mission.submitted_asset_id = asset.id
        mission.accepted_asset_id = None
        mission.status = "submitted"
        mission.validation = {}
        mission.error_message = None
        project.status = ProjectStatus.NEEDS_PICKUPS
        project.error_message = None
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        Path(stored.storage_path).unlink(missing_ok=True)
        raise
    db.refresh(asset)
    return PickupSubmissionAccepted(
        project_id=project.id,
        mission_id=mission.id,
        asset=asset,
        mission_status=mission.status,
    )


@router.post(
    "/projects/{project_id}/director-camera/resume",
    response_model=ProjectAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def resume_director_camera(
    project_id: UUID,
    db: Session = Depends(get_db),
) -> ProjectAccepted:
    project = _project(db, project_id)
    if project.status != ProjectStatus.NEEDS_PICKUPS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project is not waiting for pickup footage",
        )
    submitted = db.scalar(
        select(PickupMission.id).where(
            PickupMission.project_id == project_id,
            PickupMission.status == "submitted",
        )
    )
    override = bool(project.contract.get("_director_camera_override"))
    if submitted is None and not override:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Submit at least one requested pickup or explicitly override the camera gate",
        )
    return _queue_project(db, project)


@router.post(
    "/projects/{project_id}/director-camera/override",
    response_model=ProjectAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def override_director_camera(
    project_id: UUID,
    payload: CameraOverrideRequest,
    db: Session = Depends(get_db),
) -> ProjectAccepted:
    project = _project(db, project_id)
    if project.status != ProjectStatus.NEEDS_PICKUPS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project is not blocked by Director Camera",
        )
    contract = dict(project.contract)
    contract["_director_camera_override"] = {
        "reason": payload.reason,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    project.contract = contract
    db.commit()
    return _queue_project(db, project)
