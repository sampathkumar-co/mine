from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.enums import AssetKind, ProjectStatus
from app.models.project import Project, ProjectAsset
from app.schemas.projects import ProjectAccepted, ProjectAssetRead, ProjectCreate, ProjectRead
from app.storage.uploads import (
    EmptyUploadError,
    UnsupportedAssetError,
    UploadTooLargeError,
    store_upload,
)
from app.worker.tasks import run_project_pipeline

router = APIRouter()
settings = get_settings()
EDITABLE_STATUSES = {
    ProjectStatus.CREATED,
    ProjectStatus.UPLOADING,
    ProjectStatus.READY_TO_QUEUE,
    ProjectStatus.FAILED,
}


def _get_project(db: Session, project_id: UUID) -> Project:
    project = db.scalar(select(Project).where(Project.id == project_id))
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "director-os-api"}


@router.post(
    "/projects",
    response_model=ProjectRead,
    status_code=status.HTTP_201_CREATED,
    tags=["projects"],
)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    project = Project(
        user_id=payload.user_id,
        status=ProjectStatus.CREATED,
        contract=payload.contract.model_dump(mode="json"),
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects/{project_id}", response_model=ProjectRead, tags=["projects"])
def get_project(project_id: UUID, db: Session = Depends(get_db)) -> Project:
    return _get_project(db, project_id)


@router.post(
    "/projects/{project_id}/assets",
    response_model=ProjectAssetRead,
    status_code=status.HTTP_201_CREATED,
    tags=["projects"],
)
async def upload_project_asset(
    project_id: UUID,
    kind: AssetKind = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ProjectAsset:
    project = _get_project(db, project_id)
    if project.status not in EDITABLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Assets cannot be changed while project is {project.status.value}",
        )

    previous_status = project.status
    project.status = ProjectStatus.UPLOADING
    project.error_message = None
    db.commit()

    try:
        stored = await store_upload(file, project_id=project.id, kind=kind, settings=settings)
    except UploadTooLargeError as exc:
        project.status = previous_status
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except EmptyUploadError as exc:
        project.status = previous_status
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except UnsupportedAssetError as exc:
        project.status = previous_status
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except OSError as exc:
        project.status = previous_status
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upload storage is unavailable",
        ) from exc

    asset = ProjectAsset(
        project_id=project.id,
        kind=kind,
        original_filename=stored.original_filename,
        stored_filename=stored.stored_filename,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_path=stored.storage_path,
    )
    project.status = ProjectStatus.READY_TO_QUEUE
    db.add(asset)
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        Path(stored.storage_path).unlink(missing_ok=True)
        raise
    db.refresh(asset)
    return asset


@router.post(
    "/projects/{project_id}/start",
    response_model=ProjectAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["projects"],
)
def start_project(project_id: UUID, db: Session = Depends(get_db)) -> ProjectAccepted:
    project = _get_project(db, project_id)
    if project.status != ProjectStatus.READY_TO_QUEUE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project must contain uploaded footage before it can start",
        )
    if not any(asset.kind == AssetKind.SOURCE_VIDEO for asset in project.assets):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one source_video asset is required",
        )

    project.status = ProjectStatus.QUEUED
    project.error_message = None
    db.commit()

    try:
        task_result = run_project_pipeline.delay(str(project.id))
    except Exception as exc:
        project.status = ProjectStatus.READY_TO_QUEUE
        project.error_message = "Queue unavailable"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing queue is unavailable",
        ) from exc

    task_id = str(getattr(task_result, "id", ""))
    if not task_id:
        project.status = ProjectStatus.READY_TO_QUEUE
        project.error_message = "Queue did not return a task identifier"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing queue did not accept the project",
        )
    project.task_id = task_id
    db.commit()
    return ProjectAccepted(project_id=project.id, status=project.status, task_id=task_id)
