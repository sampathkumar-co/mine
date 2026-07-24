from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.enums import AssetKind, ProjectStatus
from app.director.revision_engine import compare_revision_graphs, normalize_revision_graph
from app.models.analysis import EditDecisionGraphRecord, EditGraphRevision, ProjectAnalysis
from app.models.project import Project, ProjectAsset
from app.schemas.projects import (
    ProjectAccepted,
    ProjectAssetRead,
    ProjectCreate,
    ProjectIntelligenceRead,
    ProjectRead,
    RevisionAccepted,
    RevisionActivated,
    RevisionComparison,
    RevisionCreate,
    RevisionDetail,
    RevisionSummary,
)
from app.storage.uploads import (
    EmptyUploadError,
    UnsupportedAssetError,
    UploadTooLargeError,
    store_upload,
)
from app.worker.tasks import run_project_pipeline, run_revision_pipeline

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


def _get_revision(db: Session, project_id: UUID, version: int) -> EditGraphRevision:
    revision = db.scalar(
        select(EditGraphRevision).where(
            EditGraphRevision.project_id == project_id,
            EditGraphRevision.version == version,
        )
    )
    if revision is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Revision not found")
    return revision


def _seed_initial_revision(
    db: Session,
    project: Project,
    graph: EditDecisionGraphRecord,
) -> EditGraphRevision:
    existing = db.scalar(
        select(EditGraphRevision).where(
            EditGraphRevision.project_id == project.id,
            EditGraphRevision.version == graph.version,
        )
    )
    if existing is not None:
        return existing

    cache_path = (
        Path(settings.output_dir)
        / str(project.id)
        / "cache"
        / f"narration-v{graph.version}.mp4"
    )
    revision = EditGraphRevision(
        project_id=project.id,
        version=graph.version,
        base_version=None,
        instruction="Initial autonomous production",
        status="ready",
        graph_payload=graph.payload,
        render_plan={"scope": "full_master", "changed_components": ["initial_render"]},
        output_path=project.output_path,
        narration_cache_path=str(cache_path) if cache_path.exists() else None,
        is_active=True,
    )
    db.add(revision)
    db.commit()
    db.refresh(revision)
    return revision


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


@router.get(
    "/projects/{project_id}/intelligence",
    response_model=ProjectIntelligenceRead,
    tags=["projects"],
)
def get_project_intelligence(
    project_id: UUID,
    db: Session = Depends(get_db),
) -> ProjectIntelligenceRead:
    _get_project(db, project_id)
    analysis = db.scalar(select(ProjectAnalysis).where(ProjectAnalysis.project_id == project_id))
    graph = db.scalar(
        select(EditDecisionGraphRecord).where(EditDecisionGraphRecord.project_id == project_id)
    )
    return ProjectIntelligenceRead(
        project_id=project_id,
        analysis=analysis.payload if analysis else None,
        edit_decision_graph=graph.payload if graph else None,
        graph_version=graph.version if graph else None,
    )


@router.get(
    "/projects/{project_id}/revisions",
    response_model=list[RevisionSummary],
    tags=["revisions"],
)
def list_project_revisions(
    project_id: UUID,
    db: Session = Depends(get_db),
) -> list[EditGraphRevision]:
    project = _get_project(db, project_id)
    active_graph = db.scalar(
        select(EditDecisionGraphRecord).where(EditDecisionGraphRecord.project_id == project_id)
    )
    if active_graph is not None:
        _seed_initial_revision(db, project, active_graph)
    return list(
        db.scalars(
            select(EditGraphRevision)
            .where(EditGraphRevision.project_id == project_id)
            .order_by(EditGraphRevision.version.desc())
        ).all()
    )


@router.get(
    "/projects/{project_id}/revisions/{version}",
    response_model=RevisionDetail,
    tags=["revisions"],
)
def get_project_revision(
    project_id: UUID,
    version: int,
    db: Session = Depends(get_db),
) -> EditGraphRevision:
    _get_project(db, project_id)
    return _get_revision(db, project_id, version)


@router.get(
    "/projects/{project_id}/revisions/{left_version}/compare/{right_version}",
    response_model=RevisionComparison,
    tags=["revisions"],
)
def compare_project_revisions(
    project_id: UUID,
    left_version: int,
    right_version: int,
    db: Session = Depends(get_db),
) -> RevisionComparison:
    _get_project(db, project_id)
    left = _get_revision(db, project_id, left_version)
    right = _get_revision(db, project_id, right_version)
    render_plan = compare_revision_graphs(
        normalize_revision_graph(left.graph_payload),
        normalize_revision_graph(right.graph_payload),
    )
    return RevisionComparison(
        project_id=project_id,
        left_version=left_version,
        right_version=right_version,
        render_plan=render_plan.model_dump(mode="json"),
    )


@router.post(
    "/projects/{project_id}/revisions",
    response_model=RevisionAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["revisions"],
)
def create_project_revision(
    project_id: UUID,
    payload: RevisionCreate,
    db: Session = Depends(get_db),
) -> RevisionAccepted:
    project = _get_project(db, project_id)
    if project.status != ProjectStatus.READY or not project.output_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project must have a ready output before it can be revised",
        )

    graph = db.scalar(
        select(EditDecisionGraphRecord).where(EditDecisionGraphRecord.project_id == project_id)
    )
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Project has no active Edit Decision Graph",
        )
    _seed_initial_revision(db, project, graph)

    base_version = payload.base_version or graph.version
    base_revision = _get_revision(db, project_id, base_version)
    if base_revision.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Base revision is not ready",
        )

    highest_version = db.scalar(
        select(func.max(EditGraphRevision.version)).where(
            EditGraphRevision.project_id == project_id
        )
    ) or graph.version
    next_version = highest_version + 1
    revision = EditGraphRevision(
        project_id=project_id,
        version=next_version,
        base_version=base_version,
        instruction=payload.instruction,
        status="queued",
        graph_payload={},
        render_plan={},
        locked_ranges=[item.model_dump(mode="json") for item in payload.locked_ranges],
        is_active=False,
    )
    db.add(revision)
    db.commit()
    db.refresh(revision)

    try:
        task_result = run_revision_pipeline.delay(str(project_id), next_version)
    except Exception as exc:
        revision.status = "failed"
        revision.error_message = "Revision queue unavailable"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Revision queue is unavailable",
        ) from exc

    task_id = str(getattr(task_result, "id", ""))
    if not task_id:
        revision.status = "failed"
        revision.error_message = "Queue did not return a task identifier"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Revision queue did not accept the request",
        )
    revision.task_id = task_id
    db.commit()
    return RevisionAccepted(
        project_id=project_id,
        version=next_version,
        base_version=base_version,
        status=revision.status,
        task_id=task_id,
    )


@router.post(
    "/projects/{project_id}/revisions/{version}/activate",
    response_model=RevisionActivated,
    tags=["revisions"],
)
def activate_project_revision(
    project_id: UUID,
    version: int,
    db: Session = Depends(get_db),
) -> RevisionActivated:
    project = _get_project(db, project_id)
    revision = _get_revision(db, project_id, version)
    if revision.status != "ready" or not revision.output_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only a ready revision with an output can be activated",
        )

    db.execute(
        update(EditGraphRevision)
        .where(EditGraphRevision.project_id == project_id)
        .values(is_active=False)
    )
    revision.is_active = True
    active_graph = db.scalar(
        select(EditDecisionGraphRecord).where(EditDecisionGraphRecord.project_id == project_id)
    )
    if active_graph is None:
        active_graph = EditDecisionGraphRecord(
            project_id=project_id,
            version=version,
            payload=revision.graph_payload,
        )
        db.add(active_graph)
    else:
        active_graph.version = version
        active_graph.payload = revision.graph_payload
    project.output_path = revision.output_path
    project.status = ProjectStatus.READY
    project.error_message = None
    db.commit()
    return RevisionActivated(project_id=project_id, version=version)


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
