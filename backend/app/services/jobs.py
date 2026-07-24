from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import ProjectStatus
from app.models.operations import ProductionJob
from app.models.project import Project

ACTIVE_JOB_STATUSES = {"queued", "dispatching", "dispatched", "running", "stalled"}


class JobConflictError(ValueError):
    pass


def enqueue_project_job(
    db: Session,
    project_id: UUID,
    *,
    kind: str,
    allowed_statuses: set[ProjectStatus],
    created_by_user_id: UUID | None = None,
    revision_version: int | None = None,
) -> ProductionJob:
    generation = db.scalar(
        update(Project)
        .where(Project.id == project_id, Project.status.in_(allowed_statuses))
        .values(
            status=ProjectStatus.QUEUED,
            error_message=None,
            run_generation=Project.run_generation + 1,
        )
        .returning(Project.run_generation)
    )
    if generation is None:
        raise JobConflictError("Project state changed before the job could be queued")
    job = ProductionJob(
        project_id=project_id,
        created_by_user_id=created_by_user_id,
        kind=kind,
        revision_version=revision_version,
        dedupe_key=f"project:{project_id}:{kind}:{int(generation)}",
        status="queued",
    )
    db.add(job)
    db.flush()
    db.execute(update(Project).where(Project.id == project_id).values(task_id=str(job.id)))
    return job


def enqueue_revision_job(
    db: Session,
    project_id: UUID,
    version: int,
    *,
    created_by_user_id: UUID | None = None,
) -> ProductionJob:
    active = db.scalar(
        select(ProductionJob.id).where(
            ProductionJob.project_id == project_id,
            ProductionJob.kind == "revision",
            ProductionJob.status.in_(ACTIVE_JOB_STATUSES),
        )
    )
    if active is not None:
        raise JobConflictError("Another revision is already queued or rendering")
    job = ProductionJob(
        project_id=project_id,
        created_by_user_id=created_by_user_id,
        kind="revision",
        revision_version=version,
        dedupe_key=f"project:{project_id}:revision:{version}",
        status="queued",
    )
    db.add(job)
    try:
        db.flush()
    except IntegrityError as exc:
        raise JobConflictError("Another revision is already queued or rendering") from exc
    return job


def claim_job(
    db: Session,
    job_id: UUID,
    kinds: set[str],
    *,
    celery_task_id: str | None = None,
) -> ProductionJob | None:
    now = datetime.now(UTC)
    allowed = ProductionJob.status == "dispatched"
    if celery_task_id:
        allowed = or_(
            allowed,
            (ProductionJob.status == "running")
            & (ProductionJob.celery_task_id == celery_task_id),
        )
    claimed_id = db.scalar(
        update(ProductionJob)
        .where(
            ProductionJob.id == job_id,
            ProductionJob.kind.in_(kinds),
            allowed,
        )
        .values(
            status="running",
            attempts=ProductionJob.attempts + 1,
            started_at=now,
            heartbeat_at=now,
            last_error=None,
        )
        .returning(ProductionJob.id)
    )
    if claimed_id is None:
        return None
    return db.get(ProductionJob, claimed_id)


def heartbeat_job(db: Session, job_id: UUID) -> None:
    db.execute(
        update(ProductionJob)
        .where(ProductionJob.id == job_id, ProductionJob.status == "running")
        .values(heartbeat_at=datetime.now(UTC))
    )


def finish_job(db: Session, job_id: UUID, *, succeeded: bool, error: str | None = None) -> None:
    db.execute(
        update(ProductionJob)
        .where(ProductionJob.id == job_id)
        .values(
            status="succeeded" if succeeded else "failed",
            completed_at=datetime.now(UTC),
            heartbeat_at=datetime.now(UTC),
            last_error=error[:2_000] if error else None,
        )
    )
