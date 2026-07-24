from __future__ import annotations

from uuid import UUID

from celery.signals import task_failure, task_success

from app.core.database import SessionLocal
from app.models.project import Project
from app.services.audit import record_audit
from app.services.billing import release_project_reservation, settle_project_credits

PROJECT_TASK = "app.worker.tasks.run_project_pipeline"


@task_success.connect
def settle_successful_project(sender=None, result=None, **_: object) -> None:
    if getattr(sender, "name", None) != PROJECT_TASK or not isinstance(result, dict):
        return
    if result.get("status") != "ready" or not result.get("project_id"):
        return
    try:
        project_id = UUID(str(result["project_id"]))
    except ValueError:
        return
    with SessionLocal() as db:
        project = db.get(Project, project_id)
        if project is None:
            return
        reservation = settle_project_credits(db, project)
        if reservation is not None:
            record_audit(
                db,
                workspace_id=project.workspace_id,
                action="billing.project_settled",
                resource_type="project",
                resource_id=str(project.id),
                payload={"settled_credits": str(reservation.settled_credits or 0)},
            )
            db.commit()


@task_failure.connect
def release_failed_project(sender=None, args=None, **_: object) -> None:
    if getattr(sender, "name", None) != PROJECT_TASK or not args:
        return
    try:
        project_id = UUID(str(args[0]))
    except (ValueError, TypeError):
        return
    with SessionLocal() as db:
        project = db.get(Project, project_id)
        if project is None:
            return
        reservation = release_project_reservation(
            db,
            project,
            reason="Released after the production worker exhausted its retries",
        )
        if reservation is not None:
            record_audit(
                db,
                workspace_id=project.workspace_id,
                action="billing.project_released",
                resource_type="project",
                resource_id=str(project.id),
                payload={"released_credits": str(reservation.reserved_credits)},
            )
            db.commit()
