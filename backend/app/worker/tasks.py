from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.enums import AssetKind, ProjectStatus
from app.models.project import Project, ProjectAsset
from app.rendering.ffmpeg import probe_media, render_vertical_baseline, validate_vertical_output
from app.worker.celery_app import celery_app

settings = get_settings()


def _set_status(
    db: Session,
    project: Project,
    status: ProjectStatus,
    *,
    error_message: str | None = None,
) -> None:
    project.status = status
    project.error_message = error_message
    db.commit()


@celery_app.task(bind=True, max_retries=3)
def run_project_pipeline(self, project_id: str) -> dict[str, str]:
    project_uuid = UUID(project_id)
    with SessionLocal() as db:
        project = db.get(Project, project_uuid)
        if project is None:
            return {"project_id": project_id, "status": "missing"}

        try:
            source = db.scalar(
                select(ProjectAsset)
                .where(
                    ProjectAsset.project_id == project.id,
                    ProjectAsset.kind == AssetKind.SOURCE_VIDEO,
                )
                .order_by(ProjectAsset.created_at)
            )
            if source is None:
                raise ValueError("Project has no source video")

            _set_status(db, project, ProjectStatus.ANALYZING)
            self.update_state(state="ANALYZING", meta={"project_id": project_id})
            probe_media(source.storage_path, settings)

            _set_status(db, project, ProjectStatus.PLANNING)
            self.update_state(state="PLANNING", meta={"project_id": project_id})

            output_path = Path(settings.output_dir) / project_id / "final.mp4"
            _set_status(db, project, ProjectStatus.RENDERING)
            self.update_state(state="RENDERING", meta={"project_id": project_id})
            render_vertical_baseline(source.storage_path, output_path, settings)

            _set_status(db, project, ProjectStatus.QUALITY_CHECK)
            self.update_state(state="QUALITY_CHECK", meta={"project_id": project_id})
            output_probe = probe_media(output_path, settings)
            validate_vertical_output(output_probe)

            project.output_path = str(output_path)
            _set_status(db, project, ProjectStatus.READY)
            return {"project_id": project_id, "status": ProjectStatus.READY.value}
        except Exception as exc:
            db.rollback()
            project = db.get(Project, project_uuid)
            if project is not None:
                if self.request.retries < self.max_retries:
                    _set_status(
                        db,
                        project,
                        ProjectStatus.QUEUED,
                        error_message=f"Retrying after processing error: {str(exc)[:500]}",
                    )
                else:
                    _set_status(
                        db,
                        project,
                        ProjectStatus.FAILED,
                        error_message=str(exc)[:2_000],
                    )
            if self.request.retries < self.max_retries:
                raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 1)) from exc
            raise
