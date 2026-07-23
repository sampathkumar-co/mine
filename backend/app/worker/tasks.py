from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.enums import AssetKind, ProjectStatus
from app.director.edit_graph import build_tier1_edit_graph
from app.models.analysis import EditDecisionGraphRecord, ProjectAnalysis
from app.models.project import Project, ProjectAsset
from app.rendering.ffmpeg import (
    extract_transcription_audio,
    probe_media,
    render_edit_decision_graph,
    validate_vertical_output,
)
from app.sensory.models import AnalysisBundle
from app.sensory.scenes import detect_scenes
from app.sensory.transcription import transcribe_audio
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


def _save_analysis(db: Session, project_id: UUID, payload: dict[str, object]) -> None:
    record = db.scalar(select(ProjectAnalysis).where(ProjectAnalysis.project_id == project_id))
    if record is None:
        record = ProjectAnalysis(project_id=project_id, payload=payload)
        db.add(record)
    else:
        record.payload = payload
    db.commit()


def _save_edit_graph(db: Session, project_id: UUID, payload: dict[str, object]) -> None:
    record = db.scalar(
        select(EditDecisionGraphRecord).where(EditDecisionGraphRecord.project_id == project_id)
    )
    if record is None:
        record = EditDecisionGraphRecord(project_id=project_id, version=1, payload=payload)
        db.add(record)
    else:
        record.version += 1
        record.payload = payload
    db.commit()


@celery_app.task(bind=True, max_retries=3)
def run_project_pipeline(self, project_id: str) -> dict[str, str]:
    project_uuid = UUID(project_id)
    with SessionLocal() as db:
        project = db.get(Project, project_uuid)
        if project is None:
            return {"project_id": project_id, "status": "missing"}

        audio_path = Path(settings.output_dir) / project_id / "analysis" / "speech.wav"
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
            media_probe = probe_media(source.storage_path, settings)
            scenes = detect_scenes(
                source.storage_path,
                duration_seconds=media_probe.duration_seconds,
                settings=settings,
            )

            transcript = None
            if media_probe.has_audio and settings.openai_api_key:
                extract_transcription_audio(source.storage_path, audio_path, settings)
                transcript = transcribe_audio(audio_path, settings)
            elif media_probe.has_audio and settings.require_transcription:
                raise ValueError("Transcription is required but no provider credentials are configured")

            analysis = AnalysisBundle(
                media=asdict(media_probe),
                transcript=transcript,
                scenes=scenes,
            )
            _save_analysis(db, project.id, analysis.model_dump(mode="json"))

            _set_status(db, project, ProjectStatus.PLANNING)
            self.update_state(state="PLANNING", meta={"project_id": project_id})
            contract = project.contract
            graph = build_tier1_edit_graph(
                analysis,
                objective=str(contract.get("objective", "Create a clear professional video")),
                target_duration_seconds=float(contract.get("target_duration_seconds", 45)),
            )
            if not graph.segments:
                raise ValueError("Director could not identify a usable edit segment")
            _save_edit_graph(db, project.id, graph.model_dump(mode="json"))

            output_path = Path(settings.output_dir) / project_id / "final.mp4"
            _set_status(db, project, ProjectStatus.RENDERING)
            self.update_state(state="RENDERING", meta={"project_id": project_id})
            render_edit_decision_graph(
                source.storage_path,
                output_path,
                graph,
                has_audio=media_probe.has_audio,
                settings=settings,
            )

            _set_status(db, project, ProjectStatus.QUALITY_CHECK)
            self.update_state(state="QUALITY_CHECK", meta={"project_id": project_id})
            output_probe = probe_media(output_path, settings)
            validate_vertical_output(output_probe)
            expected_duration = graph.selected_duration_seconds
            if abs(output_probe.duration_seconds - expected_duration) > max(1.0, expected_duration * 0.08):
                raise ValueError("Rendered duration differs materially from the Edit Decision Graph")

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
        finally:
            audio_path.unlink(missing_ok=True)
