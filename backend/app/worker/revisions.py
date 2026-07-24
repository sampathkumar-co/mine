from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.enums import AssetKind, ProjectStatus
from app.director.revision_engine import (
    RevisionEditDecisionGraph,
    apply_graph_revision,
    normalize_revision_graph,
)
from app.director.revisions import LockedRange
from app.director.style import ProductionStyle
from app.models.analysis import EditDecisionGraphRecord, EditGraphRevision, ProjectAnalysis
from app.models.project import Project, ProjectAsset
from app.quality.editorial import review_and_repair_edit_graph
from app.rendering.captions import write_ass_captions
from app.rendering.ffmpeg import RenderSource, probe_media, validate_vertical_output
from app.rendering.semantic_overlays import render_semantic_production_graph
from app.sensory.models import AnalysisBundle
from app.services.jobs import claim_job, finish_job
from app.worker.celery_app import celery_app

settings = get_settings()


def _all_assets(db: Session, project_id: UUID, kind: AssetKind) -> list[ProjectAsset]:
    return list(
        db.scalars(
            select(ProjectAsset)
            .where(ProjectAsset.project_id == project_id, ProjectAsset.kind == kind)
            .order_by(ProjectAsset.created_at)
        ).all()
    )


def _style_with_overrides(
    analysis: AnalysisBundle,
    graph: RevisionEditDecisionGraph,
) -> ProductionStyle:
    style = ProductionStyle.model_validate(analysis.production_style)
    overrides = graph.render_overrides
    caption_updates: dict[str, object] = {}
    if "caption_all_caps" in overrides:
        caption_updates["all_caps"] = bool(overrides["caption_all_caps"])
    size_delta = int(overrides.get("caption_size_delta", 0) or 0)
    if size_delta:
        caption_updates["font_size"] = min(120, max(36, style.caption.font_size + size_delta))
    if caption_updates:
        style = style.model_copy(
            update={"caption": style.caption.model_copy(update=caption_updates)}
        )
    if "music_enabled" in overrides:
        style = style.model_copy(
            update={
                "music": style.music.model_copy(
                    update={"enabled": bool(overrides["music_enabled"])}
                )
            }
        )
    return style


def _render_sources(
    source_assets: list[ProjectAsset],
    analysis: AnalysisBundle,
) -> list[RenderSource]:
    clips_by_id = {clip.asset_id: clip for clip in analysis.source_clips}
    sources: list[RenderSource] = []
    for asset in source_assets:
        asset_id = str(asset.id)
        clip = clips_by_id.get(asset_id)
        sources.append(
            RenderSource(
                asset_id=asset_id,
                path=asset.storage_path,
                probe=probe_media(asset.storage_path, settings),
                subject_center_x=(
                    clip.subject_framing.normalized_center_x if clip is not None else 0.5
                ),
            )
        )
    return sources


def _activate_revision(
    db: Session,
    project: Project,
    revision: EditGraphRevision,
    graph: RevisionEditDecisionGraph,
) -> None:
    db.execute(
        update(EditGraphRevision)
        .where(EditGraphRevision.project_id == project.id)
        .values(is_active=False)
    )
    revision.is_active = True
    active_graph = db.scalar(
        select(EditDecisionGraphRecord).where(
            EditDecisionGraphRecord.project_id == project.id
        )
    )
    payload = graph.model_dump(mode="json")
    if active_graph is None:
        active_graph = EditDecisionGraphRecord(
            project_id=project.id,
            version=revision.version,
            payload=payload,
        )
        db.add(active_graph)
    else:
        active_graph.version = revision.version
        active_graph.payload = payload
    project.output_path = revision.output_path
    project.status = ProjectStatus.READY
    project.error_message = None


@celery_app.task(bind=True, max_retries=2)
def run_revision_pipeline(
    self,
    project_id: str,
    version: int,
    job_id: str | None = None,
) -> dict[str, str]:
    project_uuid = UUID(project_id)
    job_uuid = UUID(job_id) if job_id else None
    with SessionLocal() as db:
        if job_uuid is not None:
            job = claim_job(
                db,
                job_uuid,
                {"revision"},
                celery_task_id=str(getattr(self.request, "id", "") or ""),
            )
            if job is None:
                db.rollback()
                return {"project_id": project_id, "version": str(version), "status": "duplicate"}
            db.commit()
        project = db.get(Project, project_uuid)
        revision = db.scalar(
            select(EditGraphRevision).where(
                EditGraphRevision.project_id == project_uuid,
                EditGraphRevision.version == version,
            )
        )
        if project is None or revision is None:
            if job_uuid is not None:
                finish_job(db, job_uuid, succeeded=False, error="Project or revision no longer exists")
                db.commit()
            return {"project_id": project_id, "status": "missing"}

        try:
            base_revision = db.scalar(
                select(EditGraphRevision).where(
                    EditGraphRevision.project_id == project_uuid,
                    EditGraphRevision.version == revision.base_version,
                )
            )
            analysis_record = db.scalar(
                select(ProjectAnalysis).where(ProjectAnalysis.project_id == project_uuid)
            )
            if base_revision is None or not base_revision.graph_payload:
                raise ValueError("Base revision graph is unavailable")
            if analysis_record is None:
                raise ValueError("Project analysis is unavailable")

            revision.status = "planning"
            revision.error_message = None
            db.commit()
            self.update_state(
                state="PLANNING_REVISION",
                meta={"project_id": project_id, "version": version},
            )

            base_graph = normalize_revision_graph(base_revision.graph_payload)
            locked = [LockedRange.model_validate(item) for item in revision.locked_ranges]
            application = apply_graph_revision(
                base_graph,
                revision.instruction or "Apply revision",
                next_version=version,
                locked_ranges=locked,
            )
            analysis = AnalysisBundle.model_validate(analysis_record.payload)
            repaired, critic_report = review_and_repair_edit_graph(
                application.graph,
                analysis,
                project.contract,
            )
            graph = normalize_revision_graph(repaired.model_dump(mode="json"))
            revision.graph_payload = graph.model_dump(mode="json")
            revision.render_plan = application.render_plan.model_dump(mode="json")
            revision.critic_report = critic_report.model_dump(mode="json")
            if settings.require_editorial_critic_pass and not critic_report.passed:
                blocking = [
                    issue.message
                    for issue in critic_report.issues
                    if issue.severity == "blocking"
                ]
                raise ValueError(
                    "Editorial critic blocked revision: " + "; ".join(blocking[:5])
                )
            if not graph.segments:
                raise ValueError("Revision removed every renderable segment")

            output_dir = Path(settings.output_dir) / project_id
            revision_dir = output_dir / "revisions"
            cache_dir = output_dir / "cache"
            revision_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            output_path = revision_dir / f"v{version}.mp4"
            caption_path = revision_dir / f"v{version}.ass"

            if application.render_plan.scope == "metadata_only" and base_revision.output_path:
                source_output = Path(base_revision.output_path)
                if source_output.exists():
                    shutil.copy2(source_output, output_path)
                    revision.narration_cache_path = base_revision.narration_cache_path
                else:
                    application.render_plan.scope = "full_master"
                    application.render_plan.notes.append(
                        "Base output was missing, so the worker fell back to a full render."
                    )

            if not output_path.exists():
                source_assets = _all_assets(db, project_uuid, AssetKind.SOURCE_VIDEO)
                if not source_assets:
                    raise ValueError("Project source footage is unavailable")
                render_sources = _render_sources(source_assets, analysis)
                style = _style_with_overrides(analysis, graph)

                transcripts = {
                    clip.asset_id: clip.transcript
                    for clip in analysis.source_clips
                    if clip.transcript is not None
                }
                captions_enabled = bool(
                    graph.render_overrides.get("captions_enabled", settings.enable_captions)
                )
                render_caption_path: Path | None = None
                if captions_enabled and transcripts:
                    count = write_ass_captions(
                        caption_path,
                        graph,
                        transcripts,
                        style=style.caption,
                    )
                    if count:
                        render_caption_path = caption_path

                selected_music_id = str(
                    analysis.production_style.get("selected_music_asset_id") or ""
                )
                music_asset = next(
                    (
                        asset
                        for asset in _all_assets(db, project_uuid, AssetKind.MUSIC)
                        if str(asset.id) == selected_music_id
                    ),
                    None,
                )
                music_path = music_asset.storage_path if music_asset and style.music.enabled else None

                reusable_cache = None
                if (
                    application.render_plan.reuse_narration_master
                    and base_revision.narration_cache_path
                    and Path(base_revision.narration_cache_path).exists()
                ):
                    reusable_cache = base_revision.narration_cache_path
                    revision.narration_cache_path = reusable_cache
                else:
                    narration_cache = cache_dir / f"narration-v{version}.mp4"
                    revision.narration_cache_path = str(narration_cache)

                revision.status = "rendering"
                revision.render_plan = application.render_plan.model_dump(mode="json")
                db.commit()
                self.update_state(
                    state="RENDERING_REVISION",
                    meta={"project_id": project_id, "version": version},
                )
                render_semantic_production_graph(
                    render_sources,
                    output_path,
                    graph,
                    caption_path=render_caption_path,
                    music_path=music_path,
                    style=style,
                    narration_cache_path=(
                        None if reusable_cache else revision.narration_cache_path
                    ),
                    reuse_narration_base_path=reusable_cache,
                    settings=settings,
                )

            revision.status = "quality_check"
            db.commit()
            output_probe = probe_media(output_path, settings)
            validate_vertical_output(output_probe, expect_audio=True)
            if abs(output_probe.duration_seconds - graph.selected_duration_seconds) > max(
                1.0, graph.selected_duration_seconds * 0.08
            ):
                raise ValueError("Revision output duration differs materially from its graph")

            revision.output_path = str(output_path)
            revision.status = "ready"
            revision.render_plan = application.render_plan.model_dump(mode="json")
            _activate_revision(db, project, revision, graph)
            if job_uuid is not None:
                finish_job(db, job_uuid, succeeded=True)
            db.commit()
            return {
                "project_id": project_id,
                "version": str(version),
                "status": "ready",
                "render_scope": application.render_plan.scope,
                "reused_narration": str(
                    application.render_plan.reuse_narration_master
                ).lower(),
            }
        except Exception as exc:
            db.rollback()
            revision = db.scalar(
                select(EditGraphRevision).where(
                    EditGraphRevision.project_id == project_uuid,
                    EditGraphRevision.version == version,
                )
            )
            if revision is not None:
                if self.request.retries < self.max_retries:
                    revision.status = "queued"
                    revision.error_message = f"Retrying revision after error: {str(exc)[:500]}"
                else:
                    revision.status = "failed"
                    revision.error_message = str(exc)[:2_000]
                db.commit()
            if job_uuid is not None:
                if self.request.retries < self.max_retries:
                    from app.models.operations import ProductionJob

                    job = db.get(ProductionJob, job_uuid)
                    if job is not None:
                        job.status = "dispatched"
                        job.last_error = str(exc)[:2_000]
                        db.commit()
                else:
                    finish_job(db, job_uuid, succeeded=False, error=str(exc))
                    db.commit()
            if self.request.retries < self.max_retries:
                raise self.retry(
                    exc=exc,
                    countdown=2 ** (self.request.retries + 1),
                ) from exc
            raise
