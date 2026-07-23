from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.enums import AssetKind, ProjectStatus
from app.director.cleanup import refine_graph_with_word_timings
from app.director.edit_graph import build_multiclip_edit_graph
from app.director.semantic_overlays import enhance_graph_with_semantic_overlays
from app.director.style import compile_production_style
from app.models.analysis import EditDecisionGraphRecord, ProjectAnalysis
from app.models.project import Project, ProjectAsset
from app.quality.editorial import review_and_repair_edit_graph
from app.rendering.captions import write_ass_captions
from app.rendering.ffmpeg import (
    MediaProbe,
    RenderSource,
    extract_transcription_audio,
    probe_media,
    validate_vertical_output,
)
from app.rendering.semantic_overlays import render_semantic_production_graph
from app.sensory.framing import detect_primary_subject
from app.sensory.models import AnalysisBundle, ClipAnalysis, MusicProfile, SubjectFraming
from app.sensory.multiclip import (
    analyze_source_clip,
    compute_perceptual_hash,
    mark_duplicate_clips,
)
from app.sensory.music import analyze_music, choose_music
from app.sensory.reference import analyze_reference_style
from app.sensory.scenes import detect_scenes
from app.sensory.semantics import analyze_visual_semantics, extract_text_semantic_tags
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


def _first_asset(db: Session, project_id: UUID, kind: AssetKind) -> ProjectAsset | None:
    return db.scalar(
        select(ProjectAsset)
        .where(ProjectAsset.project_id == project_id, ProjectAsset.kind == kind)
        .order_by(ProjectAsset.created_at)
    )


def _all_assets(db: Session, project_id: UUID, kind: AssetKind) -> list[ProjectAsset]:
    return list(
        db.scalars(
            select(ProjectAsset)
            .where(ProjectAsset.project_id == project_id, ProjectAsset.kind == kind)
            .order_by(ProjectAsset.created_at)
        ).all()
    )


def _analyze_source_assets(
    source_assets: list[ProjectAsset],
    *,
    project_output_dir: Path,
    analysis_notes: list[str],
) -> tuple[list[ClipAnalysis], dict[str, MediaProbe], list[Path]]:
    clip_analyses: list[ClipAnalysis] = []
    probes: dict[str, MediaProbe] = {}
    temporary_audio_paths: list[Path] = []

    for asset in source_assets:
        asset_id = str(asset.id)
        media_probe = probe_media(asset.storage_path, settings)
        probes[asset_id] = media_probe
        scenes = detect_scenes(
            asset.storage_path,
            duration_seconds=media_probe.duration_seconds,
            settings=settings,
        )

        subject_framing = SubjectFraming()
        if settings.enable_subject_framing:
            try:
                subject_framing = detect_primary_subject(
                    asset.storage_path,
                    max_samples=settings.subject_frame_samples,
                )
            except Exception as exc:
                analysis_notes.append(
                    f"Subject framing for {asset.original_filename} used centre fallback: {str(exc)[:180]}"
                )

        transcript = None
        if media_probe.has_audio and settings.openai_api_key:
            audio_path = project_output_dir / "analysis" / f"speech-{asset_id}.wav"
            temporary_audio_paths.append(audio_path)
            extract_transcription_audio(asset.storage_path, audio_path, settings)
            transcript = transcribe_audio(audio_path, settings)
        elif media_probe.has_audio and settings.require_transcription:
            raise ValueError(
                f"Transcription is required but no provider credentials are configured for {asset.original_filename}"
            )

        text_tags = extract_text_semantic_tags(asset.original_filename, transcript)
        visual_tags, continuity = analyze_visual_semantics(
            asset.storage_path,
            media=asdict(media_probe),
            subject_framing=subject_framing,
            max_samples=settings.semantic_frame_samples,
        )
        clip_analyses.append(
            analyze_source_clip(
                asset_id=asset_id,
                filename=asset.original_filename,
                sha256=asset.sha256,
                media=asdict(media_probe),
                transcript=transcript,
                scenes=scenes,
                subject_framing=subject_framing,
                perceptual_hash=compute_perceptual_hash(asset.storage_path),
                semantic_tags=[*text_tags, *visual_tags],
                continuity=continuity,
            )
        )

    return (
        mark_duplicate_clips(
            clip_analyses,
            perceptual_distance_threshold=settings.duplicate_hash_distance,
        ),
        probes,
        temporary_audio_paths,
    )


@celery_app.task(bind=True, max_retries=3)
def run_project_pipeline(self, project_id: str) -> dict[str, str]:
    project_uuid = UUID(project_id)
    temporary_audio_paths: list[Path] = []
    with SessionLocal() as db:
        project = db.get(Project, project_uuid)
        if project is None:
            return {"project_id": project_id, "status": "missing"}

        project_output_dir = Path(settings.output_dir) / project_id
        caption_path = project_output_dir / "analysis" / "captions.ass"
        try:
            source_assets = _all_assets(db, project.id, AssetKind.SOURCE_VIDEO)
            if not source_assets:
                raise ValueError("Project has no source video")
            if len(source_assets) > settings.max_source_clips:
                raise ValueError(
                    f"Project has {len(source_assets)} source clips; maximum is {settings.max_source_clips}"
                )
            reference = _first_asset(db, project.id, AssetKind.REFERENCE_VIDEO)
            music_assets = _all_assets(db, project.id, AssetKind.MUSIC)

            _set_status(db, project, ProjectStatus.ANALYZING)
            self.update_state(state="ANALYZING", meta={"project_id": project_id})
            analysis_notes: list[str] = []
            clip_analyses, probes, temporary_audio_paths = _analyze_source_assets(
                source_assets,
                project_output_dir=project_output_dir,
                analysis_notes=analysis_notes,
            )

            reference_style = None
            if reference is not None and settings.enable_reference_style:
                try:
                    reference_probe = probe_media(reference.storage_path, settings)
                    reference_style = analyze_reference_style(
                        reference.storage_path,
                        asset_id=str(reference.id),
                        media_probe=reference_probe,
                        settings=settings,
                    )
                except Exception as exc:
                    analysis_notes.append(f"Reference style analysis skipped: {str(exc)[:240]}")

            contract = project.contract
            objective = str(contract.get("objective", "Create a clear professional video"))
            target_duration = float(contract.get("target_duration_seconds", 45))
            production_style = compile_production_style(
                contract,
                reference_style,
                default_caption_margin=settings.caption_margin_vertical,
                default_caption_max_words=settings.caption_max_words,
                default_music_volume=settings.music_default_volume,
                default_ducking_threshold=settings.music_ducking_threshold,
                default_music_fade_seconds=settings.music_fade_seconds,
            )

            music_profiles: list[MusicProfile] = []
            music_asset_by_id = {str(asset.id): asset for asset in music_assets}
            if settings.enable_music and production_style.music.enabled:
                for asset in music_assets:
                    try:
                        music_profiles.append(
                            analyze_music(
                                asset.storage_path,
                                asset_id=str(asset.id),
                                filename=asset.original_filename,
                                settings=settings,
                            )
                        )
                    except Exception as exc:
                        analysis_notes.append(
                            f"Music asset {asset.original_filename} skipped: {str(exc)[:180]}"
                        )

            selected_music = choose_music(
                music_profiles,
                desired_energy=production_style.music.desired_energy,
            )
            selected_music_asset = (
                music_asset_by_id.get(selected_music.asset_id) if selected_music is not None else None
            )

            style_payload = production_style.model_dump(mode="json")
            style_payload["selected_music_asset_id"] = (
                selected_music.asset_id if selected_music is not None else None
            )
            style_payload["analysis_notes"] = analysis_notes

            first_accepted = next(
                (clip for clip in clip_analyses if clip.role != "rejected"),
                clip_analyses[0],
            )
            analysis = AnalysisBundle(
                media=first_accepted.media,
                transcript=first_accepted.transcript,
                scenes=first_accepted.scenes,
                subject_framing=first_accepted.subject_framing,
                source_clips=clip_analyses,
                reference_style=reference_style,
                music_profiles=music_profiles,
                production_style=style_payload,
            )
            _save_analysis(db, project.id, analysis.model_dump(mode="json"))

            _set_status(db, project, ProjectStatus.PLANNING)
            self.update_state(state="PLANNING", meta={"project_id": project_id})
            graph = build_multiclip_edit_graph(
                analysis,
                objective=objective,
                target_duration_seconds=target_duration,
            )
            transcripts_by_asset = {
                clip.asset_id: clip.transcript
                for clip in clip_analyses
                if clip.transcript is not None
            }
            if settings.enable_word_cleanup:
                graph = refine_graph_with_word_timings(
                    graph,
                    transcripts_by_asset,
                    silence_threshold_seconds=settings.silence_threshold_seconds,
                    speech_padding_seconds=settings.speech_padding_seconds,
                )
            graph = enhance_graph_with_semantic_overlays(
                graph,
                analysis,
                objective=objective,
                target_duration_seconds=target_duration,
                max_overlays=(settings.max_visual_overlays if settings.enable_semantic_overlays else 0),
                minimum_match_score=settings.minimum_overlay_match_score,
            )
            graph, critic_report = review_and_repair_edit_graph(graph, analysis, contract)
            analysis = analysis.model_copy(
                update={"editorial_report": critic_report.model_dump(mode="json")}
            )
            _save_analysis(db, project.id, analysis.model_dump(mode="json"))
            if settings.require_editorial_critic_pass and not critic_report.passed:
                blocking_messages = [
                    issue.message for issue in critic_report.issues if issue.severity == "blocking"
                ]
                raise ValueError(
                    "Editorial critic blocked rendering: " + "; ".join(blocking_messages[:5])
                )
            if not graph.segments:
                raise ValueError("Director could not identify a usable edit segment")

            graph_notes = [*graph.notes]
            if reference_style is not None:
                graph_notes.append(
                    "Applied reference fingerprint: "
                    f"{reference_style.pace} pace, "
                    f"{reference_style.cuts_per_minute:.1f} cuts/minute."
                )
            if selected_music is not None:
                graph_notes.append(
                    f"Selected uploaded music '{selected_music.filename}' "
                    f"for energy match {selected_music.energy:.2f}."
                )
            graph = graph.model_copy(update={"notes": graph_notes})

            caption_count = 0
            render_caption_path: Path | None = None
            if settings.enable_captions and transcripts_by_asset:
                caption_count = write_ass_captions(
                    caption_path,
                    graph,
                    transcripts_by_asset,
                    style=production_style.caption,
                )
                if caption_count:
                    render_caption_path = caption_path
                    graph = graph.model_copy(
                        update={
                            "notes": [
                                *graph.notes,
                                f"Generated {caption_count} multi-source brand-aware caption cue(s).",
                            ]
                        }
                    )
            _save_edit_graph(db, project.id, graph.model_dump(mode="json"))

            clip_by_asset_id = {clip.asset_id: clip for clip in clip_analyses}
            render_sources = [
                RenderSource(
                    asset_id=str(asset.id),
                    path=asset.storage_path,
                    probe=probes[str(asset.id)],
                    subject_center_x=clip_by_asset_id[str(asset.id)].subject_framing.normalized_center_x,
                )
                for asset in source_assets
            ]

            output_path = project_output_dir / "final.mp4"
            _set_status(db, project, ProjectStatus.RENDERING)
            self.update_state(state="RENDERING", meta={"project_id": project_id})
            render_semantic_production_graph(
                render_sources,
                output_path,
                graph,
                caption_path=render_caption_path,
                music_path=(selected_music_asset.storage_path if selected_music_asset else None),
                style=production_style,
                settings=settings,
            )

            _set_status(db, project, ProjectStatus.QUALITY_CHECK)
            self.update_state(state="QUALITY_CHECK", meta={"project_id": project_id})
            output_probe = probe_media(output_path, settings)
            expect_audio = any(
                source.probe.has_audio
                for source in render_sources
                if any(
                    segment.source_asset_id == source.asset_id
                    for segment in graph.segments
                )
            ) or selected_music_asset is not None
            validate_vertical_output(output_probe, expect_audio=expect_audio)
            expected_duration = graph.selected_duration_seconds
            if abs(output_probe.duration_seconds - expected_duration) > max(
                1.0, expected_duration * 0.08
            ):
                raise ValueError("Rendered duration differs materially from the Edit Decision Graph")

            project.output_path = str(output_path)
            _set_status(db, project, ProjectStatus.READY)
            rejected_count = sum(1 for clip in clip_analyses if clip.role == "rejected")
            return {
                "project_id": project_id,
                "status": ProjectStatus.READY.value,
                "source_clips": str(len(source_assets)),
                "rejected_clips": str(rejected_count),
                "semantic_overlays": str(len(graph.overlays)),
                "critic_score": f"{critic_report.score:.3f}",
                "caption_cues": str(caption_count),
                "reference_pace": reference_style.pace if reference_style else "none",
                "music_asset_id": selected_music.asset_id if selected_music else "none",
            }
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
            for audio_path in temporary_audio_paths:
                audio_path.unlink(missing_ok=True)
