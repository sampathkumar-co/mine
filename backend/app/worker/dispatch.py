from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.operations import ProductionJob
from app.worker.celery_app import celery_app
from app.worker.revisions import run_revision_pipeline
from app.worker.tasks import run_project_pipeline

settings = get_settings()


@celery_app.task
def dispatch_pending_jobs(limit: int = 25) -> dict[str, int]:
    now = datetime.now(UTC)
    stale_dispatch = now - timedelta(minutes=5)
    stale_running = now - timedelta(seconds=settings.render_timeout_seconds + 300)
    with SessionLocal() as db:
        db.execute(
            update(ProductionJob)
            .where(
                ProductionJob.status == "dispatching",
                ProductionJob.dispatching_at < stale_dispatch,
            )
            .values(status="queued", dispatching_at=None)
        )
        # Never automatically duplicate a possibly-live FFmpeg process. A stale
        # running job is fenced as stalled and requires an explicit operator retry.
        db.execute(
            update(ProductionJob)
            .where(
                ProductionJob.status == "running",
                ProductionJob.heartbeat_at < stale_running,
            )
            .values(
                status="stalled",
                last_error="Worker heartbeat expired; operator verification is required before retry",
            )
        )
        jobs = list(
            db.scalars(
                select(ProductionJob)
                .where(
                    ProductionJob.status == "queued",
                    ProductionJob.available_at <= now,
                )
                .order_by(ProductionJob.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            ).all()
        )
        ids = [job.id for job in jobs]
        for job in jobs:
            job.status = "dispatching"
            job.dispatching_at = now
        db.commit()

    dispatched = 0
    failed = 0
    for job_id in ids:
        with SessionLocal() as db:
            job = db.get(ProductionJob, job_id)
            if job is None or job.status != "dispatching":
                continue
            task_id = str(job.id)
            job.status = "dispatched"
            job.celery_task_id = task_id
            job.dispatching_at = None
            db.commit()
            try:
                if job.kind == "revision":
                    run_revision_pipeline.apply_async(
                        args=[str(job.project_id), int(job.revision_version or 0), str(job.id)],
                        queue="director.revisions",
                        task_id=task_id,
                    )
                else:
                    run_project_pipeline.apply_async(
                        args=[str(job.project_id), str(job.id)],
                        queue="director.render",
                        task_id=task_id,
                    )
                dispatched += 1
            except Exception as exc:
                db.execute(
                    update(ProductionJob)
                    .where(
                        ProductionJob.id == job.id,
                        ProductionJob.status == "dispatched",
                    )
                    .values(
                        status="queued",
                        available_at=datetime.now(UTC) + timedelta(seconds=30),
                        last_error=str(exc)[:2_000],
                    )
                )
                db.commit()
                failed += 1
    return {"dispatched": dispatched, "failed": failed}
