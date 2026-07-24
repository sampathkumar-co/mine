from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "director_os",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.worker.tasks",
        "app.worker.revisions",
        "app.worker.maintenance",
        "app.worker.governance",
        "app.worker.dispatch",
    ],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_reject_on_worker_lost=True,
    result_expires=86_400,
    task_routes={
        "app.worker.tasks.run_project_pipeline": {"queue": "director.render"},
        "app.worker.revisions.run_revision_pipeline": {"queue": "director.revisions"},
        "app.worker.dispatch.dispatch_pending_jobs": {"queue": "director.maintenance"},
        "app.worker.maintenance.deliver_pending_email": {"queue": "director.email"},
        "app.worker.maintenance.cleanup_expired_operations": {"queue": "director.maintenance"},
        "app.worker.governance.*": {"queue": "director.governance"},
    },
    beat_schedule={
        "dispatch-pending-production-jobs": {
            "task": "app.worker.dispatch.dispatch_pending_jobs",
            "schedule": 10.0,
        },
        "deliver-pending-email": {
            "task": "app.worker.maintenance.deliver_pending_email",
            "schedule": 60.0,
        },
        "cleanup-expired-operations": {
            "task": "app.worker.maintenance.cleanup_expired_operations",
            "schedule": 3_600.0,
        },
        "process-privacy-lifecycle": {
            "task": "app.worker.governance.process_privacy_lifecycle",
            "schedule": 900.0,
        },
    },
)

# Register billing settlement/release signal handlers in every worker process.
import app.services.billing_signals  # noqa: E402,F401
