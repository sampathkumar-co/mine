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
    ],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_reject_on_worker_lost=True,
    result_expires=86_400,
    beat_schedule={
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
