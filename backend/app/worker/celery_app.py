from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "director_os",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_reject_on_worker_lost=True,
    result_expires=86_400,
)
