from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.governance import PrivacyRequest
from app.services.governance import (
    expire_ready_exports,
    generate_workspace_export,
    purge_workspace,
)
from app.worker.celery_app import celery_app

settings = get_settings()


@celery_app.task
def generate_workspace_export_task(request_id: str) -> dict[str, str]:
    with SessionLocal() as db:
        request = db.get(PrivacyRequest, UUID(request_id))
        if request is None:
            return {"status": "missing"}
        if request.status == "ready":
            return {"status": "ready"}
        try:
            generate_workspace_export(db, request, settings)
            db.commit()
            return {"status": request.status, "request_id": str(request.id)}
        except Exception:
            db.commit()
            raise


@celery_app.task
def process_privacy_lifecycle() -> dict[str, int]:
    now = datetime.now(UTC)
    counters = {"deleted_workspaces": 0, "expired_exports": 0, "failed_deletions": 0}
    with SessionLocal() as db:
        counters["expired_exports"] = expire_ready_exports(db, settings)
        db.commit()

    with SessionLocal() as db:
        due = list(
            db.scalars(
                select(PrivacyRequest)
                .where(
                    PrivacyRequest.kind == "deletion",
                    PrivacyRequest.status == "scheduled",
                    PrivacyRequest.execute_after <= now,
                )
                .order_by(PrivacyRequest.execute_after)
                .limit(25)
                .with_for_update(skip_locked=True)
            ).all()
        )
        request_ids = [request.id for request in due]
        for request in due:
            request.status = "processing"
        db.commit()

    for request_id in request_ids:
        with SessionLocal() as db:
            request = db.get(PrivacyRequest, request_id)
            if request is None:
                continue
            try:
                purge_workspace(db, request.workspace_id, settings)
                db.commit()
                counters["deleted_workspaces"] += 1
            except Exception as exc:
                db.rollback()
                request = db.get(PrivacyRequest, request_id)
                if request is not None:
                    request.status = "failed"
                    request.error_message = str(exc)[:2_000]
                    db.commit()
                counters["failed_deletions"] += 1
    return counters
