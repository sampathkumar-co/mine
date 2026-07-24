from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select, update

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.operations import (
    AccountToken,
    AuditEvent,
    AuthSessionRecord,
    EmailOutbox,
    MultipartUpload,
    WorkspaceInvitation,
)
from app.models.platform import ResumableUpload
from app.services.email import deliver_outbox_message
from app.services.storage import cleanup_expired_multipart_upload
from app.worker.celery_app import celery_app

settings = get_settings()


@celery_app.task
def deliver_pending_email(batch_size: int = 50) -> dict[str, int]:
    sent = 0
    failed = 0
    with SessionLocal() as db:
        messages = list(
            db.scalars(
                select(EmailOutbox)
                .where(EmailOutbox.status.in_(["queued", "retrying"]))
                .order_by(EmailOutbox.created_at)
                .limit(max(1, min(batch_size, 500)))
                .with_for_update(skip_locked=True)
            ).all()
        )
        for message in messages:
            message.attempts += 1
            try:
                deliver_outbox_message(message, settings)
                message.last_error = None
                sent += 1
            except Exception as exc:
                message.last_error = str(exc)[:2_000]
                message.status = "failed" if message.attempts >= 5 else "retrying"
                failed += 1
        db.commit()
    return {"sent": sent, "failed": failed}


@celery_app.task
def cleanup_expired_operations() -> dict[str, int]:
    now = datetime.now(UTC)
    counters = {
        "multipart_uploads": 0,
        "resumable_uploads": 0,
        "account_tokens": 0,
        "sessions": 0,
        "invitations": 0,
        "audit_events": 0,
        "email_bodies": 0,
    }
    with SessionLocal() as db:
        multipart = list(
            db.scalars(
                select(MultipartUpload).where(
                    MultipartUpload.expires_at <= now,
                    MultipartUpload.status.in_(["uploading", "failed"]),
                )
            ).all()
        )
        for upload in multipart:
            cleanup_expired_multipart_upload(db, upload, settings)
        counters["multipart_uploads"] = len(multipart)

        stale_resumable = list(
            db.scalars(
                select(ResumableUpload).where(
                    ResumableUpload.status.in_(["uploading", "failed"]),
                    ResumableUpload.updated_at <= now - timedelta(hours=settings.upload_session_hours),
                )
            ).all()
        )
        for upload in stale_resumable:
            Path(upload.storage_path).unlink(missing_ok=True)
            upload.status = "expired"
            upload.error_message = "Upload session expired before completion"
        counters["resumable_uploads"] = len(stale_resumable)

        token_result = db.execute(
            delete(AccountToken).where(
                AccountToken.expires_at <= now - timedelta(days=7)
            )
        )
        counters["account_tokens"] = int(token_result.rowcount or 0)

        session_result = db.execute(
            delete(AuthSessionRecord).where(
                AuthSessionRecord.expires_at <= now - timedelta(days=30)
            )
        )
        counters["sessions"] = int(session_result.rowcount or 0)

        email_result = db.execute(
            update(EmailOutbox)
            .where(
                EmailOutbox.status == "sent",
                EmailOutbox.sent_at.is_not(None),
                EmailOutbox.sent_at <= now - timedelta(hours=settings.email_body_retention_hours),
                EmailOutbox.body_text != "[redacted after retention]",
                EmailOutbox.body_text != "[redacted after delivery]",
            )
            .values(body_text="[redacted after retention]")
        )
        counters["email_bodies"] = int(email_result.rowcount or 0)

        invitation_result = db.execute(
            update(WorkspaceInvitation)
            .where(
                WorkspaceInvitation.expires_at <= now,
                WorkspaceInvitation.accepted_at.is_(None),
                WorkspaceInvitation.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        counters["invitations"] = int(invitation_result.rowcount or 0)

        audit_result = db.execute(
            delete(AuditEvent).where(
                AuditEvent.created_at
                <= now - timedelta(days=settings.audit_retention_days)
            )
        )
        counters["audit_events"] = int(audit_result.rowcount or 0)
        db.commit()
    return counters
