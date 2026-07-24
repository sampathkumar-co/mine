from __future__ import annotations

import hashlib
import secrets
import smtplib
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.operations import AccountToken, EmailOutbox, UserEmailStatus
from app.models.platform import User


class AccountTokenError(ValueError):
    pass


def _token_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def issue_account_token(
    db: Session,
    user_id: UUID,
    purpose: str,
    *,
    lifetime_minutes: int,
) -> str:
    now = datetime.now(UTC)
    db.query(AccountToken).filter(
        AccountToken.user_id == user_id,
        AccountToken.purpose == purpose,
        AccountToken.consumed_at.is_(None),
    ).update({"consumed_at": now}, synchronize_session=False)
    raw = secrets.token_urlsafe(40)
    db.add(
        AccountToken(
            user_id=user_id,
            purpose=purpose,
            token_hash=_token_hash(raw),
            expires_at=now + timedelta(minutes=lifetime_minutes),
        )
    )
    db.flush()
    return raw


def consume_account_token(db: Session, raw_token: str, purpose: str) -> User:
    now = datetime.now(UTC)
    token = db.scalar(
        select(AccountToken).where(
            AccountToken.token_hash == _token_hash(raw_token),
            AccountToken.purpose == purpose,
        )
    )
    if token is None or token.consumed_at is not None:
        raise AccountTokenError("Token is invalid or has already been used")
    if token.expires_at <= now:
        token.consumed_at = now
        db.flush()
        raise AccountTokenError("Token has expired")
    user = db.get(User, token.user_id)
    if user is None:
        token.consumed_at = now
        db.flush()
        raise AccountTokenError("Token user no longer exists")
    token.consumed_at = now
    db.flush()
    return user


def ensure_email_status(db: Session, user_id: UUID) -> UserEmailStatus:
    status = db.get(UserEmailStatus, user_id)
    if status is None:
        status = UserEmailStatus(user_id=user_id)
        db.add(status)
        db.flush()
    return status


def mark_email_verified(db: Session, user_id: UUID) -> UserEmailStatus:
    status = ensure_email_status(db, user_id)
    status.verified_at = datetime.now(UTC)
    db.flush()
    return status


def email_is_verified(db: Session, user_id: UUID) -> bool:
    status = db.get(UserEmailStatus, user_id)
    return bool(status and status.verified_at)


def queue_email(db: Session, *, recipient: str, subject: str, body_text: str) -> EmailOutbox:
    message = EmailOutbox(recipient=recipient, subject=subject, body_text=body_text)
    db.add(message)
    db.flush()
    return message


def queue_verification_email(db: Session, user: User, settings: Settings) -> str:
    token = issue_account_token(
        db,
        user.id,
        "verify_email",
        lifetime_minutes=settings.email_verification_minutes,
    )
    status = ensure_email_status(db, user.id)
    status.verification_sent_at = datetime.now(UTC)
    url = f"{settings.public_app_url.rstrip('/')}/verify-email?token={token}"
    queue_email(
        db,
        recipient=user.email,
        subject="Verify your Director OS email",
        body_text=(
            f"Hello {user.display_name},\n\nVerify your Director OS email by opening:\n{url}\n\n"
            "This link expires automatically. If you did not create this account, ignore this message."
        ),
    )
    return token


def queue_password_reset_email(db: Session, user: User, settings: Settings) -> str:
    token = issue_account_token(
        db,
        user.id,
        "reset_password",
        lifetime_minutes=settings.password_reset_minutes,
    )
    url = f"{settings.public_app_url.rstrip('/')}/reset-password?token={token}"
    queue_email(
        db,
        recipient=user.email,
        subject="Reset your Director OS password",
        body_text=(
            f"Hello {user.display_name},\n\nReset your Director OS password by opening:\n{url}\n\n"
            "This link expires automatically. If you did not request it, ignore this message."
        ),
    )
    return token


def deliver_outbox_message(message: EmailOutbox, settings: Settings) -> None:
    if settings.email_provider == "database":
        message.status = "sent"
        message.sent_at = datetime.now(UTC)
        return
    if settings.email_provider != "smtp":
        raise ValueError(f"Unsupported email provider: {settings.email_provider}")
    if not settings.smtp_host or not settings.smtp_from_email:
        raise ValueError("SMTP host and sender address are required")

    email = EmailMessage()
    email["From"] = settings.smtp_from_email
    email["To"] = message.recipient
    email["Subject"] = message.subject
    email.set_content(message.body_text)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as client:
        if settings.smtp_starttls:
            client.starttls()
        if settings.smtp_username:
            client.login(settings.smtp_username, settings.smtp_password or "")
        client.send_message(email)
    message.status = "sent"
    message.sent_at = datetime.now(UTC)
