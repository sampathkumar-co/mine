from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import create_access_token
from app.models.operations import AuthSessionRecord
from app.models.platform import User


class SessionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IssuedSession:
    access_token: str
    refresh_token: str
    access_expires_at: datetime
    refresh_expires_at: datetime
    record: AuthSessionRecord


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _refresh_secret() -> str:
    return secrets.token_urlsafe(48)


def _request_details(request) -> tuple[str | None, str | None]:
    if request is None:
        return None, None
    user_agent = request.headers.get("user-agent")
    forwarded = request.headers.get("x-forwarded-for", "")
    ip_address = forwarded.split(",", 1)[0].strip() or (
        request.client.host if request.client else None
    )
    return user_agent[:500] if user_agent else None, ip_address[:64] if ip_address else None


def issue_session(
    db: Session,
    user: User,
    settings: Settings,
    *,
    request=None,
    family_id: UUID | None = None,
) -> IssuedSession:
    now = datetime.now(UTC)
    refresh_token = _refresh_secret()
    user_agent, ip_address = _request_details(request)
    record = AuthSessionRecord(
        user_id=user.id,
        family_id=family_id or uuid4(),
        refresh_token_hash=token_hash(refresh_token),
        expires_at=now + timedelta(days=settings.refresh_token_days),
        user_agent=user_agent,
        ip_address=ip_address,
        last_used_at=now,
    )
    db.add(record)
    db.flush()
    access_token, access_expires_at = create_access_token(
        user.id,
        settings,
        session_id=record.id,
    )
    return IssuedSession(
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_at=access_expires_at,
        refresh_expires_at=record.expires_at,
        record=record,
    )


def rotate_session(
    db: Session,
    refresh_token: str,
    settings: Settings,
    *,
    request=None,
) -> IssuedSession:
    now = datetime.now(UTC)
    digest = token_hash(refresh_token)
    record = db.scalar(
        select(AuthSessionRecord).where(AuthSessionRecord.refresh_token_hash == digest)
    )
    if record is None:
        raise SessionError("Refresh session is invalid")
    if record.rotated_at is not None or record.replaced_by_session_id is not None:
        db.execute(
            update(AuthSessionRecord)
            .where(
                AuthSessionRecord.family_id == record.family_id,
                AuthSessionRecord.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        db.flush()
        raise SessionError("Refresh session reuse was detected; the session family was revoked")
    if record.revoked_at is not None:
        raise SessionError("Refresh session has been revoked")
    if record.expires_at <= now:
        record.revoked_at = now
        db.flush()
        raise SessionError("Refresh session has expired")

    user = db.get(User, record.user_id)
    if user is None:
        record.revoked_at = now
        db.flush()
        raise SessionError("Refresh session user no longer exists")

    issued = issue_session(
        db,
        user,
        settings,
        request=request,
        family_id=record.family_id,
    )
    record.rotated_at = now
    record.last_used_at = now
    record.replaced_by_session_id = issued.record.id
    db.flush()
    return issued


def revoke_session(db: Session, session_id: UUID, user_id: UUID) -> bool:
    record = db.get(AuthSessionRecord, session_id)
    if record is None or record.user_id != user_id:
        return False
    if record.revoked_at is None:
        record.revoked_at = datetime.now(UTC)
        db.flush()
    return True


def revoke_all_sessions(db: Session, user_id: UUID, *, except_session_id: UUID | None = None) -> int:
    statement = update(AuthSessionRecord).where(
        AuthSessionRecord.user_id == user_id,
        AuthSessionRecord.revoked_at.is_(None),
    )
    if except_session_id is not None:
        statement = statement.where(AuthSessionRecord.id != except_session_id)
    result = db.execute(statement.values(revoked_at=datetime.now(UTC)))
    db.flush()
    return int(result.rowcount or 0)


def session_is_active(db: Session, session_id: UUID, user_id: UUID) -> bool:
    now = datetime.now(UTC)
    record = db.get(AuthSessionRecord, session_id)
    if record is None or record.user_id != user_id:
        return False
    return record.revoked_at is None and record.expires_at > now
