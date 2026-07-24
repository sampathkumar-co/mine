from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def is_expired_at(value: datetime, *, now: datetime | None = None) -> bool:
    return as_utc(value) <= as_utc(now or utc_now())
