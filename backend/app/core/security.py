from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from app.core.config import Settings

PASSWORD_ITERATIONS = 600_000


class InvalidTokenError(ValueError):
    pass


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
    except (TypeError, ValueError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _sign(encoded: str, secret: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()


def sign_payload(payload: dict[str, Any], settings: Settings) -> str:
    value = {**payload, "kid": settings.auth_key_id}
    encoded = _b64encode(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{encoded}.{_b64encode(_sign(encoded, settings.auth_secret))}"


def verify_signed_payload(token: str, settings: Settings) -> dict[str, Any]:
    try:
        encoded, signature_text = token.split(".", 1)
        supplied = _b64decode(signature_text)
        payload = json.loads(_b64decode(encoded))
        if not isinstance(payload, dict):
            raise InvalidTokenError("Invalid token payload")
        key_id = str(payload.get("kid", ""))
        candidates: list[str] = []
        if key_id == settings.auth_key_id:
            candidates.append(settings.auth_secret)
        if (
            settings.auth_previous_secret
            and settings.auth_previous_key_id
            and key_id == settings.auth_previous_key_id
        ):
            candidates.append(settings.auth_previous_secret)
        if not candidates or not any(
            hmac.compare_digest(supplied, _sign(encoded, secret)) for secret in candidates
        ):
            raise InvalidTokenError("Invalid token signature or key identifier")
        expires_at = int(payload.get("exp", 0))
        if expires_at <= int(datetime.now(UTC).timestamp()):
            raise InvalidTokenError("Token has expired")
        return payload
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, InvalidTokenError):
            raise
        raise InvalidTokenError("Malformed token") from exc


def create_access_token(user_id: UUID, settings: Settings, *, session_id: UUID) -> tuple[str, datetime]:
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.access_token_minutes)
    payload: dict[str, Any] = {
        "purpose": "access",
        "sub": str(user_id),
        "sid": str(session_id),
        "exp": int(expires_at.timestamp()),
    }
    return sign_payload(payload, settings), expires_at


def decode_access_session(token: str, settings: Settings) -> tuple[UUID, UUID]:
    payload = verify_signed_payload(token, settings)
    if payload.get("purpose") != "access":
        raise InvalidTokenError("Token is not an access session")
    try:
        user_id = UUID(str(payload["sub"]))
        session_id = UUID(str(payload["sid"]))
    except (KeyError, ValueError) as exc:
        raise InvalidTokenError("Access token has invalid session claims") from exc
    return user_id, session_id


def decode_access_token(token: str, settings: Settings) -> UUID:
    user_id, _ = decode_access_session(token, settings)
    return user_id
