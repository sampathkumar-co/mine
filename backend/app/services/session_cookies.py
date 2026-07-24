from __future__ import annotations

import secrets

from fastapi import Request, Response

from app.core.config import Settings
from app.services.sessions import IssuedSession

CSRF_HEADER = "X-CSRF-Token"


def cookie_secure(settings: Settings) -> bool:
    return settings.auth_cookie_secure or settings.environment.casefold() == "production"


def set_session_cookies(response: Response, issued: IssuedSession, settings: Settings) -> None:
    max_age = settings.refresh_token_days * 24 * 60 * 60
    common = {
        "max_age": max_age,
        "secure": cookie_secure(settings),
        "samesite": settings.auth_cookie_samesite,
    }
    response.set_cookie(
        settings.refresh_cookie_name,
        issued.refresh_token,
        httponly=True,
        path=f"{settings.api_v1_prefix}/auth",
        **common,
    )
    # The double-submit CSRF value is deliberately readable at the web-app path.
    # It is not an authentication credential; the matching server-side hash is authoritative.
    response.set_cookie(
        settings.csrf_cookie_name,
        issued.csrf_token,
        httponly=False,
        path="/",
        **common,
    )


def clear_session_cookies(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        settings.refresh_cookie_name,
        path=f"{settings.api_v1_prefix}/auth",
    )
    response.delete_cookie(settings.csrf_cookie_name, path="/")


def session_cookie_values(request: Request, settings: Settings) -> tuple[str, str]:
    refresh_token = request.cookies.get(settings.refresh_cookie_name, "")
    csrf_cookie = request.cookies.get(settings.csrf_cookie_name, "")
    csrf_header = request.headers.get(CSRF_HEADER, "")
    if not refresh_token:
        raise ValueError("Refresh session cookie is missing")
    if not csrf_cookie or not csrf_header or not secrets_compare(csrf_cookie, csrf_header):
        raise ValueError("Refresh session CSRF check failed")
    return refresh_token, csrf_header


def secrets_compare(left: str, right: str) -> bool:
    return secrets.compare_digest(left, right)
