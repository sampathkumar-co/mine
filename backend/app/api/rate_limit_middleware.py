from __future__ import annotations

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.core.config import get_settings
from app.services.rate_limit import FixedWindowRateLimiter

settings = get_settings()
limiter = FixedWindowRateLimiter(settings)
AUTH_PATHS = {
    "/api/v1/auth/register",
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/auth/password-reset/request",
    "/api/v1/auth/password-reset/confirm",
    "/api/v1/auth/email-verification/request",
    "/api/v1/auth/email-verification/confirm",
    "/api/v1/invitations/accept",
}
WEBHOOK_PATHS = {"/api/v1/billing/webhooks/stripe"}
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _client_identifier(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if user_id is not None:
        return f"user:{user_id}"
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return f"ip:{forwarded}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def _policy(request: Request) -> tuple[str, int] | None:
    path = request.url.path
    if request.method == "OPTIONS":
        return None
    if path in AUTH_PATHS:
        return "auth", settings.auth_rate_limit_requests
    if path in WEBHOOK_PATHS:
        return "webhook", settings.webhook_rate_limit_requests
    if request.method in MUTATING_METHODS:
        return "mutation", settings.mutation_rate_limit_requests
    return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not settings.rate_limit_enabled:
            return await call_next(request)
        policy = _policy(request)
        if policy is None:
            return await call_next(request)
        group, limit = policy
        identifier = _client_identifier(request)
        decision = limiter.check(
            f"{group}:{identifier}",
            limit=limit,
            window_seconds=settings.rate_limit_window_seconds,
        )
        headers = {
            "X-RateLimit-Limit": str(decision.limit),
            "X-RateLimit-Remaining": str(decision.remaining),
            "X-RateLimit-Reset": str(decision.reset_after_seconds),
        }
        if not decision.allowed:
            headers["Retry-After"] = str(decision.reset_after_seconds)
            return JSONResponse(
                {"detail": "Too many requests. Retry after the rate-limit window resets."},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers=headers,
            )
        response = await call_next(request)
        response.headers.update(headers)
        return response
