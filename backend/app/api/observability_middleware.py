from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.services.observability import metrics, normalize_path

logger = logging.getLogger("director.request")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        status_code = 500
        metrics.start()
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            metrics.exception()
            logger.exception(
                "request_failed",
                extra={
                    "request_id": getattr(request.state, "request_id", None),
                    "method": request.method,
                    "path": normalize_path(request.url.path),
                    "status_code": status_code,
                    "user_id": getattr(request.state, "user_id", None),
                    "workspace_id": getattr(request.state, "workspace_id", None),
                },
            )
            raise
        finally:
            duration = time.perf_counter() - started
            metrics.finish(request.method, request.url.path, status_code, duration)
            logger.info(
                "request_completed",
                extra={
                    "request_id": getattr(request.state, "request_id", None),
                    "method": request.method,
                    "path": normalize_path(request.url.path),
                    "status_code": status_code,
                    "duration_ms": round(duration * 1_000, 2),
                    "user_id": getattr(request.state, "user_id", None),
                    "workspace_id": getattr(request.state, "workspace_id", None),
                },
            )
