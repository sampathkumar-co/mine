from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.config import get_settings
from app.services.observability import VERSION, metrics, readiness_snapshot

router = APIRouter(tags=["system"])
settings = get_settings()


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive", "version": VERSION}


@router.get("/health/ready")
def readiness() -> JSONResponse:
    overall, components = readiness_snapshot(settings)
    if settings.environment.casefold() == "production":
        components = {
            name: {"status": value["status"], "detail": None}
            for name, value in components.items()
        }
    payload = {"status": overall, "version": VERSION, "components": components}
    code = status.HTTP_200_OK if overall in {"ready", "degraded"} else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(payload, status_code=code)


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics(request: Request) -> PlainTextResponse:
    if not settings.metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if settings.metrics_token:
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.casefold() != "bearer" or not secrets.compare_digest(token, settings.metrics_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Metrics token required")
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")
