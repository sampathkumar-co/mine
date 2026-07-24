from __future__ import annotations

import re
from uuid import UUID

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.operations import BillingAccount
from app.models.project import Project
from app.services.billing import (
    BillingError,
    InsufficientCreditsError,
    ensure_billing_account,
    post_adjustment,
    release_project_reservation,
    reserve_project_credits,
)

settings = get_settings()
START_PATH = re.compile(
    r"^/api/v1/projects/([0-9a-fA-F-]{36})(?:/start|/director-camera/resume)$"
)


def _error(detail: str, status_code: int) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status_code)


class BillingReservationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        match = START_PATH.match(request.url.path)
        if not settings.billing_enabled or request.method != "POST" or match is None:
            return await call_next(request)

        project_id = UUID(match.group(1))
        actor_user_id = getattr(request.state, "user_id", None)
        with SessionLocal() as db:
            project = db.get(Project, project_id)
            if project is None or project.workspace_id is None:
                return await call_next(request)
            if db.get(BillingAccount, project.workspace_id) is None:
                ensure_billing_account(db, project.workspace_id)
                post_adjustment(
                    db,
                    project.workspace_id,
                    settings.starter_credits,
                    idempotency_key=f"workspace:{project.workspace_id}:starter-grant",
                    description="Starter credits granted when billing was first activated",
                    actor_user_id=actor_user_id,
                    kind="grant",
                )
            try:
                reserve_project_credits(db, project, actor_user_id=actor_user_id)
                db.commit()
            except InsufficientCreditsError as exc:
                db.rollback()
                return _error(str(exc), status.HTTP_402_PAYMENT_REQUIRED)
            except BillingError as exc:
                db.rollback()
                return _error(str(exc), status.HTTP_409_CONFLICT)

        try:
            response = await call_next(request)
        except Exception:
            with SessionLocal() as db:
                project = db.get(Project, project_id)
                if project is not None:
                    release_project_reservation(
                        db,
                        project,
                        reason="Released because the production request failed before queue acceptance",
                    )
                    db.commit()
            raise

        if response.status_code >= 400:
            with SessionLocal() as db:
                project = db.get(Project, project_id)
                if project is not None:
                    release_project_reservation(
                        db,
                        project,
                        reason=f"Released because queue request returned HTTP {response.status_code}",
                    )
                    db.commit()
        return response
