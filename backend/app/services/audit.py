from __future__ import annotations

import re
import secrets
from uuid import UUID

from fastapi import Request
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.database import SessionLocal
from app.models.operations import AuditEvent
from app.models.project import Project

PROJECT_PATH = re.compile(r"^/api/v1/projects/([0-9a-fA-F-]{36})(?:/|$)")
WORKSPACE_PATH = re.compile(r"^/api/v1/workspaces/([0-9a-fA-F-]{36})(?:/|$)")


def record_audit(
    db: Session,
    *,
    action: str,
    resource_type: str,
    workspace_id: UUID | None = None,
    actor_user_id: UUID | None = None,
    resource_id: str | None = None,
    request_id: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    payload: dict[str, object] | None = None,
) -> AuditEvent:
    event = AuditEvent(
        workspace_id=workspace_id,
        actor_user_id=actor_user_id,
        action=action[:120],
        resource_type=resource_type[:80],
        resource_id=resource_id[:120] if resource_id else None,
        request_id=request_id[:80] if request_id else None,
        ip_address=ip_address[:64] if ip_address else None,
        user_agent=user_agent[:500] if user_agent else None,
        payload=payload or {},
    )
    db.add(event)
    db.flush()
    return event


def _request_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",", 1)[0].strip() or (request.client.host if request.client else None)


def _resource_context(db: Session, path: str) -> tuple[UUID | None, str, str | None]:
    project_match = PROJECT_PATH.match(path)
    if project_match:
        project_id = UUID(project_match.group(1))
        project = db.get(Project, project_id)
        return project.workspace_id if project else None, "project", str(project_id)
    workspace_match = WORKSPACE_PATH.match(path)
    if workspace_match:
        workspace_id = UUID(workspace_match.group(1))
        return workspace_id, "workspace", str(workspace_id)
    return None, "platform", None


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or secrets.token_hex(12)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            try:
                with SessionLocal() as db:
                    workspace_id, resource_type, resource_id = _resource_context(
                        db, request.url.path
                    )
                    record_audit(
                        db,
                        workspace_id=workspace_id,
                        actor_user_id=getattr(request.state, "user_id", None),
                        action=f"{request.method.lower()}:{request.url.path}",
                        resource_type=resource_type,
                        resource_id=resource_id,
                        request_id=request_id,
                        ip_address=_request_ip(request),
                        user_agent=request.headers.get("user-agent"),
                        payload={"status_code": response.status_code},
                    )
                    db.commit()
            except Exception:
                # Audit persistence must not turn a successful production action into a failure.
                pass
        return response
