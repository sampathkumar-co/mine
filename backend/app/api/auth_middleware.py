from __future__ import annotations

import re
from uuid import UUID

from fastapi import Request, status
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.security import InvalidTokenError, decode_access_token
from app.models.platform import ResumableUpload, User, WorkspaceMembership
from app.models.project import Project

settings = get_settings()
PROJECT_PATH = re.compile(r"^/api/v1/projects/([0-9a-fA-F-]{36})(?:/|$)")
WORKSPACE_PATH = re.compile(r"^/api/v1/workspaces/([0-9a-fA-F-]{36})(?:/|$)")
UPLOAD_PATH = re.compile(r"^/api/v1/uploads/([0-9a-fA-F-]{36})(?:/|$)")
USER_PATH = re.compile(r"^/api/v1/users/([0-9a-fA-F-]{36})(?:/|$)")
PUBLIC_PATHS = {
    "/api/v1/health",
    "/api/v1/auth/register",
    "/api/v1/auth/login",
}


def _error(detail: str, status_code: int) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status_code)


def _membership_exists(db, workspace_id: UUID, user_id: UUID) -> bool:
    return (
        db.scalar(
            select(WorkspaceMembership.id).where(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == user_id,
            )
        )
        is not None
    )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if request.method == "OPTIONS" or path in PUBLIC_PATHS or path.startswith(
            "/api/v1/deliveries/"
        ):
            return await call_next(request)

        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.casefold() != "bearer" or not token:
            if not settings.auth_required:
                return await call_next(request)
            return _error("Authentication required", status.HTTP_401_UNAUTHORIZED)

        try:
            user_id = decode_access_token(token, settings)
        except InvalidTokenError as exc:
            return _error(str(exc), status.HTTP_401_UNAUTHORIZED)

        with SessionLocal() as db:
            user = db.get(User, user_id)
            if user is None:
                return _error("Authentication session is no longer valid", status.HTTP_401_UNAUTHORIZED)
            request.state.user_id = user.id

            project_match = PROJECT_PATH.match(path)
            if project_match:
                try:
                    project_id = UUID(project_match.group(1))
                except ValueError:
                    return _error("Project not found", status.HTTP_404_NOT_FOUND)
                project = db.get(Project, project_id)
                if project is None:
                    return _error("Project not found", status.HTTP_404_NOT_FOUND)
                allowed = project.user_id == user.id
                if project.workspace_id is not None:
                    allowed = _membership_exists(db, project.workspace_id, user.id)
                if not allowed:
                    return _error("Project not found", status.HTTP_404_NOT_FOUND)

            workspace_match = WORKSPACE_PATH.match(path)
            if workspace_match:
                try:
                    workspace_id = UUID(workspace_match.group(1))
                except ValueError:
                    return _error("Workspace not found", status.HTTP_404_NOT_FOUND)
                if not _membership_exists(db, workspace_id, user.id):
                    return _error("Workspace not found", status.HTTP_404_NOT_FOUND)

            upload_match = UPLOAD_PATH.match(path)
            if upload_match:
                try:
                    upload_id = UUID(upload_match.group(1))
                except ValueError:
                    return _error("Upload session not found", status.HTTP_404_NOT_FOUND)
                upload = db.get(ResumableUpload, upload_id)
                if upload is None:
                    return _error("Upload session not found", status.HTTP_404_NOT_FOUND)
                project = db.get(Project, upload.project_id)
                if project is None or project.workspace_id is None:
                    return _error("Upload session not found", status.HTTP_404_NOT_FOUND)
                if not _membership_exists(db, project.workspace_id, user.id):
                    return _error("Upload session not found", status.HTTP_404_NOT_FOUND)

            user_match = USER_PATH.match(path)
            if user_match:
                try:
                    requested_user_id = UUID(user_match.group(1))
                except ValueError:
                    return _error("User not found", status.HTTP_404_NOT_FOUND)
                if requested_user_id != user.id:
                    return _error("User not found", status.HTTP_404_NOT_FOUND)

        return await call_next(request)
