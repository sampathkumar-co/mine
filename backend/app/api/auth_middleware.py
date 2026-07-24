from __future__ import annotations

import re
from uuid import UUID

from fastapi import Request, status
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.security import InvalidTokenError, decode_access_session
from app.models.operations import MultipartUpload
from app.models.platform import ResumableUpload, User, WorkspaceMembership
from app.models.project import Project
from app.services.email import email_is_verified
from app.services.entitlements import EntitlementError, enforce_contract_entitlements
from app.services.governance import workspace_pending_deletion
from app.services.permissions import can_edit, can_manage_billing, can_manage_members
from app.services.sessions import session_is_active

settings = get_settings()
PROJECT_PATH = re.compile(r"^/api/v1/projects/([0-9a-fA-F-]{36})(?:/|$)")
WORKSPACE_PATH = re.compile(r"^/api/v1/workspaces/([0-9a-fA-F-]{36})(?:/|$)")
UPLOAD_PATH = re.compile(r"^/api/v1/uploads/([0-9a-fA-F-]{36})(?:/|$)")
MULTIPART_PATH = re.compile(r"^/api/v1/multipart-uploads/([0-9a-fA-F-]{36})(?:/|$)")
USER_PATH = re.compile(r"^/api/v1/users/([0-9a-fA-F-]{36})(?:/|$)")
PUBLIC_PATHS = {
    "/api/v1/health",
    "/api/v1/health/live",
    "/api/v1/health/ready",
    "/api/v1/metrics",
    "/api/v1/auth/register",
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/auth/password-reset/request",
    "/api/v1/auth/password-reset/confirm",
    "/api/v1/auth/email-verification/confirm",
    "/api/v1/billing/webhooks/stripe",
}
UNVERIFIED_PATHS = {
    "/api/v1/auth/account",
    "/api/v1/auth/session",
    "/api/v1/auth/logout",
    "/api/v1/auth/logout-all",
    "/api/v1/auth/email-verification/request",
    "/api/v1/invitations/accept",
}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
INTERNAL_ONLY_PATHS = {"/api/v1/operations/uploads/cleanup"}


def _error(detail: str, status_code: int) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status_code)


def _membership(db, workspace_id: UUID, user_id: UUID) -> WorkspaceMembership | None:
    return db.scalar(
        select(WorkspaceMembership).where(
            WorkspaceMembership.workspace_id == workspace_id,
            WorkspaceMembership.user_id == user_id,
        )
    )


def _deletion_denied(request: Request, db, workspace_id: UUID) -> JSONResponse | None:
    if request.method in SAFE_METHODS:
        return None
    if request.method == "DELETE" and "/privacy/deletion/" in request.url.path:
        return None
    if workspace_pending_deletion(db, workspace_id):
        return _error(
            "Workspace deletion is scheduled. Cancel deletion before making changes.",
            status.HTTP_423_LOCKED,
        )
    return None


def _authorize_workspace_request(request: Request, role: str) -> JSONResponse | None:
    path = request.url.path
    if request.method in SAFE_METHODS:
        return None
    if path.endswith("/delivery"):
        return None
    if "/billing/adjustments" in path and not can_manage_billing(role):
        return _error("Workspace owner permission is required", status.HTTP_403_FORBIDDEN)
    if any(marker in path for marker in ("/members", "/invitations", "/audit-events", "/privacy")):
        if not can_manage_members(role):
            return _error(
                "Workspace administrator permission is required",
                status.HTTP_403_FORBIDDEN,
            )
        return None
    if not can_edit(role):
        return _error("Workspace editor permission is required", status.HTTP_403_FORBIDDEN)
    return None


async def _authorize_project_creation(
    request: Request,
    db,
    user_id: UUID,
) -> JSONResponse | None:
    if request.method != "POST" or request.url.path != "/api/v1/projects":
        return None
    try:
        payload = await request.json()
        workspace_id = UUID(str(payload["workspace_id"]))
    except (KeyError, TypeError, ValueError):
        return None
    membership = _membership(db, workspace_id, user_id)
    if membership is None:
        return _error("Workspace not found", status.HTTP_404_NOT_FOUND)
    request.state.workspace_id = workspace_id
    request.state.workspace_role = membership.role
    deletion_denied = _deletion_denied(request, db, workspace_id)
    if deletion_denied is not None:
        return deletion_denied
    if not can_edit(membership.role):
        return _error("Workspace editor permission is required", status.HTTP_403_FORBIDDEN)
    contract = payload.get("contract")
    if isinstance(contract, dict):
        try:
            enforce_contract_entitlements(db, workspace_id, contract, settings)
        except EntitlementError as exc:
            return _error(str(exc), status.HTTP_402_PAYMENT_REQUIRED)
    return None


class AuthenticationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if (
            request.method == "OPTIONS"
            or path in PUBLIC_PATHS
            or path.startswith("/api/v1/deliveries/")
            or path.startswith("/api/v1/privacy-deliveries/")
        ):
            return await call_next(request)

        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.casefold() != "bearer" or not token:
            if not settings.auth_required:
                return await call_next(request)
            return _error("Authentication required", status.HTTP_401_UNAUTHORIZED)

        try:
            user_id, session_id = decode_access_session(token, settings)
        except InvalidTokenError as exc:
            return _error(str(exc), status.HTTP_401_UNAUTHORIZED)

        with SessionLocal() as db:
            user = db.get(User, user_id)
            if user is None:
                return _error(
                    "Authentication session is no longer valid",
                    status.HTTP_401_UNAUTHORIZED,
                )
            if session_id is not None and not session_is_active(db, session_id, user_id):
                return _error(
                    "Authentication session has been revoked",
                    status.HTTP_401_UNAUTHORIZED,
                )
            request.state.user_id = user.id
            request.state.session_id = session_id
            if (
                settings.require_verified_email
                and path not in UNVERIFIED_PATHS
                and not email_is_verified(db, user.id)
            ):
                return _error(
                    "Email verification is required before this action",
                    status.HTTP_403_FORBIDDEN,
                )
            if path in INTERNAL_ONLY_PATHS:
                return _error("Not found", status.HTTP_404_NOT_FOUND)

            project_create_denied = await _authorize_project_creation(request, db, user.id)
            if project_create_denied is not None:
                return project_create_denied

            project_match = PROJECT_PATH.match(path)
            if project_match:
                try:
                    project_id = UUID(project_match.group(1))
                except ValueError:
                    return _error("Project not found", status.HTTP_404_NOT_FOUND)
                project = db.get(Project, project_id)
                if project is None:
                    return _error("Project not found", status.HTTP_404_NOT_FOUND)
                if project.workspace_id is None:
                    if project.user_id != user.id:
                        return _error("Project not found", status.HTTP_404_NOT_FOUND)
                else:
                    membership = _membership(db, project.workspace_id, user.id)
                    if membership is None:
                        return _error("Project not found", status.HTTP_404_NOT_FOUND)
                    request.state.workspace_id = project.workspace_id
                    request.state.workspace_role = membership.role
                    deletion_denied = _deletion_denied(request, db, project.workspace_id)
                    if deletion_denied is not None:
                        return deletion_denied
                    denied = _authorize_workspace_request(request, membership.role)
                    if denied is not None:
                        return denied

            workspace_match = WORKSPACE_PATH.match(path)
            if workspace_match:
                try:
                    workspace_id = UUID(workspace_match.group(1))
                except ValueError:
                    return _error("Workspace not found", status.HTTP_404_NOT_FOUND)
                membership = _membership(db, workspace_id, user.id)
                if membership is None:
                    return _error("Workspace not found", status.HTTP_404_NOT_FOUND)
                request.state.workspace_id = workspace_id
                request.state.workspace_role = membership.role
                deletion_denied = _deletion_denied(request, db, workspace_id)
                if deletion_denied is not None:
                    return deletion_denied
                denied = _authorize_workspace_request(request, membership.role)
                if denied is not None:
                    return denied

            upload_match = UPLOAD_PATH.match(path)
            if upload_match:
                upload = db.get(ResumableUpload, UUID(upload_match.group(1)))
                project = db.get(Project, upload.project_id) if upload else None
                membership = (
                    _membership(db, project.workspace_id, user.id)
                    if project and project.workspace_id
                    else None
                )
                if upload is None or project is None or membership is None:
                    return _error("Upload session not found", status.HTTP_404_NOT_FOUND)
                deletion_denied = _deletion_denied(request, db, project.workspace_id)
                if deletion_denied is not None:
                    return deletion_denied
                if request.method not in SAFE_METHODS and not can_edit(membership.role):
                    return _error(
                        "Workspace editor permission is required",
                        status.HTTP_403_FORBIDDEN,
                    )

            multipart_match = MULTIPART_PATH.match(path)
            if multipart_match:
                upload = db.get(MultipartUpload, UUID(multipart_match.group(1)))
                project = db.get(Project, upload.project_id) if upload else None
                membership = (
                    _membership(db, project.workspace_id, user.id)
                    if project and project.workspace_id
                    else None
                )
                if upload is None or project is None or membership is None:
                    return _error("Multipart upload not found", status.HTTP_404_NOT_FOUND)
                deletion_denied = _deletion_denied(request, db, project.workspace_id)
                if deletion_denied is not None:
                    return deletion_denied
                if request.method not in SAFE_METHODS and not can_edit(membership.role):
                    return _error(
                        "Workspace editor permission is required",
                        status.HTTP_403_FORBIDDEN,
                    )

            user_match = USER_PATH.match(path)
            if user_match:
                try:
                    requested_user_id = UUID(user_match.group(1))
                except ValueError:
                    return _error("User not found", status.HTTP_404_NOT_FOUND)
                if requested_user_id != user.id:
                    return _error("User not found", status.HTTP_404_NOT_FOUND)

        return await call_next(request)
