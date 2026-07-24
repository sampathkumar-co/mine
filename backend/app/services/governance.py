from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.enums import ProjectStatus
from app.models.analysis import EditDecisionGraphRecord, EditGraphRevision, ProjectAnalysis
from app.models.camera import DirectorCameraAudit, PickupMission
from app.models.governance import PrivacyRequest
from app.models.memory import DirectorMemoryEvidence, ProjectPerformanceSignal
from app.models.operations import (
    AuditEvent,
    BillingAccount,
    BillingEntry,
    ProjectBillingReservation,
    StoredObject,
)
from app.models.platform import User, Workspace, WorkspaceMembership
from app.models.project import Project, ProjectAsset
from app.models.subscriptions import WorkspaceSubscription
from app.services.storage_lifecycle import delete_stored_object

ACTIVE_PROJECT_STATUSES = {
    ProjectStatus.QUEUED,
    ProjectStatus.ANALYZING,
    ProjectStatus.PLANNING,
    ProjectStatus.RENDERING,
    ProjectStatus.QUALITY_CHECK,
}
ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing", "past_due", "paused"}
SENSITIVE_COLUMNS = {
    "password_hash",
    "token_hash",
    "refresh_token_hash",
    "storage_path",
    "output_path",
    "narration_cache_path",
    "result_path",
    "local_cache_path",
    "provider_upload_id",
}


class GovernanceError(ValueError):
    pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _row(value: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    hidden = SENSITIVE_COLUMNS | (exclude or set())
    return {
        column.name: _jsonable(getattr(value, column.name))
        for column in value.__table__.columns
        if column.name not in hidden
    }


def workspace_pending_deletion(db: Session, workspace_id: UUID) -> bool:
    request = db.scalar(
        select(PrivacyRequest).where(
            PrivacyRequest.workspace_id == workspace_id,
            PrivacyRequest.kind == "deletion",
            PrivacyRequest.status.in_(["scheduled", "processing"]),
        )
    )
    return request is not None


def workspace_deletion_blockers(db: Session, workspace_id: UUID) -> list[str]:
    blockers: list[str] = []
    active_project = db.scalar(
        select(Project.id).where(
            Project.workspace_id == workspace_id,
            Project.status.in_(ACTIVE_PROJECT_STATUSES),
        ).limit(1)
    )
    if active_project is not None:
        blockers.append("Wait for active production and revision jobs to finish.")
    subscription = db.get(WorkspaceSubscription, workspace_id)
    if subscription and subscription.status in ACTIVE_SUBSCRIPTION_STATUSES:
        blockers.append("Cancel the active subscription before scheduling workspace deletion.")
    return blockers


def _project_rows(db: Session, workspace_id: UUID) -> tuple[list[Project], list[UUID]]:
    projects = list(
        db.scalars(
            select(Project)
            .where(Project.workspace_id == workspace_id)
            .order_by(Project.created_at)
        ).all()
    )
    return projects, [project.id for project in projects]


def _query_for_projects(db: Session, model: Any, project_ids: list[UUID]) -> list[Any]:
    if not project_ids:
        return []
    return list(db.scalars(select(model).where(model.project_id.in_(project_ids))).all())


def _export_payload(db: Session, workspace_id: UUID) -> dict[str, Any]:
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        raise GovernanceError("Workspace no longer exists")
    memberships = list(
        db.execute(
            select(WorkspaceMembership, User)
            .join(User, User.id == WorkspaceMembership.user_id)
            .where(WorkspaceMembership.workspace_id == workspace_id)
            .order_by(WorkspaceMembership.created_at)
        ).all()
    )
    projects, project_ids = _project_rows(db, workspace_id)
    assets = _query_for_projects(db, ProjectAsset, project_ids)
    analyses = _query_for_projects(db, ProjectAnalysis, project_ids)
    graphs = _query_for_projects(db, EditDecisionGraphRecord, project_ids)
    revisions = _query_for_projects(db, EditGraphRevision, project_ids)
    camera_audits = _query_for_projects(db, DirectorCameraAudit, project_ids)
    pickup_missions = _query_for_projects(db, PickupMission, project_ids)
    memory_evidence = _query_for_projects(db, DirectorMemoryEvidence, project_ids)
    performance = _query_for_projects(db, ProjectPerformanceSignal, project_ids)
    billing_account = db.get(BillingAccount, workspace_id)
    billing_entries = list(
        db.scalars(
            select(BillingEntry)
            .where(BillingEntry.workspace_id == workspace_id)
            .order_by(BillingEntry.created_at)
        ).all()
    )
    reservations = list(
        db.scalars(
            select(ProjectBillingReservation).where(
                ProjectBillingReservation.workspace_id == workspace_id
            )
        ).all()
    )
    audit_events = list(
        db.scalars(
            select(AuditEvent)
            .where(AuditEvent.workspace_id == workspace_id)
            .order_by(AuditEvent.created_at)
        ).all()
    )
    subscription = db.get(WorkspaceSubscription, workspace_id)
    return {
        "exported_at": datetime.now(UTC).isoformat(),
        "scope": "workspace metadata, editorial decisions, usage, and content manifest",
        "workspace": _row(workspace),
        "members": [
            {"membership": _row(membership), "user": _row(user)}
            for membership, user in memberships
        ],
        "projects": [_row(item) for item in projects],
        "assets": [_row(item) for item in assets],
        "project_analyses": [_row(item) for item in analyses],
        "edit_decision_graphs": [_row(item) for item in graphs],
        "edit_graph_revisions": [_row(item) for item in revisions],
        "director_camera_audits": [_row(item) for item in camera_audits],
        "pickup_missions": [_row(item) for item in pickup_missions],
        "director_memory_evidence": [_row(item) for item in memory_evidence],
        "performance_signals": [_row(item) for item in performance],
        "billing_account": _row(billing_account) if billing_account else None,
        "billing_entries": [_row(item) for item in billing_entries],
        "billing_reservations": [_row(item) for item in reservations],
        "subscription": _row(subscription) if subscription else None,
        "audit_events": [_row(item) for item in audit_events],
    }


def generate_workspace_export(
    db: Session,
    privacy_request: PrivacyRequest,
    settings: Settings,
) -> PrivacyRequest:
    if privacy_request.kind != "export":
        raise GovernanceError("Privacy request is not an export")
    privacy_request.status = "processing"
    privacy_request.error_message = None
    db.flush()
    destination_dir = Path(settings.output_dir) / "privacy" / str(privacy_request.workspace_id)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{privacy_request.id}.zip"
    temporary = destination.with_suffix(".zip.tmp")
    try:
        payload = _export_payload(db, privacy_request.workspace_id)
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        readme = (
            "Director OS workspace export\n\n"
            "This archive contains account/workspace metadata, project contracts, editorial decisions, "
            "usage records, and a manifest of uploaded assets. Large raw media binaries are not duplicated "
            "inside this archive; their filenames, sizes, hashes, and project relationships are included.\n"
        )
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("workspace-export.json", encoded)
            archive.writestr("README.txt", readme)
        os.replace(temporary, destination)
        digest = hashlib.sha256()
        with destination.open("rb") as source:
            while chunk := source.read(settings.upload_chunk_bytes):
                digest.update(chunk)
        now = datetime.now(UTC)
        privacy_request.result_path = str(destination)
        privacy_request.result_sha256 = digest.hexdigest()
        privacy_request.result_size_bytes = destination.stat().st_size
        privacy_request.available_until = now + timedelta(hours=settings.data_export_retention_hours)
        privacy_request.completed_at = now
        privacy_request.status = "ready"
        db.flush()
        return privacy_request
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        privacy_request.status = "failed"
        privacy_request.error_message = str(exc)[:2_000]
        db.flush()
        raise


def purge_workspace(db: Session, workspace_id: UUID, settings: Settings) -> dict[str, int]:
    blockers = workspace_deletion_blockers(db, workspace_id)
    if blockers:
        raise GovernanceError(" ".join(blockers))
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        return {"projects": 0, "files": 0, "objects": 0}
    projects, project_ids = _project_rows(db, workspace_id)
    files: set[str] = set()
    objects: list[StoredObject] = []
    if project_ids:
        assets = _query_for_projects(db, ProjectAsset, project_ids)
        for asset in assets:
            files.add(asset.storage_path)
        revisions = _query_for_projects(db, EditGraphRevision, project_ids)
        for revision in revisions:
            if revision.output_path:
                files.add(revision.output_path)
            if revision.narration_cache_path:
                files.add(revision.narration_cache_path)
        for project in projects:
            if project.output_path:
                files.add(project.output_path)
        asset_ids = [asset.id for asset in assets]
        if asset_ids:
            objects = list(
                db.scalars(select(StoredObject).where(StoredObject.asset_id.in_(asset_ids))).all()
            )
    for stored in objects:
        delete_stored_object(settings, stored)
    deleted_files = 0
    for raw_path in files:
        path = Path(raw_path)
        if path.exists():
            path.unlink(missing_ok=True)
            deleted_files += 1
    db.delete(workspace)
    db.flush()
    return {"projects": len(projects), "files": deleted_files, "objects": len(objects)}


def expire_ready_exports(db: Session, settings: Settings) -> int:
    now = datetime.now(UTC)
    requests = list(
        db.scalars(
            select(PrivacyRequest).where(
                PrivacyRequest.kind == "export",
                PrivacyRequest.status == "ready",
                PrivacyRequest.available_until <= now,
            )
        ).all()
    )
    for request in requests:
        if request.result_path:
            Path(request.result_path).unlink(missing_ok=True)
        request.result_path = None
        request.status = "completed"
        request.completed_at = request.completed_at or now
    db.flush()
    return len(requests)
