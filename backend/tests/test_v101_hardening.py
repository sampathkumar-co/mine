from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.core.database import Base, SessionLocal, engine
from app.core.enums import ProjectStatus
from app.core.security import InvalidTokenError, decode_access_session, sign_payload
from app.main import app
from app.models.governance import PrivacyRequest, WorkspaceDeletionTombstone
from app.models.operations import EmailOutbox, ProductionJob, WorkspaceInvitation
from app.models.platform import Workspace
from app.models.project import Project
from app.services.email import deliver_outbox_message, queue_email
from app.services.governance import purge_workspace
from app.services.jobs import JobConflictError, enqueue_revision_job
from app.worker import dispatch as dispatch_module


@pytest.fixture(autouse=True)
def reset_runtime() -> Generator[None, None, None]:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    root = Path(".test-data")
    if root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def register(client: TestClient, email: str) -> dict[str, object]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "correct-horse-battery-staple",
            "display_name": "Hardening Test",
            "workspace_name": "Hardening Studio",
        },
    )
    assert response.status_code == 201
    return response.json()


def auth(session: dict[str, object]) -> dict[str, str]:
    return {"Authorization": f"Bearer {session['access_token']}"}


def test_stateless_access_token_is_rejected() -> None:
    settings = Settings(auth_secret="hardening-token-secret-change-me")
    token = sign_payload(
        {
            "purpose": "access",
            "sub": "fe2bc5d2-45d9-42a5-a9bc-f4866cbfb1b2",
            "exp": int(datetime.now(UTC).timestamp()) + 300,
        },
        settings,
    )
    with pytest.raises(InvalidTokenError, match="session claims"):
        decode_access_session(token, settings)


def test_signing_key_rotation_requires_known_key_identifier() -> None:
    old = Settings(auth_secret="old-secret-that-is-long-enough-000001", auth_key_id="old")
    token = sign_payload(
        {
            "purpose": "access",
            "sub": "fe2bc5d2-45d9-42a5-a9bc-f4866cbfb1b2",
            "sid": "abcc5d22-1d80-40c6-9c5c-275b67f63933",
            "exp": int(datetime.now(UTC).timestamp()) + 300,
        },
        old,
    )
    rotated = Settings(
        auth_secret="new-secret-that-is-long-enough-000001",
        auth_key_id="new",
        auth_previous_secret=old.auth_secret,
        auth_previous_key_id="old",
    )
    assert decode_access_session(token, rotated)[0] == UUID(
        "fe2bc5d2-45d9-42a5-a9bc-f4866cbfb1b2"
    )
    unknown = Settings(
        auth_secret=rotated.auth_secret,
        auth_key_id="new",
        auth_previous_secret=old.auth_secret,
        auth_previous_key_id="different-old-key",
    )
    with pytest.raises(InvalidTokenError, match="key identifier"):
        decode_access_session(token, unknown)


def test_only_one_active_revision_job_per_project(client: TestClient) -> None:
    session = register(client, "one-revision@example.com")
    workspace_id = UUID(str(session["workspaces"][0]["id"]))  # type: ignore[index]
    user_id = UUID(str(session["user"]["id"]))  # type: ignore[index]
    with SessionLocal() as db:
        project = Project(
            user_id=user_id,
            workspace_id=workspace_id,
            status=ProjectStatus.READY,
            contract={"objective": "Revision concurrency"},
        )
        db.add(project)
        db.flush()
        enqueue_revision_job(db, project.id, 2, created_by_user_id=user_id)
        db.commit()
        with pytest.raises(JobConflictError):
            enqueue_revision_job(db, project.id, 3, created_by_user_id=user_id)


def test_dispatch_uses_stable_task_id_and_is_idempotent(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = register(client, "dispatch@example.com")
    workspace_id = UUID(str(session["workspaces"][0]["id"]))  # type: ignore[index]
    user_id = UUID(str(session["user"]["id"]))  # type: ignore[index]
    with SessionLocal() as db:
        project = Project(
            user_id=user_id,
            workspace_id=workspace_id,
            status=ProjectStatus.QUEUED,
            contract={"objective": "Durable dispatch"},
        )
        db.add(project)
        db.flush()
        job = ProductionJob(
            project_id=project.id,
            created_by_user_id=user_id,
            kind="production",
            dedupe_key=f"test:{project.id}",
            status="queued",
        )
        db.add(job)
        db.commit()
        job_id = job.id

    calls: list[dict[str, object]] = []

    def fake_apply_async(*, args, queue, task_id):
        calls.append({"args": args, "queue": queue, "task_id": task_id})
        return SimpleNamespace(id=task_id)

    monkeypatch.setattr(dispatch_module.run_project_pipeline, "apply_async", fake_apply_async)
    assert dispatch_module.dispatch_pending_jobs()["dispatched"] == 1
    assert dispatch_module.dispatch_pending_jobs()["dispatched"] == 0
    assert calls == [
        {
            "args": [str(calls[0]["args"][0]), str(job_id)],
            "queue": "director.render",
            "task_id": str(job_id),
        }
    ]
    with SessionLocal() as db:
        persisted = db.get(ProductionJob, job_id)
        assert persisted is not None
        assert persisted.status == "dispatched"
        assert persisted.celery_task_id == str(job_id)


def test_repeated_invite_leaves_one_active_invitation(client: TestClient) -> None:
    session = register(client, "invite-owner@example.com")
    workspace_id = session["workspaces"][0]["id"]  # type: ignore[index]
    for role in ("viewer", "editor"):
        response = client.post(
            f"/api/v1/workspaces/{workspace_id}/invitations",
            headers=auth(session),
            json={"email": "same-invitee@example.com", "role": role},
        )
        assert response.status_code == 201
    with SessionLocal() as db:
        active = list(
            db.scalars(
                select(WorkspaceInvitation).where(
                    WorkspaceInvitation.workspace_id == UUID(str(workspace_id)),
                    WorkspaceInvitation.email == "same-invitee@example.com",
                    WorkspaceInvitation.accepted_at.is_(None),
                    WorkspaceInvitation.revoked_at.is_(None),
                )
            ).all()
        )
        assert len(active) == 1
        assert active[0].role == "editor"


def test_email_body_is_encrypted_at_rest() -> None:
    settings = Settings(
        email_body_encryption_key="MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="
    )
    with SessionLocal() as db:
        message = queue_email(
            db,
            recipient="encrypted@example.com",
            subject="Encrypted",
            body_text="reset-token-secret",
            settings=settings,
        )
        db.commit()
        assert message.body_text.startswith("fernet:")
        assert "reset-token-secret" not in message.body_text


def test_smtp_delivery_redacts_token_bearing_body(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []

    class SMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self):
            pass

        def login(self, username, password):
            pass

        def send_message(self, message):
            sent.append(message.get_content())

    monkeypatch.setattr("smtplib.SMTP", SMTP)
    message = EmailOutbox(
        recipient="person@example.com",
        subject="Reset",
        body_text="https://director.example/reset-password?token=secret-value",
    )
    settings = Settings(
        email_provider="smtp",
        smtp_host="smtp.example.com",
        smtp_username="user",
        smtp_password="password",
        smtp_from_email="director@example.com",
    )
    deliver_outbox_message(message, settings)
    assert sent and "secret-value" in sent[0]
    assert message.body_text == "[redacted after delivery]"


def test_workspace_purge_leaves_independent_tombstone(client: TestClient) -> None:
    session = register(client, "delete-tombstone@example.com")
    workspace_id = UUID(str(session["workspaces"][0]["id"]))  # type: ignore[index]
    user_id = UUID(str(session["user"]["id"]))  # type: ignore[index]
    with SessionLocal() as db:
        workspace = db.get(Workspace, workspace_id)
        assert workspace is not None
        request = PrivacyRequest(
            workspace_id=workspace_id,
            requested_by_user_id=user_id,
            kind="deletion",
            status="processing",
            execute_after=datetime.now(UTC),
            request_metadata={"reason": "Hardening tombstone test"},
        )
        db.add(request)
        db.commit()
        request_id = request.id
        purge_workspace(db, workspace_id, Settings(), privacy_request=request)
        db.commit()
    with SessionLocal() as db:
        assert db.get(Workspace, workspace_id) is None
        tombstone = db.scalar(
            select(WorkspaceDeletionTombstone).where(
                WorkspaceDeletionTombstone.request_id == request_id
            )
        )
        assert tombstone is not None
        assert tombstone.workspace_id == workspace_id
        assert tombstone.summary["reason"] == "Hardening tombstone test"
