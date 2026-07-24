from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.database import Base, SessionLocal, engine
from app.main import app
from app.models.operations import (
    AuthSessionRecord,
    BillingAccount,
    EmailOutbox,
    UserStarterCreditGrant,
)


@pytest.fixture(autouse=True)
def reset_operations_runtime() -> Generator[None, None, None]:
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


def register(client: TestClient, email: str, *, workspace: str = "Studio") -> dict[str, object]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "correct-horse-battery-staple",
            "display_name": email.split("@", 1)[0],
            "workspace_name": workspace,
        },
    )
    assert response.status_code == 201
    return response.json()


def refreshable_session(client: TestClient, bootstrap: dict[str, object]) -> dict[str, object]:
    assert bootstrap["refresh_token"] is None
    assert client.cookies.get("director_refresh")
    assert client.cookies.get("director_csrf")
    return bootstrap


def headers(session: dict[str, object]) -> dict[str, str]:
    return {"Authorization": f"Bearer {session['access_token']}"}


def extract_link_token(body: str) -> str:
    match = re.search(r"[?&]token=([^\s]+)", body)
    assert match is not None
    return match.group(1)


def create_project(client: TestClient, session: dict[str, object]) -> str:
    workspace_id = session["workspaces"][0]["id"]  # type: ignore[index]
    response = client.post(
        "/api/v1/projects",
        headers=headers(session),
        json={
            "workspace_id": workspace_id,
            "contract": {
                "objective": "Create a launch video with product proof",
                "target_duration_seconds": 30,
            },
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_refresh_rotation_detects_reuse_and_revokes_family(client: TestClient) -> None:
    session = refreshable_session(client, register(client, "sessions@example.com"))
    old_refresh = client.cookies.get("director_refresh")
    old_csrf = client.cookies.get("director_csrf")
    assert old_refresh and old_csrf

    rotated = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": old_csrf},
    )
    assert rotated.status_code == 200
    rotated_payload = rotated.json()
    assert rotated_payload["refresh_token"] is None
    assert client.cookies.get("director_refresh") != old_refresh

    with TestClient(app) as replay_client:
        replay_client.cookies.set("director_refresh", old_refresh, path="/api/v1/auth")
        replay_client.cookies.set("director_csrf", old_csrf, path="/api/v1/auth")
        reused = replay_client.post(
            "/api/v1/auth/refresh",
            headers={"X-CSRF-Token": old_csrf},
        )
    assert reused.status_code == 401
    blocked = client.get("/api/v1/auth/account", headers=headers(rotated_payload))
    assert blocked.status_code == 401

    with SessionLocal() as db:
        records = list(db.scalars(select(AuthSessionRecord)).all())
        assert len(records) == 2
        assert all(record.revoked_at is not None for record in records)


def test_refresh_requires_csrf_and_register_uses_httponly_cookie(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "cookie-security@example.com",
            "password": "correct-horse-battery-staple",
            "display_name": "Cookie Security",
            "workspace_name": "Cookie Studio",
        },
    )
    assert response.status_code == 201
    cookies = response.headers.get_list("set-cookie")
    refresh_cookie = next(item for item in cookies if item.startswith("director_refresh="))
    assert "HttpOnly" in refresh_cookie
    assert "SameSite=strict" in refresh_cookie
    assert response.json()["refresh_token"] is None
    assert client.post("/api/v1/auth/refresh").status_code == 401


def test_auth_me_never_mints_a_new_access_token(client: TestClient) -> None:
    session = register(client, "me-no-token@example.com")
    response = client.get("/api/v1/auth/me", headers=headers(session))
    assert response.status_code == 200
    assert "access_token" not in response.json()
    assert response.json()["email"] == "me-no-token@example.com"


def test_verification_and_password_reset_are_single_use(client: TestClient) -> None:
    session = refreshable_session(client, register(client, "recovery@example.com"))
    requested = client.post(
        "/api/v1/auth/email-verification/request",
        headers=headers(session),
    )
    assert requested.status_code == 200

    with SessionLocal() as db:
        verification = db.scalar(
            select(EmailOutbox)
            .where(EmailOutbox.recipient == "recovery@example.com")
            .order_by(EmailOutbox.created_at.desc())
        )
        assert verification is not None
        verification_token = extract_link_token(verification.body_text)

    confirmed = client.post(
        "/api/v1/auth/email-verification/confirm",
        json={"token": verification_token},
    )
    assert confirmed.status_code == 200
    repeated = client.post(
        "/api/v1/auth/email-verification/confirm",
        json={"token": verification_token},
    )
    assert repeated.status_code == 422

    account = client.get("/api/v1/auth/account", headers=headers(session))
    assert account.status_code == 200
    assert account.json()["user"]["email_verified"] is True

    reset_requested = client.post(
        "/api/v1/auth/password-reset/request",
        json={"email": "recovery@example.com"},
    )
    assert reset_requested.status_code == 202
    with SessionLocal() as db:
        reset_email = db.scalar(
            select(EmailOutbox)
            .where(
                EmailOutbox.recipient == "recovery@example.com",
                EmailOutbox.subject.contains("Reset"),
            )
            .order_by(EmailOutbox.created_at.desc())
        )
        assert reset_email is not None
        reset_token = extract_link_token(reset_email.body_text)

    changed = client.post(
        "/api/v1/auth/password-reset/confirm",
        json={
            "token": reset_token,
            "new_password": "a-new-correct-horse-battery-staple",
        },
    )
    assert changed.status_code == 200
    assert client.get("/api/v1/auth/account", headers=headers(session)).status_code == 401
    assert client.post(
        "/api/v1/auth/login",
        json={
            "email": "recovery@example.com",
            "password": "a-new-correct-horse-battery-staple",
        },
    ).status_code == 200


def test_workspace_invitation_roles_and_last_owner_guard(client: TestClient) -> None:
    owner = refreshable_session(client, register(client, "owner@example.com", workspace="Team"))
    invitee = refreshable_session(client, register(client, "editor@example.com", workspace="Solo"))
    workspace_id = owner["workspaces"][0]["id"]  # type: ignore[index]

    invited = client.post(
        f"/api/v1/workspaces/{workspace_id}/invitations",
        headers=headers(owner),
        json={"email": "editor@example.com", "role": "editor"},
    )
    assert invited.status_code == 201
    with SessionLocal() as db:
        invite_email = db.scalar(
            select(EmailOutbox)
            .where(EmailOutbox.recipient == "editor@example.com")
            .order_by(EmailOutbox.created_at.desc())
        )
        assert invite_email is not None
        invitation_token = extract_link_token(invite_email.body_text)

    accepted = client.post(
        "/api/v1/invitations/accept",
        headers=headers(invitee),
        json={"token": invitation_token},
    )
    assert accepted.status_code == 200

    members = client.get(
        f"/api/v1/workspaces/{workspace_id}/members",
        headers=headers(owner),
    )
    assert members.status_code == 200
    editor = next(item for item in members.json() if item["email"] == "editor@example.com")
    promoted = client.patch(
        f"/api/v1/workspaces/{workspace_id}/members/{editor['id']}",
        headers=headers(owner),
        json={"role": "admin"},
    )
    assert promoted.status_code == 200
    assert promoted.json()["role"] == "admin"

    owner_member = next(item for item in members.json() if item["email"] == "owner@example.com")
    last_owner = client.patch(
        f"/api/v1/workspaces/{workspace_id}/members/{owner_member['id']}",
        headers=headers(owner),
        json={"role": "admin"},
    )
    assert last_owner.status_code == 409


def test_local_multipart_upload_finalizes_verified_asset(client: TestClient) -> None:
    session = refreshable_session(client, register(client, "uploads@example.com"))
    project_id = create_project(client, session)
    first_part = b"a" * 5_242_880
    final_part = b"end"
    total = len(first_part) + len(final_part)

    created = client.post(
        f"/api/v1/projects/{project_id}/multipart-uploads",
        headers=headers(session),
        json={
            "kind": "source_video",
            "original_filename": "source.mp4",
            "content_type": "video/mp4",
            "total_bytes": total,
            "part_size": len(first_part),
        },
    )
    assert created.status_code == 201
    upload_id = created.json()["id"]

    completed_parts = []
    for part_number, payload in ((1, first_part), (2, final_part)):
        target = client.post(
            f"/api/v1/multipart-uploads/{upload_id}/parts/{part_number}/target",
            headers=headers(session),
        )
        assert target.status_code == 200
        uploaded = client.put(
            target.json()["url"],
            headers={**headers(session), "Content-Type": "application/octet-stream"},
            content=payload,
        )
        assert uploaded.status_code == 200
        completed_parts.append(uploaded.json())

    completed = client.post(
        f"/api/v1/multipart-uploads/{upload_id}/complete",
        headers=headers(session),
        json={"parts": completed_parts},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    repeated = client.post(
        f"/api/v1/multipart-uploads/{upload_id}/complete",
        headers=headers(session),
        json={"parts": completed_parts},
    )
    assert repeated.status_code == 200
    assert repeated.json()["asset_id"] == completed.json()["asset_id"]
    project = client.get(f"/api/v1/projects/{project_id}", headers=headers(session))
    assert project.status_code == 200
    assert project.json()["status"] == "ready_to_queue"
    assert project.json()["assets"][0]["size_bytes"] == total


def test_workspace_owner_cannot_mint_credits(client: TestClient) -> None:
    session = refreshable_session(client, register(client, "billing@example.com"))
    workspace_id = session["workspaces"][0]["id"]  # type: ignore[index]
    initial = client.get(
        f"/api/v1/workspaces/{workspace_id}/billing",
        headers=headers(session),
    )
    assert initial.status_code == 200
    original_balance = initial.json()["balance_credits"]

    response = client.post(
        f"/api/v1/workspaces/{workspace_id}/billing/adjustments",
        headers=headers(session),
        json={
            "amount_credits": "999999.0000",
            "idempotency_key": "customer-mint-attempt",
            "description": "This must not be reachable",
        },
    )
    assert response.status_code == 404
    current = client.get(
        f"/api/v1/workspaces/{workspace_id}/billing",
        headers=headers(session),
    )
    assert current.json()["balance_credits"] == original_balance


def test_starter_credit_is_granted_once_per_user(client: TestClient) -> None:
    session = refreshable_session(client, register(client, "one-trial@example.com"))
    first_workspace_id = session["workspaces"][0]["id"]  # type: ignore[index]
    first = client.get(
        f"/api/v1/workspaces/{first_workspace_id}/billing",
        headers=headers(session),
    )
    assert first.status_code == 200
    assert float(first.json()["balance_credits"]) == 100.0

    with SessionLocal() as db:
        from app.services.email import mark_email_verified

        user_id = UUID(str(session["user"]["id"]))  # type: ignore[index]
        mark_email_verified(db, user_id)
        db.commit()

    second_workspace = client.post(
        "/api/v1/workspaces",
        headers=headers(session),
        json={"name": "Second Workspace"},
    )
    assert second_workspace.status_code == 201
    second_workspace_id = second_workspace.json()["id"]
    second = client.get(
        f"/api/v1/workspaces/{second_workspace_id}/billing",
        headers=headers(session),
    )
    assert second.status_code == 200
    assert float(second.json()["balance_credits"]) == 0.0

    with SessionLocal() as db:
        assert len(list(db.scalars(select(UserStarterCreditGrant)).all())) == 1
        accounts = list(db.scalars(select(BillingAccount)).all())
        assert len(accounts) == 2
