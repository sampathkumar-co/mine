from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.core.database import Base, SessionLocal, engine
from app.main import app
from app.models.platform import WorkspaceMembership


@pytest.fixture(autouse=True)
def reset_authorization_runtime() -> Generator[None, None, None]:
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


def session(client: TestClient, email: str) -> dict[str, object]:
    registered = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "correct-horse-battery-staple",
            "display_name": email.split("@", 1)[0],
            "workspace_name": f"{email} workspace",
        },
    )
    assert registered.status_code == 201
    assert registered.json()["refresh_token"] is None
    assert client.cookies.get("director_refresh")
    return registered.json()


def auth(value: dict[str, object]) -> dict[str, str]:
    return {"Authorization": f"Bearer {value['access_token']}"}


def test_viewer_cannot_create_project_in_workspace(client: TestClient) -> None:
    owner = session(client, "owner-edge@example.com")
    viewer = session(client, "viewer-edge@example.com")
    workspace_id = UUID(str(owner["workspaces"][0]["id"]))  # type: ignore[index]
    viewer_id = UUID(str(viewer["user"]["id"]))  # type: ignore[index]

    with SessionLocal() as db:
        db.add(
            WorkspaceMembership(
                workspace_id=workspace_id,
                user_id=viewer_id,
                role="viewer",
            )
        )
        db.commit()

    denied = client.post(
        "/api/v1/projects",
        headers=auth(viewer),
        json={
            "workspace_id": str(workspace_id),
            "contract": {
                "objective": "A viewer must not create this production",
                "target_duration_seconds": 30,
            },
        },
    )
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Workspace editor permission is required"

    allowed = client.post(
        "/api/v1/projects",
        headers=auth(owner),
        json={
            "workspace_id": str(workspace_id),
            "contract": {
                "objective": "An owner can create this production",
                "target_duration_seconds": 30,
            },
        },
    )
    assert allowed.status_code == 201


def test_global_cleanup_endpoint_is_not_publicly_callable(client: TestClient) -> None:
    owner = session(client, "cleanup-edge@example.com")
    response = client.post(
        "/api/v1/operations/uploads/cleanup",
        headers=auth(owner),
    )
    assert response.status_code == 404
