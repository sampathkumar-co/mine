from collections.abc import Generator
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.core.database import Base, SessionLocal, engine
from app.core.enums import ProjectStatus
from app.main import app
from app.models.project import Project


@pytest.fixture(autouse=True)
def reset_runtime() -> Generator[None, None, None]:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    runtime_dir = Path("./.test-data")
    if runtime_dir.exists():
        for path in sorted(runtime_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def register(client: TestClient, email: str) -> tuple[dict[str, str], str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "correct-horse-battery-staple",
            "display_name": email.split("@", 1)[0],
            "workspace_name": "Production Studio",
        },
    )
    assert response.status_code == 201
    payload = response.json()
    return {"Authorization": f"Bearer {payload['access_token']}"}, payload["workspaces"][0]["id"]


def create_project(client: TestClient, headers: dict[str, str], workspace_id: str) -> str:
    response = client.post(
        "/api/v1/projects",
        headers=headers,
        json={
            "workspace_id": workspace_id,
            "contract": {
                "objective": "Create a secure launch video",
                "target_duration_seconds": 30,
            },
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_register_login_and_workspace_library(client: TestClient) -> None:
    headers, workspace_id = register(client, "owner@example.com")
    project_id = create_project(client, headers, workspace_id)

    login = client.post(
        "/api/v1/auth/login",
        json={
            "email": "owner@example.com",
            "password": "correct-horse-battery-staple",
        },
    )
    assert login.status_code == 200
    assert login.json()["workspaces"][0]["id"] == workspace_id

    projects = client.get(f"/api/v1/workspaces/{workspace_id}/projects", headers=headers)
    assert projects.status_code == 200
    assert projects.json()[0]["id"] == project_id
    assert projects.json()[0]["objective"] == "Create a secure launch video"


def test_workspace_membership_blocks_other_accounts(client: TestClient) -> None:
    owner_headers, workspace_id = register(client, "owner@example.com")
    project_id = create_project(client, owner_headers, workspace_id)
    outsider_headers, _ = register(client, "outsider@example.com")

    response = client.get(f"/api/v1/projects/{project_id}", headers=outsider_headers)
    assert response.status_code == 404


def test_resumable_upload_finalizes_into_project_asset(client: TestClient) -> None:
    headers, workspace_id = register(client, "upload@example.com")
    project_id = create_project(client, headers, workspace_id)
    payload = b"first-second"

    created = client.post(
        f"/api/v1/projects/{project_id}/uploads",
        headers=headers,
        json={
            "kind": "source_video",
            "original_filename": "source.mp4",
            "content_type": "video/mp4",
            "total_bytes": len(payload),
        },
    )
    assert created.status_code == 201
    upload_id = created.json()["id"]

    first = client.patch(
        f"/api/v1/uploads/{upload_id}",
        headers={**headers, "Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        content=payload[:5],
    )
    assert first.status_code == 200
    assert first.json()["received_bytes"] == 5
    assert first.json()["status"] == "uploading"

    wrong_offset = client.patch(
        f"/api/v1/uploads/{upload_id}",
        headers={**headers, "Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"},
        content=b"bad",
    )
    assert wrong_offset.status_code == 409
    assert wrong_offset.headers["Upload-Offset"] == "5"

    completed = client.patch(
        f"/api/v1/uploads/{upload_id}",
        headers={**headers, "Upload-Offset": "5", "Content-Type": "application/offset+octet-stream"},
        content=payload[5:],
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["asset_id"] is not None

    project = client.get(f"/api/v1/projects/{project_id}", headers=headers)
    assert project.json()["status"] == "ready_to_queue"
    assert project.json()["assets"][0]["size_bytes"] == len(payload)


def test_signed_delivery_link_hides_storage_path(client: TestClient) -> None:
    headers, workspace_id = register(client, "delivery@example.com")
    project_id = create_project(client, headers, workspace_id)
    output = Path("./.test-data/outputs") / project_id / "final.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"publishable-video")

    with SessionLocal() as db:
        project = db.get(Project, UUID(project_id))
        assert project is not None
        project.output_path = str(output)
        project.status = ProjectStatus.READY
        db.commit()

    link = client.post(f"/api/v1/projects/{project_id}/delivery", headers=headers)
    assert link.status_code == 200
    url = link.json()["url"]
    assert str(output) not in url

    delivered = client.get(urlsplit(url).path)
    assert delivered.status_code == 200
    assert delivered.content == b"publishable-video"
    assert delivered.headers["content-disposition"].startswith("inline")
