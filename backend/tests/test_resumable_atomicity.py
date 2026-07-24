from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.database import Base, engine
from app.main import app


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


def test_oversized_chunk_keeps_server_offset_unchanged() -> None:
    with TestClient(app) as client:
        registered = client.post(
            "/api/v1/auth/register",
            json={
                "email": "atomic@example.com",
                "password": "correct-horse-battery-staple",
                "display_name": "Atomic Upload",
                "workspace_name": "Atomic Studio",
            },
        ).json()
        headers = {"Authorization": f"Bearer {registered['access_token']}"}
        workspace_id = registered["workspaces"][0]["id"]
        project_id = client.post(
            "/api/v1/projects",
            headers=headers,
            json={
                "workspace_id": workspace_id,
                "contract": {"objective": "Test atomic resumable uploads"},
            },
        ).json()["id"]
        upload_id = client.post(
            f"/api/v1/projects/{project_id}/uploads",
            headers=headers,
            json={
                "kind": "source_video",
                "original_filename": "large.mp4",
                "content_type": "video/mp4",
                "total_bytes": 70_000,
            },
        ).json()["id"]

        rejected = client.patch(
            f"/api/v1/uploads/{upload_id}",
            headers={
                **headers,
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"x" * 65_537,
        )
        assert rejected.status_code == 413

        status_response = client.get(f"/api/v1/uploads/{upload_id}", headers=headers)
        assert status_response.status_code == 200
        assert status_response.json()["received_bytes"] == 0
        assert status_response.json()["status"] == "uploading"

        accepted = client.patch(
            f"/api/v1/uploads/{upload_id}",
            headers={
                **headers,
                "Upload-Offset": "0",
                "Content-Type": "application/offset+octet-stream",
            },
            content=b"valid",
        )
        assert accepted.status_code == 200
        assert accepted.json()["received_bytes"] == 5
