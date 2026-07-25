from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import routes
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


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def account(client: TestClient) -> tuple[dict[str, str], str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "director@example.com",
            "password": "correct-horse-battery-staple",
            "display_name": "Test Director",
            "workspace_name": "Studio",
        },
    )
    assert response.status_code == 201
    payload = response.json()
    return {"Authorization": f"Bearer {payload['access_token']}"}, payload["workspaces"][0]["id"]


def project_payload(workspace_id: str) -> dict[str, object]:
    return {
        "workspace_id": workspace_id,
        "contract": {
            "objective": "Create a polished business reel",
            "tier": 1,
            "target_duration_seconds": 45,
            "must_avoid": ["emojis"],
        },
    }


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "director-os-api"}


def test_project_upload_and_queue(
    client: TestClient,
    account: tuple[dict[str, str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers, workspace_id = account
    created = client.post(
        "/api/v1/projects",
        json=project_payload(workspace_id),
        headers=headers,
    )
    assert created.status_code == 201
    project_id = created.json()["id"]
    assert created.json()["workspace_id"] == workspace_id
    assert created.json()["status"] == "created"
    assert created.json()["contract"]["music_direction_mode"] == "balanced"
    assert created.json()["contract"]["dialogue_protection"] == "automatic"

    intelligence = client.get(
        f"/api/v1/projects/{project_id}/intelligence",
        headers=headers,
    )
    assert intelligence.status_code == 200
    assert intelligence.json()["analysis"] is None
    assert intelligence.json()["edit_decision_graph"] is None

    uploaded = client.post(
        f"/api/v1/projects/{project_id}/assets",
        data={"kind": "source_video"},
        files={"file": ("raw.mp4", b"video-bytes", "video/mp4")},
        headers=headers,
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["kind"] == "source_video"
    assert uploaded.json()["size_bytes"] == 11

    monkeypatch.setattr(
        routes.run_project_pipeline,
        "delay",
        lambda project_id: SimpleNamespace(id=f"task-{project_id}"),
    )
    queued = client.post(f"/api/v1/projects/{project_id}/start", headers=headers)
    assert queued.status_code == 202
    assert queued.json()["status"] == "queued"
    assert queued.json()["task_id"] == f"task-{project_id}"

    fetched = client.get(f"/api/v1/projects/{project_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "queued"
    assert fetched.json()["output_available"] is False
    assert len(fetched.json()["assets"]) == 1


def test_accepts_audio_music_assets(
    client: TestClient,
    account: tuple[dict[str, str], str],
) -> None:
    headers, workspace_id = account
    created = client.post(
        "/api/v1/projects",
        json=project_payload(workspace_id),
        headers=headers,
    )
    project_id = created.json()["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/assets",
        data={"kind": "music"},
        files={"file": ("licensed-track.mp3", b"audio-bytes", "audio/mpeg")},
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json()["kind"] == "music"


def test_rejects_non_audio_music_assets(
    client: TestClient,
    account: tuple[dict[str, str], str],
) -> None:
    headers, workspace_id = account
    created = client.post(
        "/api/v1/projects",
        json=project_payload(workspace_id),
        headers=headers,
    )
    project_id = created.json()["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/assets",
        data={"kind": "music"},
        files={"file": ("not-music.mp4", b"video-bytes", "video/mp4")},
        headers=headers,
    )

    assert response.status_code == 415


def test_rejects_non_video_source_asset(
    client: TestClient,
    account: tuple[dict[str, str], str],
) -> None:
    headers, workspace_id = account
    created = client.post(
        "/api/v1/projects",
        json=project_payload(workspace_id),
        headers=headers,
    )
    project_id = created.json()["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/assets",
        data={"kind": "source_video"},
        files={"file": ("notes.txt", b"not a video", "text/plain")},
        headers=headers,
    )
    assert response.status_code == 415


def test_rejects_conflicting_director_contract(
    client: TestClient,
    account: tuple[dict[str, str], str],
) -> None:
    headers, workspace_id = account
    payload = project_payload(workspace_id)
    payload["contract"] = {
        "objective": "Create a product reel",
        "must_include": ["emoji captions"],
        "must_avoid": ["Emoji Captions"],
    }
    response = client.post("/api/v1/projects", json=payload, headers=headers)
    assert response.status_code == 422
