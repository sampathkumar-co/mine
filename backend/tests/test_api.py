import os
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ["DIRECTOR_DATABASE_URL"] = "sqlite+pysqlite:///./test_director.db"
os.environ["DIRECTOR_UPLOAD_DIR"] = "./.test-data/uploads"
os.environ["DIRECTOR_OUTPUT_DIR"] = "./.test-data/outputs"
os.environ["DIRECTOR_MAX_UPLOAD_BYTES"] = "64"

from app.api import routes  # noqa: E402
from app.core.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


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


def project_payload() -> dict[str, object]:
    return {
        "user_id": "9afc424f-91af-4f13-b917-44f778f18b9d",
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


def test_project_upload_and_queue(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    created = client.post("/api/v1/projects", json=project_payload())
    assert created.status_code == 201
    project_id = created.json()["id"]
    assert created.json()["status"] == "created"

    intelligence = client.get(f"/api/v1/projects/{project_id}/intelligence")
    assert intelligence.status_code == 200
    assert intelligence.json()["analysis"] is None
    assert intelligence.json()["edit_decision_graph"] is None

    uploaded = client.post(
        f"/api/v1/projects/{project_id}/assets",
        data={"kind": "source_video"},
        files={"file": ("raw.mp4", b"video-bytes", "video/mp4")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["kind"] == "source_video"
    assert uploaded.json()["size_bytes"] == 11

    monkeypatch.setattr(
        routes.run_project_pipeline,
        "delay",
        lambda project_id: SimpleNamespace(id=f"task-{project_id}"),
    )
    queued = client.post(f"/api/v1/projects/{project_id}/start")
    assert queued.status_code == 202
    assert queued.json()["status"] == "queued"
    assert queued.json()["task_id"] == f"task-{project_id}"

    fetched = client.get(f"/api/v1/projects/{project_id}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "queued"
    assert fetched.json()["output_available"] is False
    assert len(fetched.json()["assets"]) == 1


def test_accepts_audio_music_assets(client: TestClient) -> None:
    created = client.post("/api/v1/projects", json=project_payload())
    project_id = created.json()["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/assets",
        data={"kind": "music"},
        files={"file": ("licensed-track.mp3", b"audio-bytes", "audio/mpeg")},
    )

    assert response.status_code == 201
    assert response.json()["kind"] == "music"


def test_rejects_non_audio_music_assets(client: TestClient) -> None:
    created = client.post("/api/v1/projects", json=project_payload())
    project_id = created.json()["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/assets",
        data={"kind": "music"},
        files={"file": ("not-music.mp4", b"video-bytes", "video/mp4")},
    )

    assert response.status_code == 415


def test_rejects_non_video_source_asset(client: TestClient) -> None:
    created = client.post("/api/v1/projects", json=project_payload())
    project_id = created.json()["id"]

    response = client.post(
        f"/api/v1/projects/{project_id}/assets",
        data={"kind": "source_video"},
        files={"file": ("notes.txt", b"not a video", "text/plain")},
    )
    assert response.status_code == 415


def test_rejects_conflicting_director_contract(client: TestClient) -> None:
    payload = project_payload()
    payload["contract"] = {
        "objective": "Create a product reel",
        "must_include": ["emoji captions"],
        "must_avoid": ["Emoji Captions"],
    }
    response = client.post("/api/v1/projects", json=payload)
    assert response.status_code == 422
