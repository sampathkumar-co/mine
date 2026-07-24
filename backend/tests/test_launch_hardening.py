from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import Base, engine
from app.main import app
from app.services.rate_limit import FixedWindowRateLimiter


@pytest.fixture(autouse=True)
def reset_launch_runtime() -> Generator[None, None, None]:
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
            "display_name": "Launch Operator",
            "workspace_name": "Launch Studio",
        },
    )
    assert registered.status_code == 201
    refreshable = client.post(
        "/api/v1/auth/session",
        headers={"Authorization": f"Bearer {registered.json()['access_token']}"},
    )
    assert refreshable.status_code == 200
    return refreshable.json()


def auth(value: dict[str, object]) -> dict[str, str]:
    return {"Authorization": f"Bearer {value['access_token']}"}


def test_fixed_window_rate_limiter_blocks_after_limit() -> None:
    limiter = FixedWindowRateLimiter(
        Settings(
            rate_limit_backend="memory",
            rate_limit_enabled=True,
            rate_limit_window_seconds=60,
        )
    )
    first = limiter.check("auth:ip:test", limit=2, window_seconds=60)
    second = limiter.check("auth:ip:test", limit=2, window_seconds=60)
    third = limiter.check("auth:ip:test", limit=2, window_seconds=60)
    assert first.allowed is True
    assert second.allowed is True
    assert second.remaining == 0
    assert third.allowed is False
    assert third.reset_after_seconds > 0


def test_liveness_readiness_and_prometheus_metrics(client: TestClient) -> None:
    live = client.get("/api/v1/health/live")
    assert live.status_code == 200
    assert live.json()["version"] == "0.14.0"

    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json()["components"]["database"]["status"] == "ok"
    assert ready.json()["components"]["outputs"]["status"] == "ok"

    metrics = client.get("/api/v1/metrics")
    assert metrics.status_code == 200
    assert "director_build_info" in metrics.text
    assert "director_http_requests_total" in metrics.text
    assert "/api/v1/health/live" in metrics.text


def test_workspace_export_is_signed_and_redacted(client: TestClient) -> None:
    owner = session(client, "privacy-export@example.com")
    workspace = owner["workspaces"][0]  # type: ignore[index]
    workspace_id = workspace["id"]

    created = client.post(
        f"/api/v1/workspaces/{workspace_id}/privacy/exports",
        headers=auth(owner),
    )
    assert created.status_code == 202
    request_payload = created.json()
    assert request_payload["status"] == "ready"
    assert request_payload["result_sha256"]

    delivery = client.get(
        f"/api/v1/workspaces/{workspace_id}/privacy/requests/{request_payload['id']}/delivery",
        headers=auth(owner),
    )
    assert delivery.status_code == 200
    downloaded = client.get(delivery.json()["url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        exported = json.loads(archive.read("workspace-export.json"))
        readme = archive.read("README.txt").decode("utf-8")
    encoded = json.dumps(exported)
    assert "password_hash" not in encoded
    assert "storage_path" not in encoded
    assert exported["workspace"]["id"] == workspace_id
    assert "raw media binaries are not duplicated" in readme


def test_deletion_grace_locks_mutations_and_can_be_cancelled(client: TestClient) -> None:
    owner = session(client, "privacy-delete@example.com")
    workspace = owner["workspaces"][0]  # type: ignore[index]
    workspace_id = workspace["id"]
    slug = workspace["slug"]

    wrong = client.post(
        f"/api/v1/workspaces/{workspace_id}/privacy/deletion",
        headers=auth(owner),
        json={"confirmation": "wrong", "reason": "Closing this test workspace"},
    )
    assert wrong.status_code == 422

    scheduled = client.post(
        f"/api/v1/workspaces/{workspace_id}/privacy/deletion",
        headers=auth(owner),
        json={"confirmation": slug, "reason": "Closing this test workspace"},
    )
    assert scheduled.status_code == 202
    assert scheduled.json()["status"] == "scheduled"

    locked = client.post(
        "/api/v1/projects",
        headers=auth(owner),
        json={
            "workspace_id": workspace_id,
            "contract": {
                "objective": "This mutation must be locked during deletion grace",
                "target_duration_seconds": 30,
            },
        },
    )
    assert locked.status_code == 423

    cancelled = client.delete(
        f"/api/v1/workspaces/{workspace_id}/privacy/deletion/{scheduled.json()['id']}",
        headers=auth(owner),
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    allowed = client.post(
        "/api/v1/projects",
        headers=auth(owner),
        json={
            "workspace_id": workspace_id,
            "contract": {
                "objective": "Mutations work after deletion cancellation",
                "target_duration_seconds": 30,
            },
        },
    )
    assert allowed.status_code == 201
