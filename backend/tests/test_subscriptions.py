from __future__ import annotations

from collections.abc import Generator
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import Base, SessionLocal, engine
from app.main import app
from app.models.operations import BillingEntry
from app.models.subscriptions import BillingWebhookEvent
from app.services.billing import ensure_billing_account
from app.services.entitlements import ensure_subscription, load_plan_catalog
from app.services.subscriptions import process_billing_event

settings = get_settings()


@pytest.fixture(autouse=True)
def reset_subscription_runtime() -> Generator[None, None, None]:
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
            "workspace_name": "Subscription Studio",
        },
    )
    assert registered.status_code == 201
    assert registered.json()["refresh_token"] is None
    assert client.cookies.get("director_refresh")
    return registered.json()


def auth(value: dict[str, object]) -> dict[str, str]:
    return {"Authorization": f"Bearer {value['access_token']}"}


def workspace_id(value: dict[str, object]) -> UUID:
    return UUID(str(value["workspaces"][0]["id"]))  # type: ignore[index]


def test_starter_plan_contract_and_seat_limits_are_enforced(client: TestClient) -> None:
    owner = session(client, "plan-owner@example.com")
    target_workspace_id = workspace_id(owner)

    too_long = client.post(
        "/api/v1/projects",
        headers=auth(owner),
        json={
            "workspace_id": str(target_workspace_id),
            "contract": {
                "objective": "This production exceeds the starter duration",
                "target_duration_seconds": 181,
            },
        },
    )
    assert too_long.status_code == 402
    assert "180 seconds" in too_long.json()["detail"]

    allowed = client.post(
        "/api/v1/projects",
        headers=auth(owner),
        json={
            "workspace_id": str(target_workspace_id),
            "contract": {
                "objective": "This production fits the starter duration",
                "target_duration_seconds": 180,
            },
        },
    )
    assert allowed.status_code == 201

    for index in range(2):
        invited = client.post(
            f"/api/v1/workspaces/{target_workspace_id}/invitations",
            headers=auth(owner),
            json={"email": f"seat-{index}@example.com", "role": "editor"},
        )
        assert invited.status_code == 201
    blocked = client.post(
        f"/api/v1/workspaces/{target_workspace_id}/invitations",
        headers=auth(owner),
        json={"email": "seat-overflow@example.com", "role": "editor"},
    )
    assert blocked.status_code == 402
    assert "3 workspace members" in blocked.json()["detail"]


def test_invoice_credit_grant_and_webhook_event_are_idempotent(client: TestClient) -> None:
    owner = session(client, "invoice-owner@example.com")
    target_workspace_id = workspace_id(owner)

    with SessionLocal() as db:
        account = ensure_billing_account(db, target_workspace_id)
        account.balance_credits = Decimal("0")
        subscription = ensure_subscription(db, target_workspace_id)
        subscription.provider_customer_id = "cus_director_test"
        subscription.provider_subscription_id = "sub_director_test"
        subscription.plan_key = "creator"
        subscription.status = "active"
        db.commit()

        event = {
            "id": "evt_invoice_paid_001",
            "type": "invoice.paid",
            "livemode": False,
            "data": {
                "object": {
                    "id": "in_director_001",
                    "customer": "cus_director_test",
                    "subscription": "sub_director_test",
                }
            },
        }
        first = process_billing_event(db, event, settings)
        db.commit()
        second = process_billing_event(db, event, settings)
        db.commit()

        assert first.duplicate is False
        assert second.duplicate is True
        account = ensure_billing_account(db, target_workspace_id)
        assert account.balance_credits == Decimal("250.0000")
        entries = list(
            db.scalars(
                select(BillingEntry).where(
                    BillingEntry.idempotency_key == "stripe:invoice:in_director_001:credits"
                )
            ).all()
        )
        assert len(entries) == 1
        events = list(
            db.scalars(
                select(BillingWebhookEvent).where(
                    BillingWebhookEvent.provider_event_id == "evt_invoice_paid_001"
                )
            ).all()
        )
        assert len(events) == 1
        assert events[0].status == "processed"


def test_subscription_updates_activate_and_cancel_plan(client: TestClient) -> None:
    owner = session(client, "subscription-owner@example.com")
    target_workspace_id = workspace_id(owner)
    assert load_plan_catalog(settings)["creator"].max_tier == 3

    with SessionLocal() as db:
        subscription = ensure_subscription(db, target_workspace_id)
        subscription.provider_customer_id = "cus_subscription_test"
        db.commit()

        activated = {
            "id": "evt_subscription_active_001",
            "type": "customer.subscription.updated",
            "livemode": False,
            "data": {
                "object": {
                    "id": "sub_subscription_test",
                    "customer": "cus_subscription_test",
                    "status": "active",
                    "metadata": {
                        "workspace_id": str(target_workspace_id),
                        "plan_key": "creator",
                    },
                    "items": {"data": []},
                    "current_period_start": 1_700_000_000,
                    "current_period_end": 1_702_592_000,
                    "cancel_at_period_end": False,
                }
            },
        }
        process_billing_event(db, activated, settings)
        db.commit()
        account = ensure_billing_account(db, target_workspace_id)
        assert account.plan == "creator"

        cancelled = {
            "id": "evt_subscription_cancelled_001",
            "type": "customer.subscription.deleted",
            "livemode": False,
            "data": {
                "object": {
                    "id": "sub_subscription_test",
                    "customer": "cus_subscription_test",
                    "status": "canceled",
                    "metadata": {
                        "workspace_id": str(target_workspace_id),
                        "plan_key": "creator",
                    },
                    "items": {"data": []},
                }
            },
        }
        process_billing_event(db, cancelled, settings)
        db.commit()
        account = ensure_billing_account(db, target_workspace_id)
        assert account.plan == "starter"
