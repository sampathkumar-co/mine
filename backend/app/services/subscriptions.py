from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.operations import BillingAccount
from app.models.subscriptions import BillingWebhookEvent, WorkspaceSubscription
from app.services.billing import credits, ensure_billing_account, post_adjustment
from app.services.entitlements import (
    ENTITLED_SUBSCRIPTION_STATUSES,
    PlanDefinition,
    ensure_subscription,
    load_plan_catalog,
    plan_for_price,
)


class SubscriptionError(ValueError):
    pass


class PaymentProviderError(SubscriptionError):
    pass


@dataclass(frozen=True, slots=True)
class HostedBillingSession:
    url: str
    expires_at: datetime | None
    customer_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessedWebhook:
    record: BillingWebhookEvent
    duplicate: bool


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(int(value), UTC)
    except (TypeError, ValueError, OSError):
        return None


def _string_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        candidate = value.get("id")
        return str(candidate) if candidate else None
    return None


def _invoice_subscription_id(invoice: dict[str, Any]) -> str | None:
    direct = _string_id(invoice.get("subscription"))
    if direct:
        return direct
    parent = invoice.get("parent")
    if isinstance(parent, dict):
        details = parent.get("subscription_details")
        if isinstance(details, dict):
            return _string_id(details.get("subscription"))
    return None


def _subscription_price_id(subscription: dict[str, Any]) -> str | None:
    items = subscription.get("items")
    if not isinstance(items, dict):
        return None
    data = items.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    price = first.get("price")
    return _string_id(price)


def _workspace_id_from_metadata(payload: dict[str, Any]) -> UUID | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("workspace_id")
    try:
        return UUID(str(raw)) if raw else None
    except ValueError:
        return None


def _find_subscription(
    db: Session,
    *,
    workspace_id: UUID | None = None,
    customer_id: str | None = None,
    subscription_id: str | None = None,
) -> WorkspaceSubscription | None:
    if workspace_id is not None:
        return db.get(WorkspaceSubscription, workspace_id)
    if subscription_id:
        found = db.scalar(
            select(WorkspaceSubscription).where(
                WorkspaceSubscription.provider_subscription_id == subscription_id
            )
        )
        if found is not None:
            return found
    if customer_id:
        return db.scalar(
            select(WorkspaceSubscription).where(
                WorkspaceSubscription.provider_customer_id == customer_id
            )
        )
    return None


def _event_summary(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("data", {}).get("object", {})
    if not isinstance(payload, dict):
        payload = {}
    return {
        "object_id": _string_id(payload.get("id")),
        "customer_id": _string_id(payload.get("customer")),
        "subscription_id": _string_id(payload.get("subscription"))
        or _invoice_subscription_id(payload),
        "workspace_id": str(_workspace_id_from_metadata(payload) or "") or None,
    }


class StripeBillingProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.stripe_secret_key:
            raise PaymentProviderError("Stripe billing is not configured")
        try:
            import stripe
        except ImportError as exc:
            raise PaymentProviderError("The Stripe Python package is not installed") from exc
        stripe.api_key = settings.stripe_secret_key
        self.stripe = stripe

    def create_checkout_session(
        self,
        *,
        workspace_id: UUID,
        workspace_name: str,
        customer_email: str,
        plan: PlanDefinition,
        customer_id: str | None,
    ) -> HostedBillingSession:
        if not plan.price_id:
            raise PaymentProviderError(f"The {plan.name} plan has no configured Stripe Price")
        try:
            if not customer_id:
                customer = self.stripe.Customer.create(
                    email=customer_email,
                    name=workspace_name,
                    metadata={"workspace_id": str(workspace_id)},
                )
                customer_id = str(customer.id)
            base_url = self.settings.public_app_url.rstrip("/")
            session = self.stripe.checkout.Session.create(
                mode="subscription",
                customer=customer_id,
                client_reference_id=str(workspace_id),
                line_items=[{"price": plan.price_id, "quantity": 1}],
                success_url=(
                    f"{base_url}/workspaces/{workspace_id}/settings?billing=success"
                    "&session_id={CHECKOUT_SESSION_ID}"
                ),
                cancel_url=f"{base_url}/workspaces/{workspace_id}/settings?billing=cancelled",
                metadata={"workspace_id": str(workspace_id), "plan_key": plan.key},
                subscription_data={
                    "metadata": {"workspace_id": str(workspace_id), "plan_key": plan.key}
                },
                allow_promotion_codes=True,
            )
        except Exception as exc:
            raise PaymentProviderError("Stripe Checkout session creation failed") from exc
        return HostedBillingSession(
            url=str(session.url),
            expires_at=_timestamp(getattr(session, "expires_at", None)),
            customer_id=customer_id,
        )

    def create_portal_session(self, *, customer_id: str, workspace_id: UUID) -> HostedBillingSession:
        try:
            kwargs: dict[str, Any] = {
                "customer": customer_id,
                "return_url": (
                    f"{self.settings.public_app_url.rstrip('/')}/workspaces/"
                    f"{workspace_id}/settings"
                ),
            }
            if self.settings.stripe_portal_configuration_id:
                kwargs["configuration"] = self.settings.stripe_portal_configuration_id
            session = self.stripe.billing_portal.Session.create(**kwargs)
        except Exception as exc:
            raise PaymentProviderError("Stripe customer portal session creation failed") from exc
        return HostedBillingSession(url=str(session.url), expires_at=None, customer_id=customer_id)

    def verify_webhook(self, payload: bytes, signature: str) -> dict[str, Any]:
        if not self.settings.stripe_webhook_secret:
            raise PaymentProviderError("Stripe webhook verification is not configured")
        try:
            event = self.stripe.Webhook.construct_event(
                payload,
                signature,
                self.settings.stripe_webhook_secret,
            )
        except Exception as exc:
            raise PaymentProviderError("Stripe webhook signature verification failed") from exc
        if hasattr(event, "to_dict_recursive"):
            return event.to_dict_recursive()
        return dict(event)


def billing_provider(settings: Settings) -> StripeBillingProvider:
    if settings.billing_provider.casefold() != "stripe":
        raise PaymentProviderError(f"Unsupported billing provider: {settings.billing_provider}")
    return StripeBillingProvider(settings)


def _sync_subscription(
    db: Session,
    payload: dict[str, Any],
    settings: Settings,
) -> WorkspaceSubscription | None:
    workspace_id = _workspace_id_from_metadata(payload)
    customer_id = _string_id(payload.get("customer"))
    subscription_id = _string_id(payload.get("id"))
    subscription = _find_subscription(
        db,
        workspace_id=workspace_id,
        customer_id=customer_id,
        subscription_id=subscription_id,
    )
    if subscription is None and workspace_id is not None:
        subscription = ensure_subscription(db, workspace_id)
    if subscription is None:
        return None

    price_id = _subscription_price_id(payload)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    requested_plan = str(metadata.get("plan_key") or "")
    plan = plan_for_price(settings, price_id)
    catalog = load_plan_catalog(settings)
    if plan is None and requested_plan in catalog:
        plan = catalog[requested_plan]

    subscription.provider_customer_id = customer_id or subscription.provider_customer_id
    subscription.provider_subscription_id = subscription_id or subscription.provider_subscription_id
    subscription.provider_price_id = price_id or subscription.provider_price_id
    subscription.status = str(payload.get("status") or subscription.status)
    subscription.current_period_start = _timestamp(payload.get("current_period_start"))
    subscription.current_period_end = _timestamp(payload.get("current_period_end"))
    subscription.cancel_at_period_end = bool(payload.get("cancel_at_period_end", False))
    if plan is not None:
        subscription.plan_key = plan.key

    account = ensure_billing_account(db, subscription.workspace_id)
    account.plan = (
        subscription.plan_key
        if subscription.status in ENTITLED_SUBSCRIPTION_STATUSES
        else "starter"
    )
    db.flush()
    return subscription


def _process_checkout_completed(
    db: Session,
    payload: dict[str, Any],
    settings: Settings,
) -> bool:
    workspace_id = _workspace_id_from_metadata(payload)
    if workspace_id is None:
        reference = payload.get("client_reference_id")
        try:
            workspace_id = UUID(str(reference)) if reference else None
        except ValueError:
            workspace_id = None
    if workspace_id is None:
        return False
    subscription = ensure_subscription(db, workspace_id)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    plan_key = str(metadata.get("plan_key") or subscription.plan_key)
    if plan_key in load_plan_catalog(settings):
        subscription.plan_key = plan_key
    subscription.provider_customer_id = (
        _string_id(payload.get("customer")) or subscription.provider_customer_id
    )
    subscription.provider_subscription_id = (
        _string_id(payload.get("subscription")) or subscription.provider_subscription_id
    )
    if subscription.status in {"inactive", "checkout_pending"}:
        subscription.status = "checkout_completed"
    db.flush()
    return True


def _process_invoice_paid(
    db: Session,
    payload: dict[str, Any],
    settings: Settings,
) -> bool:
    invoice_id = _string_id(payload.get("id"))
    subscription = _find_subscription(
        db,
        customer_id=_string_id(payload.get("customer")),
        subscription_id=_invoice_subscription_id(payload),
    )
    if subscription is None or invoice_id is None:
        return False
    plan = load_plan_catalog(settings).get(subscription.plan_key)
    if plan is None:
        return False
    subscription.last_invoice_id = invoice_id
    subscription.last_payment_failed_at = None
    account = ensure_billing_account(db, subscription.workspace_id)
    if subscription.status in ENTITLED_SUBSCRIPTION_STATUSES:
        account.plan = subscription.plan_key
    if plan.monthly_credits > 0:
        post_adjustment(
            db,
            subscription.workspace_id,
            plan.monthly_credits,
            idempotency_key=f"stripe:invoice:{invoice_id}:credits",
            description=f"{plan.name} subscription credit grant",
            kind="subscription_grant",
            metadata={
                "provider": "stripe",
                "invoice_id": invoice_id,
                "plan_key": plan.key,
                "credits": str(credits(plan.monthly_credits)),
            },
        )
    db.flush()
    return True


def _process_invoice_failed(db: Session, payload: dict[str, Any]) -> bool:
    subscription = _find_subscription(
        db,
        customer_id=_string_id(payload.get("customer")),
        subscription_id=_invoice_subscription_id(payload),
    )
    if subscription is None:
        return False
    subscription.status = "past_due"
    subscription.last_payment_failed_at = datetime.now(UTC)
    db.flush()
    return True


def process_billing_event(
    db: Session,
    event: dict[str, Any],
    settings: Settings,
) -> ProcessedWebhook:
    event_id = str(event.get("id") or "")
    event_type = str(event.get("type") or "")
    if not event_id or not event_type:
        raise SubscriptionError("Billing event is missing an id or type")

    existing = db.scalar(
        select(BillingWebhookEvent).where(
            BillingWebhookEvent.provider == "stripe",
            BillingWebhookEvent.provider_event_id == event_id,
        )
    )
    if existing is not None and existing.status in {"processed", "ignored"}:
        return ProcessedWebhook(record=existing, duplicate=True)

    record = existing or BillingWebhookEvent(
        provider="stripe",
        provider_event_id=event_id,
        event_type=event_type,
        livemode=bool(event.get("livemode", False)),
    )
    if existing is None:
        db.add(record)
    record.status = "processing"
    record.error_message = None
    record.event_metadata = _event_summary(event)
    db.flush()

    payload = event.get("data", {}).get("object", {})
    if not isinstance(payload, dict):
        payload = {}
    handled = False
    try:
        if event_type == "checkout.session.completed":
            handled = _process_checkout_completed(db, payload, settings)
        elif event_type in {
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
            "customer.subscription.paused",
            "customer.subscription.resumed",
        }:
            handled = _sync_subscription(db, payload, settings) is not None
        elif event_type == "invoice.paid":
            handled = _process_invoice_paid(db, payload, settings)
        elif event_type == "invoice.payment_failed":
            handled = _process_invoice_failed(db, payload)
    except Exception as exc:
        record.status = "failed"
        record.error_message = str(exc)[:2_000]
        db.flush()
        raise

    record.status = "processed" if handled else "ignored"
    record.processed_at = datetime.now(UTC)
    db.flush()
    return ProcessedWebhook(record=record, duplicate=False)


def billing_overview(
    db: Session,
    workspace_id: UUID,
    settings: Settings,
) -> tuple[BillingAccount, WorkspaceSubscription | None, PlanDefinition]:
    account = ensure_billing_account(db, workspace_id)
    subscription = db.get(WorkspaceSubscription, workspace_id)
    catalog = load_plan_catalog(settings)
    plan = catalog.get(account.plan, catalog["starter"])
    return account, subscription, plan
