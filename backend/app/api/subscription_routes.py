from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.platform import User, Workspace
from app.schemas.subscriptions import (
    BillingOverviewRead,
    CheckoutCreate,
    HostedBillingSessionRead,
    PlanRead,
    SubscriptionRead,
    WebhookAccepted,
)
from app.services.billing import credits
from app.services.entitlements import (
    ensure_subscription,
    load_plan_catalog,
    public_plan_payload,
)
from app.services.permissions import membership_for
from app.services.subscriptions import (
    PaymentProviderError,
    SubscriptionError,
    billing_overview,
    billing_provider,
    process_billing_event,
)

router = APIRouter()
settings = get_settings()


def _current_user_id(request: Request) -> UUID:
    user_id = getattr(request.state, "user_id", None)
    if not isinstance(user_id, UUID):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user_id


def _membership(db: Session, workspace_id: UUID, request: Request):
    membership = membership_for(db, workspace_id, _current_user_id(request))
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    return membership


def _owner(db: Session, workspace_id: UUID, request: Request):
    membership = _membership(db, workspace_id, request)
    if membership.role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace owner permission is required",
        )
    return membership


@router.get("/billing/plans", response_model=list[PlanRead], tags=["subscriptions"])
def list_billing_plans() -> list[PlanRead]:
    return [
        PlanRead.model_validate(public_plan_payload(plan))
        for plan in load_plan_catalog(settings).values()
    ]


@router.get(
    "/workspaces/{workspace_id}/subscription",
    response_model=BillingOverviewRead,
    tags=["subscriptions"],
)
def get_subscription_overview(
    workspace_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> BillingOverviewRead:
    _membership(db, workspace_id, request)
    account, subscription, plan = billing_overview(db, workspace_id, settings)
    db.commit()
    plan_payload = public_plan_payload(plan, current=True)
    return BillingOverviewRead(
        workspace_id=workspace_id,
        plan=PlanRead.model_validate(plan_payload),
        subscription=(SubscriptionRead.model_validate(subscription) if subscription else None),
        balance_credits=account.balance_credits,
        reserved_credits=account.reserved_credits,
        available_credits=credits(account.balance_credits - account.reserved_credits),
        portal_available=bool(
            settings.subscriptions_enabled
            and subscription
            and subscription.provider_customer_id
        ),
    )


@router.post(
    "/workspaces/{workspace_id}/subscription/checkout",
    response_model=HostedBillingSessionRead,
    tags=["subscriptions"],
)
def create_subscription_checkout(
    workspace_id: UUID,
    payload: CheckoutCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> HostedBillingSessionRead:
    if not settings.subscriptions_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription checkout is not enabled",
        )
    membership = _owner(db, workspace_id, request)
    workspace = db.get(Workspace, workspace_id)
    user = db.get(User, membership.user_id)
    if workspace is None or user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    plans = load_plan_catalog(settings)
    plan = plans.get(payload.plan_key)
    if plan is None or plan.key == "starter":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown paid plan")
    if not plan.checkout_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The {plan.name} plan has no configured checkout Price",
        )

    subscription = ensure_subscription(db, workspace_id)
    if (
        subscription.provider_subscription_id
        and subscription.status in {"active", "trialing", "past_due"}
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Manage the existing subscription through the billing portal",
        )
    try:
        hosted = billing_provider(settings).create_checkout_session(
            workspace_id=workspace_id,
            workspace_name=workspace.name,
            customer_email=user.email,
            plan=plan,
            customer_id=subscription.provider_customer_id,
        )
    except PaymentProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    subscription.provider_customer_id = hosted.customer_id or subscription.provider_customer_id
    subscription.plan_key = plan.key
    subscription.status = "checkout_pending"
    db.commit()
    return HostedBillingSessionRead(url=hosted.url, expires_at=hosted.expires_at)


@router.post(
    "/workspaces/{workspace_id}/subscription/portal",
    response_model=HostedBillingSessionRead,
    tags=["subscriptions"],
)
def create_subscription_portal(
    workspace_id: UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> HostedBillingSessionRead:
    if not settings.subscriptions_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subscription billing is not enabled",
        )
    _owner(db, workspace_id, request)
    subscription = ensure_subscription(db, workspace_id)
    if not subscription.provider_customer_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This workspace does not have a billing customer yet",
        )
    try:
        hosted = billing_provider(settings).create_portal_session(
            customer_id=subscription.provider_customer_id,
            workspace_id=workspace_id,
        )
    except PaymentProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    db.commit()
    return HostedBillingSessionRead(url=hosted.url, expires_at=hosted.expires_at)


@router.post(
    "/billing/webhooks/stripe",
    response_model=WebhookAccepted,
    tags=["subscriptions"],
)
async def receive_stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
    db: Session = Depends(get_db),
) -> WebhookAccepted:
    if not settings.subscriptions_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not stripe_signature:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing Stripe signature")
    payload = await request.body()
    try:
        event = billing_provider(settings).verify_webhook(payload, stripe_signature)
    except PaymentProviderError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    try:
        processed = process_billing_event(db, event, settings)
        db.commit()
    except SubscriptionError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Billing event processing failed",
        ) from exc
    return WebhookAccepted(
        duplicate=processed.duplicate,
        status=processed.record.status,
    )
