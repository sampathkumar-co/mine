from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.operations import BillingAccount, WorkspaceInvitation
from app.models.platform import WorkspaceMembership
from app.models.project import Project
from app.models.subscriptions import WorkspaceSubscription

ENTITLED_SUBSCRIPTION_STATUSES = {"active", "trialing", "past_due"}


class EntitlementError(ValueError):
    pass


class PlanConfigurationError(ValueError):
    pass


class PlanDefinition(BaseModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    name: str = Field(min_length=2, max_length=80)
    description: str = Field(default="", max_length=240)
    price_id: str | None = Field(default=None, max_length=255)
    monthly_credits: Decimal = Field(default=Decimal("0"), ge=0)
    max_source_clips: int = Field(ge=1, le=100)
    max_target_duration_seconds: int = Field(ge=15, le=10_800)
    max_members: int = Field(ge=1, le=1_000)
    max_tier: int = Field(ge=1, le=6)

    @property
    def checkout_available(self) -> bool:
        return bool(self.price_id)


def load_plan_catalog(settings: Settings) -> dict[str, PlanDefinition]:
    try:
        raw = json.loads(settings.billing_plans_json)
    except json.JSONDecodeError as exc:
        raise PlanConfigurationError("DIRECTOR_BILLING_PLANS_JSON is not valid JSON") from exc
    if not isinstance(raw, dict) or not raw:
        raise PlanConfigurationError("At least one billing plan is required")

    plans: dict[str, PlanDefinition] = {}
    try:
        for key, payload in raw.items():
            if not isinstance(payload, dict):
                raise PlanConfigurationError(f"Billing plan {key!r} must be an object")
            plans[str(key)] = PlanDefinition(key=str(key), **payload)
    except ValidationError as exc:
        raise PlanConfigurationError(f"Billing plan configuration is invalid: {exc}") from exc
    if "starter" not in plans:
        raise PlanConfigurationError("Billing plan catalog must include a starter plan")
    return plans


def plan_for_price(settings: Settings, price_id: str | None) -> PlanDefinition | None:
    if not price_id:
        return None
    return next(
        (plan for plan in load_plan_catalog(settings).values() if plan.price_id == price_id),
        None,
    )


def ensure_subscription(db: Session, workspace_id: UUID) -> WorkspaceSubscription:
    subscription = db.get(WorkspaceSubscription, workspace_id)
    if subscription is None:
        subscription = WorkspaceSubscription(workspace_id=workspace_id)
        db.add(subscription)
        db.flush()
    return subscription


def active_plan_key(db: Session, workspace_id: UUID, settings: Settings) -> str:
    plans = load_plan_catalog(settings)
    subscription = db.get(WorkspaceSubscription, workspace_id)
    if subscription is not None:
        if subscription.status in ENTITLED_SUBSCRIPTION_STATUSES and subscription.plan_key in plans:
            return subscription.plan_key
        if subscription.status not in {"inactive", "checkout_pending"}:
            return "starter"
    account = db.get(BillingAccount, workspace_id)
    if account is not None and account.plan in plans:
        return account.plan
    return "starter"


def active_plan(db: Session, workspace_id: UUID, settings: Settings) -> PlanDefinition:
    plans = load_plan_catalog(settings)
    return plans.get(active_plan_key(db, workspace_id, settings), plans["starter"])


def public_plan_payload(plan: PlanDefinition, *, current: bool = False) -> dict[str, Any]:
    return {
        "key": plan.key,
        "name": plan.name,
        "description": plan.description,
        "monthly_credits": plan.monthly_credits,
        "max_source_clips": plan.max_source_clips,
        "max_target_duration_seconds": plan.max_target_duration_seconds,
        "max_members": plan.max_members,
        "max_tier": plan.max_tier,
        "checkout_available": plan.checkout_available,
        "current": current,
    }


def enforce_contract_entitlements(
    db: Session,
    workspace_id: UUID,
    contract: dict[str, Any],
    settings: Settings,
) -> PlanDefinition:
    plan = active_plan(db, workspace_id, settings)
    if not settings.entitlements_enabled:
        return plan
    duration = int(contract.get("target_duration_seconds", 45))
    tier = int(contract.get("tier", 1))
    if duration > plan.max_target_duration_seconds:
        raise EntitlementError(
            f"The {plan.name} plan supports videos up to "
            f"{plan.max_target_duration_seconds} seconds"
        )
    if tier > plan.max_tier:
        raise EntitlementError(f"The {plan.name} plan supports Director Tier {plan.max_tier} or lower")
    return plan


def enforce_project_entitlements(
    db: Session,
    project: Project,
    settings: Settings,
) -> PlanDefinition:
    if project.workspace_id is None:
        raise EntitlementError("Project has no entitlement workspace")
    plan = enforce_contract_entitlements(db, project.workspace_id, project.contract, settings)
    if not settings.entitlements_enabled:
        return plan
    source_count = sum(
        1
        for asset in project.assets
        if asset.kind.value in {"source_video", "pickup_video"}
    )
    if source_count > plan.max_source_clips:
        raise EntitlementError(
            f"The {plan.name} plan supports up to {plan.max_source_clips} source and pickup clips"
        )
    return plan


def enforce_member_limit(db: Session, workspace_id: UUID, settings: Settings) -> PlanDefinition:
    plan = active_plan(db, workspace_id, settings)
    if not settings.entitlements_enabled:
        return plan
    member_count = int(
        db.scalar(
            select(func.count(WorkspaceMembership.id)).where(
                WorkspaceMembership.workspace_id == workspace_id
            )
        )
        or 0
    )
    now = datetime.now(UTC)
    open_invites = int(
        db.scalar(
            select(func.count(WorkspaceInvitation.id)).where(
                WorkspaceInvitation.workspace_id == workspace_id,
                WorkspaceInvitation.accepted_at.is_(None),
                WorkspaceInvitation.revoked_at.is_(None),
                WorkspaceInvitation.expires_at > now,
            )
        )
        or 0
    )
    if member_count + open_invites >= plan.max_members:
        raise EntitlementError(
            f"The {plan.name} plan supports up to {plan.max_members} workspace members"
        )
    return plan
