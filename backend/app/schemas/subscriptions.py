from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PlanRead(BaseModel):
    key: str
    name: str
    description: str
    monthly_credits: Decimal
    max_source_clips: int
    max_target_duration_seconds: int
    max_members: int
    max_tier: int
    checkout_available: bool
    current: bool = False


class SubscriptionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    workspace_id: UUID
    provider: str
    plan_key: str
    status: str
    current_period_start: datetime | None
    current_period_end: datetime | None
    cancel_at_period_end: bool
    last_payment_failed_at: datetime | None
    updated_at: datetime


class BillingOverviewRead(BaseModel):
    workspace_id: UUID
    plan: PlanRead
    subscription: SubscriptionRead | None
    balance_credits: Decimal
    reserved_credits: Decimal
    available_credits: Decimal
    portal_available: bool


class CheckoutCreate(BaseModel):
    plan_key: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_-]+$")


class HostedBillingSessionRead(BaseModel):
    url: str
    expires_at: datetime | None = None


class WebhookAccepted(BaseModel):
    received: bool = True
    duplicate: bool = False
    status: str
