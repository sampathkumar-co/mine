"""Add immutable subscription and webhook schema.

Revision ID: 20260724_0003
Revises: 20260724_0002
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260724_0003"
down_revision = "20260724_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('billing_webhook_events',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('provider', sa.String(length=32), nullable=False),
    sa.Column('provider_event_id', sa.String(length=255), nullable=False),
    sa.Column('event_type', sa.String(length=120), nullable=False),
    sa.Column('livemode', sa.Boolean(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('event_metadata', sa.JSON(), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('provider', 'provider_event_id', name='uq_billing_webhook_event')
    )
    op.create_index(op.f('ix_billing_webhook_events_event_type'), 'billing_webhook_events', ['event_type'], unique=False)
    op.create_index(op.f('ix_billing_webhook_events_provider'), 'billing_webhook_events', ['provider'], unique=False)
    op.create_index(op.f('ix_billing_webhook_events_provider_event_id'), 'billing_webhook_events', ['provider_event_id'], unique=False)
    op.create_index(op.f('ix_billing_webhook_events_status'), 'billing_webhook_events', ['status'], unique=False)
    op.create_table('workspace_subscriptions',
    sa.Column('workspace_id', sa.Uuid(), nullable=False),
    sa.Column('provider', sa.String(length=32), nullable=False),
    sa.Column('provider_customer_id', sa.String(length=255), nullable=True),
    sa.Column('provider_subscription_id', sa.String(length=255), nullable=True),
    sa.Column('provider_price_id', sa.String(length=255), nullable=True),
    sa.Column('plan_key', sa.String(length=64), nullable=False),
    sa.Column('status', sa.String(length=40), nullable=False),
    sa.Column('current_period_start', sa.DateTime(timezone=True), nullable=True),
    sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=True),
    sa.Column('cancel_at_period_end', sa.Boolean(), nullable=False),
    sa.Column('last_invoice_id', sa.String(length=255), nullable=True),
    sa.Column('last_payment_failed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('workspace_id')
    )
    op.create_index(op.f('ix_workspace_subscriptions_last_invoice_id'), 'workspace_subscriptions', ['last_invoice_id'], unique=False)
    op.create_index(op.f('ix_workspace_subscriptions_plan_key'), 'workspace_subscriptions', ['plan_key'], unique=False)
    op.create_index(op.f('ix_workspace_subscriptions_provider_customer_id'), 'workspace_subscriptions', ['provider_customer_id'], unique=True)
    op.create_index(op.f('ix_workspace_subscriptions_provider_price_id'), 'workspace_subscriptions', ['provider_price_id'], unique=False)
    op.create_index(op.f('ix_workspace_subscriptions_provider_subscription_id'), 'workspace_subscriptions', ['provider_subscription_id'], unique=True)
    op.create_index(op.f('ix_workspace_subscriptions_status'), 'workspace_subscriptions', ['status'], unique=False)


def downgrade() -> None:
    op.drop_table("workspace_subscriptions")
    op.drop_table("billing_webhook_events")
