"""Add subscription state and billing webhook events.

Revision ID: 20260724_0003
Revises: 20260724_0002
"""
from __future__ import annotations

from alembic import op

from app.core.database import Base, import_models

revision = "20260724_0003"
down_revision = "20260724_0002"
branch_labels = None
depends_on = None

SUBSCRIPTION_TABLES = {"workspace_subscriptions", "billing_webhook_events"}


def upgrade() -> None:
    import_models()
    bind = op.get_bind()
    for table in Base.metadata.sorted_tables:
        if table.name in SUBSCRIPTION_TABLES:
            table.create(bind, checkfirst=True)


def downgrade() -> None:
    import_models()
    bind = op.get_bind()
    for table in reversed(Base.metadata.sorted_tables):
        if table.name in SUBSCRIPTION_TABLES:
            table.drop(bind, checkfirst=True)
