"""Baseline the Director OS schema before production operations.

Revision ID: 20260724_0001
Revises: None
"""
from __future__ import annotations

from alembic import op
from app.core.database import Base, import_models

revision = "20260724_0001"
down_revision = None
branch_labels = None
depends_on = None

OPERATION_TABLES = {
    "account_tokens",
    "user_email_statuses",
    "email_outbox",
    "auth_sessions",
    "workspace_invitations",
    "audit_events",
    "billing_accounts",
    "billing_entries",
    "project_billing_reservations",
    "multipart_uploads",
    "multipart_upload_parts",
    "stored_objects",
}


def upgrade() -> None:
    import_models()
    bind = op.get_bind()
    for table in Base.metadata.sorted_tables:
        if table.name not in OPERATION_TABLES:
            table.create(bind, checkfirst=True)


def downgrade() -> None:
    # This revision is a stampable baseline for existing installations.
    # Destructive downgrade is intentionally disabled.
    pass
