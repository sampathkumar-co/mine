"""Add sessions, recovery, teams, multipart storage, audit, and billing.

Revision ID: 20260724_0002
Revises: 20260724_0001
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.core.database import Base, import_models

revision = "20260724_0002"
down_revision = "20260724_0001"
branch_labels = None
depends_on = None

PLATFORM_TABLES = {
    "users",
    "workspaces",
    "workspace_memberships",
    "resumable_uploads",
}
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


def _add_workspace_ownership_if_missing() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "projects" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("projects")}
    if "workspace_id" in columns:
        return
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("projects") as batch:
            batch.add_column(sa.Column("workspace_id", sa.Uuid(), nullable=True))
            batch.create_foreign_key(
                "fk_projects_workspace_id_workspaces",
                "workspaces",
                ["workspace_id"],
                ["id"],
                ondelete="CASCADE",
            )
            batch.create_index("ix_projects_workspace_id", ["workspace_id"])
    else:
        op.add_column("projects", sa.Column("workspace_id", sa.Uuid(), nullable=True))
        op.create_foreign_key(
            "fk_projects_workspace_id_workspaces",
            "projects",
            "workspaces",
            ["workspace_id"],
            ["id"],
            ondelete="CASCADE",
        )
        op.create_index("ix_projects_workspace_id", "projects", ["workspace_id"])


def upgrade() -> None:
    import_models()
    bind = op.get_bind()
    selected = PLATFORM_TABLES | OPERATION_TABLES
    for table in Base.metadata.sorted_tables:
        if table.name in selected:
            table.create(bind, checkfirst=True)
    _add_workspace_ownership_if_missing()


def downgrade() -> None:
    import_models()
    bind = op.get_bind()
    for table in reversed(Base.metadata.sorted_tables):
        if table.name in OPERATION_TABLES:
            table.drop(bind, checkfirst=True)
