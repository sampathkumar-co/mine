"""Harden sessions, jobs, trials, invitations, and deletion records.

Revision ID: 20260725_0005
Revises: 20260724_0004
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260725_0005"
down_revision = "20260724_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("auth_sessions", sa.Column("csrf_token_hash", sa.String(length=64), nullable=True))
    op.add_column(
        "projects",
        sa.Column("run_generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "projects",
        sa.Column("revision_generation", sa.Integer(), nullable=False, server_default=sa.text("1")),
    )

    op.create_table(
        "user_starter_credit_grants",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=True),
        sa.Column("amount_credits", sa.Numeric(precision=14, scale=4), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("workspace_id"),
    )
    op.create_index(
        "ix_user_starter_credit_grants_workspace_id",
        "user_starter_credit_grants",
        ["workspace_id"],
        unique=True,
    )

    op.create_table(
        "production_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("revision_version", sa.Integer(), nullable=True),
        sa.Column("dedupe_key", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dispatching_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dedupe_key"),
    )
    for column in (
        "available_at",
        "celery_task_id",
        "created_by_user_id",
        "dedupe_key",
        "kind",
        "project_id",
        "status",
    ):
        op.create_index(
            f"ix_production_jobs_{column}",
            "production_jobs",
            [column],
            unique=column == "dedupe_key",
        )
    op.create_index(
        "uq_active_revision_job",
        "production_jobs",
        ["project_id"],
        unique=True,
        postgresql_where=sa.text(
            "kind = 'revision' AND status IN ('queued','dispatching','dispatched','running','stalled')"
        ),
        sqlite_where=sa.text(
            "kind = 'revision' AND status IN ('queued','dispatching','dispatched','running','stalled')"
        ),
    )

    op.create_table(
        "workspace_deletion_tombstones",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("requested_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("workspace_slug", sa.String(length=120), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    op.create_index("ix_workspace_deletion_tombstones_request_id", "workspace_deletion_tombstones", ["request_id"], unique=True)
    op.create_index("ix_workspace_deletion_tombstones_workspace_id", "workspace_deletion_tombstones", ["workspace_id"], unique=False)

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("workspace_invitations") as batch:
            batch.drop_constraint("uq_open_workspace_invite", type_="unique")
    else:
        op.drop_constraint("uq_open_workspace_invite", "workspace_invitations", type_="unique")
    op.create_index(
        "uq_open_workspace_invite",
        "workspace_invitations",
        ["workspace_id", "email"],
        unique=True,
        postgresql_where=sa.text("accepted_at IS NULL AND revoked_at IS NULL"),
        sqlite_where=sa.text("accepted_at IS NULL AND revoked_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_open_workspace_invite", table_name="workspace_invitations")
    op.create_unique_constraint(
        "uq_open_workspace_invite",
        "workspace_invitations",
        ["workspace_id", "email", "accepted_at"],
    )
    op.drop_index("ix_workspace_deletion_tombstones_workspace_id", table_name="workspace_deletion_tombstones")
    op.drop_index("ix_workspace_deletion_tombstones_request_id", table_name="workspace_deletion_tombstones")
    op.drop_table("workspace_deletion_tombstones")
    op.drop_index("uq_active_revision_job", table_name="production_jobs")
    for column in (
        "status",
        "project_id",
        "kind",
        "dedupe_key",
        "created_by_user_id",
        "celery_task_id",
        "available_at",
    ):
        op.drop_index(f"ix_production_jobs_{column}", table_name="production_jobs")
    op.drop_table("production_jobs")
    op.drop_index("ix_user_starter_credit_grants_workspace_id", table_name="user_starter_credit_grants")
    op.drop_table("user_starter_credit_grants")
    op.drop_column("projects", "revision_generation")
    op.drop_column("projects", "run_generation")
    op.drop_column("auth_sessions", "csrf_token_hash")
