"""Add immutable platform, sessions, teams, storage, audit, and billing schema.

Revision ID: 20260724_0002
Revises: 20260724_0001
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260724_0002"
down_revision = "20260724_0001"
branch_labels = None
depends_on = None


def _add_workspace_ownership() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("projects")}
    if "workspace_id" in columns:
        return
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("projects") as batch:
            batch.add_column(sa.Column("workspace_id", sa.Uuid(), nullable=True))
            batch.create_foreign_key(
                "fk_projects_workspace_id_workspaces", "workspaces", ["workspace_id"], ["id"], ondelete="CASCADE"
            )
            batch.create_index("ix_projects_workspace_id", ["workspace_id"])
    else:
        op.add_column("projects", sa.Column("workspace_id", sa.Uuid(), nullable=True))
        op.create_foreign_key(
            "fk_projects_workspace_id_workspaces", "projects", "workspaces", ["workspace_id"], ["id"], ondelete="CASCADE"
        )
        op.create_index("ix_projects_workspace_id", "projects", ["workspace_id"])


def upgrade() -> None:
    op.create_table('email_outbox',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('recipient', sa.String(length=320), nullable=False),
    sa.Column('subject', sa.String(length=300), nullable=False),
    sa.Column('body_text', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_email_outbox_recipient'), 'email_outbox', ['recipient'], unique=False)
    op.create_index(op.f('ix_email_outbox_status'), 'email_outbox', ['status'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('display_name', sa.String(length=160), nullable=False),
    sa.Column('password_hash', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_table('account_tokens',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('purpose', sa.String(length=32), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('consumed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_account_tokens_expires_at'), 'account_tokens', ['expires_at'], unique=False)
    op.create_index(op.f('ix_account_tokens_purpose'), 'account_tokens', ['purpose'], unique=False)
    op.create_index(op.f('ix_account_tokens_token_hash'), 'account_tokens', ['token_hash'], unique=True)
    op.create_index(op.f('ix_account_tokens_user_id'), 'account_tokens', ['user_id'], unique=False)
    op.create_table('auth_sessions',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('family_id', sa.Uuid(), nullable=False),
    sa.Column('refresh_token_hash', sa.String(length=64), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('rotated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('replaced_by_session_id', sa.Uuid(), nullable=True),
    sa.Column('user_agent', sa.String(length=500), nullable=True),
    sa.Column('ip_address', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['replaced_by_session_id'], ['auth_sessions.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_auth_sessions_expires_at'), 'auth_sessions', ['expires_at'], unique=False)
    op.create_index(op.f('ix_auth_sessions_family_id'), 'auth_sessions', ['family_id'], unique=False)
    op.create_index(op.f('ix_auth_sessions_refresh_token_hash'), 'auth_sessions', ['refresh_token_hash'], unique=True)
    op.create_index(op.f('ix_auth_sessions_user_id'), 'auth_sessions', ['user_id'], unique=False)
    op.create_table('user_email_statuses',
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('verification_sent_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id')
    )
    op.create_table('workspaces',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=180), nullable=False),
    sa.Column('slug', sa.String(length=120), nullable=False),
    sa.Column('created_by_user_id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workspaces_created_by_user_id'), 'workspaces', ['created_by_user_id'], unique=False)
    op.create_index(op.f('ix_workspaces_slug'), 'workspaces', ['slug'], unique=True)
    op.create_table('audit_events',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('workspace_id', sa.Uuid(), nullable=True),
    sa.Column('actor_user_id', sa.Uuid(), nullable=True),
    sa.Column('action', sa.String(length=120), nullable=False),
    sa.Column('resource_type', sa.String(length=80), nullable=False),
    sa.Column('resource_id', sa.String(length=120), nullable=True),
    sa.Column('request_id', sa.String(length=80), nullable=True),
    sa.Column('ip_address', sa.String(length=64), nullable=True),
    sa.Column('user_agent', sa.String(length=500), nullable=True),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['actor_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_events_action'), 'audit_events', ['action'], unique=False)
    op.create_index(op.f('ix_audit_events_actor_user_id'), 'audit_events', ['actor_user_id'], unique=False)
    op.create_index(op.f('ix_audit_events_created_at'), 'audit_events', ['created_at'], unique=False)
    op.create_index(op.f('ix_audit_events_request_id'), 'audit_events', ['request_id'], unique=False)
    op.create_index(op.f('ix_audit_events_resource_id'), 'audit_events', ['resource_id'], unique=False)
    op.create_index(op.f('ix_audit_events_resource_type'), 'audit_events', ['resource_type'], unique=False)
    op.create_index(op.f('ix_audit_events_workspace_id'), 'audit_events', ['workspace_id'], unique=False)
    op.create_table('billing_accounts',
    sa.Column('workspace_id', sa.Uuid(), nullable=False),
    sa.Column('balance_credits', sa.Numeric(precision=14, scale=4), nullable=False),
    sa.Column('reserved_credits', sa.Numeric(precision=14, scale=4), nullable=False),
    sa.Column('plan', sa.String(length=40), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('workspace_id')
    )
    op.create_table('workspace_invitations',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('workspace_id', sa.Uuid(), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('role', sa.String(length=24), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('invited_by_user_id', sa.Uuid(), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('accepted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['invited_by_user_id'], ['users.id'], ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('workspace_id', 'email', 'accepted_at', name='uq_open_workspace_invite')
    )
    op.create_index(op.f('ix_workspace_invitations_email'), 'workspace_invitations', ['email'], unique=False)
    op.create_index(op.f('ix_workspace_invitations_expires_at'), 'workspace_invitations', ['expires_at'], unique=False)
    op.create_index(op.f('ix_workspace_invitations_invited_by_user_id'), 'workspace_invitations', ['invited_by_user_id'], unique=False)
    op.create_index(op.f('ix_workspace_invitations_token_hash'), 'workspace_invitations', ['token_hash'], unique=True)
    op.create_index(op.f('ix_workspace_invitations_workspace_id'), 'workspace_invitations', ['workspace_id'], unique=False)
    op.create_table('workspace_memberships',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('workspace_id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('role', sa.String(length=24), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('workspace_id', 'user_id', name='uq_workspace_membership')
    )
    op.create_index(op.f('ix_workspace_memberships_user_id'), 'workspace_memberships', ['user_id'], unique=False)
    op.create_index(op.f('ix_workspace_memberships_workspace_id'), 'workspace_memberships', ['workspace_id'], unique=False)
    op.create_table('billing_entries',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('workspace_id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=True),
    sa.Column('actor_user_id', sa.Uuid(), nullable=True),
    sa.Column('kind', sa.String(length=32), nullable=False),
    sa.Column('amount_credits', sa.Numeric(precision=14, scale=4), nullable=False),
    sa.Column('idempotency_key', sa.String(length=180), nullable=False),
    sa.Column('description', sa.String(length=500), nullable=False),
    sa.Column('entry_metadata', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['actor_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_billing_entries_created_at'), 'billing_entries', ['created_at'], unique=False)
    op.create_index(op.f('ix_billing_entries_idempotency_key'), 'billing_entries', ['idempotency_key'], unique=True)
    op.create_index(op.f('ix_billing_entries_kind'), 'billing_entries', ['kind'], unique=False)
    op.create_index(op.f('ix_billing_entries_project_id'), 'billing_entries', ['project_id'], unique=False)
    op.create_index(op.f('ix_billing_entries_workspace_id'), 'billing_entries', ['workspace_id'], unique=False)
    op.create_table('project_billing_reservations',
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('workspace_id', sa.Uuid(), nullable=False),
    sa.Column('reserved_credits', sa.Numeric(precision=14, scale=4), nullable=False),
    sa.Column('settled_credits', sa.Numeric(precision=14, scale=4), nullable=True),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('project_id')
    )
    op.create_index(op.f('ix_project_billing_reservations_status'), 'project_billing_reservations', ['status'], unique=False)
    op.create_index(op.f('ix_project_billing_reservations_workspace_id'), 'project_billing_reservations', ['workspace_id'], unique=False)
    op.create_table('multipart_uploads',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('created_by_user_id', sa.Uuid(), nullable=False),
    sa.Column('asset_id', sa.Uuid(), nullable=True),
    sa.Column('provider', sa.String(length=24), nullable=False),
    sa.Column('provider_upload_id', sa.String(length=500), nullable=True),
    sa.Column('object_key', sa.String(length=1024), nullable=False),
    sa.Column('kind', sa.String(length=32), nullable=False),
    sa.Column('original_filename', sa.String(length=512), nullable=False),
    sa.Column('content_type', sa.String(length=255), nullable=False),
    sa.Column('total_bytes', sa.BigInteger(), nullable=False),
    sa.Column('part_size', sa.BigInteger(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['asset_id'], ['project_assets.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('object_key')
    )
    op.create_index(op.f('ix_multipart_uploads_asset_id'), 'multipart_uploads', ['asset_id'], unique=False)
    op.create_index(op.f('ix_multipart_uploads_created_by_user_id'), 'multipart_uploads', ['created_by_user_id'], unique=False)
    op.create_index(op.f('ix_multipart_uploads_expires_at'), 'multipart_uploads', ['expires_at'], unique=False)
    op.create_index(op.f('ix_multipart_uploads_project_id'), 'multipart_uploads', ['project_id'], unique=False)
    op.create_index(op.f('ix_multipart_uploads_status'), 'multipart_uploads', ['status'], unique=False)
    op.create_table('resumable_uploads',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('created_by_user_id', sa.Uuid(), nullable=False),
    sa.Column('asset_id', sa.Uuid(), nullable=True),
    sa.Column('kind', sa.String(length=32), nullable=False),
    sa.Column('original_filename', sa.String(length=512), nullable=False),
    sa.Column('content_type', sa.String(length=255), nullable=False),
    sa.Column('total_bytes', sa.BigInteger(), nullable=False),
    sa.Column('received_bytes', sa.BigInteger(), nullable=False),
    sa.Column('storage_path', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['asset_id'], ['project_assets.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_resumable_uploads_asset_id'), 'resumable_uploads', ['asset_id'], unique=False)
    op.create_index(op.f('ix_resumable_uploads_created_by_user_id'), 'resumable_uploads', ['created_by_user_id'], unique=False)
    op.create_index(op.f('ix_resumable_uploads_project_id'), 'resumable_uploads', ['project_id'], unique=False)
    op.create_index(op.f('ix_resumable_uploads_status'), 'resumable_uploads', ['status'], unique=False)
    op.create_table('stored_objects',
    sa.Column('asset_id', sa.Uuid(), nullable=False),
    sa.Column('provider', sa.String(length=24), nullable=False),
    sa.Column('bucket', sa.String(length=255), nullable=True),
    sa.Column('object_key', sa.String(length=1024), nullable=False),
    sa.Column('local_cache_path', sa.Text(), nullable=True),
    sa.Column('verified', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['asset_id'], ['project_assets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('asset_id')
    )
    op.create_index(op.f('ix_stored_objects_object_key'), 'stored_objects', ['object_key'], unique=False)
    op.create_table('multipart_upload_parts',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('upload_id', sa.Uuid(), nullable=False),
    sa.Column('part_number', sa.Integer(), nullable=False),
    sa.Column('etag', sa.String(length=300), nullable=False),
    sa.Column('size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['upload_id'], ['multipart_uploads.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('upload_id', 'part_number', name='uq_multipart_upload_part')
    )
    op.create_index(op.f('ix_multipart_upload_parts_upload_id'), 'multipart_upload_parts', ['upload_id'], unique=False)
    _add_workspace_ownership()


def downgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("projects")}
    if "workspace_id" in columns:
        if bind.dialect.name == "sqlite":
            with op.batch_alter_table("projects") as batch:
                batch.drop_index("ix_projects_workspace_id")
                batch.drop_constraint("fk_projects_workspace_id_workspaces", type_="foreignkey")
                batch.drop_column("workspace_id")
        else:
            op.drop_index("ix_projects_workspace_id", table_name="projects")
            op.drop_constraint("fk_projects_workspace_id_workspaces", "projects", type_="foreignkey")
            op.drop_column("projects", "workspace_id")
    op.drop_table("multipart_upload_parts")
    op.drop_table("stored_objects")
    op.drop_table("resumable_uploads")
    op.drop_table("multipart_uploads")
    op.drop_table("project_billing_reservations")
    op.drop_table("billing_entries")
    op.drop_table("workspace_memberships")
    op.drop_table("workspace_invitations")
    op.drop_table("billing_accounts")
    op.drop_table("audit_events")
    op.drop_table("workspaces")
    op.drop_table("user_email_statuses")
    op.drop_table("auth_sessions")
    op.drop_table("account_tokens")
    op.drop_table("users")
    op.drop_table("email_outbox")
