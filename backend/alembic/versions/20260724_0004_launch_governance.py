"""Add immutable privacy export and deletion schema.

Revision ID: 20260724_0004
Revises: 20260724_0003
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260724_0004"
down_revision = "20260724_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('privacy_requests',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('workspace_id', sa.Uuid(), nullable=False),
    sa.Column('requested_by_user_id', sa.Uuid(), nullable=True),
    sa.Column('kind', sa.String(length=24), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('result_path', sa.Text(), nullable=True),
    sa.Column('result_sha256', sa.String(length=64), nullable=True),
    sa.Column('result_size_bytes', sa.BigInteger(), nullable=True),
    sa.Column('available_until', sa.DateTime(timezone=True), nullable=True),
    sa.Column('execute_after', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('request_metadata', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['requested_by_user_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['workspace_id'], ['workspaces.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_privacy_requests_kind'), 'privacy_requests', ['kind'], unique=False)
    op.create_index(op.f('ix_privacy_requests_requested_by_user_id'), 'privacy_requests', ['requested_by_user_id'], unique=False)
    op.create_index(op.f('ix_privacy_requests_status'), 'privacy_requests', ['status'], unique=False)
    op.create_index(op.f('ix_privacy_requests_workspace_id'), 'privacy_requests', ['workspace_id'], unique=False)


def downgrade() -> None:
    op.drop_table("privacy_requests")
