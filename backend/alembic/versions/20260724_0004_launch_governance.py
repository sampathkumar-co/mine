"""Add privacy export and deletion requests.

Revision ID: 20260724_0004
Revises: 20260724_0003
"""
from __future__ import annotations

from alembic import op
from app.core.database import Base, import_models

revision = "20260724_0004"
down_revision = "20260724_0003"
branch_labels = None
depends_on = None

GOVERNANCE_TABLES = {"privacy_requests"}


def upgrade() -> None:
    import_models()
    bind = op.get_bind()
    for table in Base.metadata.sorted_tables:
        if table.name in GOVERNANCE_TABLES:
            table.create(bind, checkfirst=True)


def downgrade() -> None:
    import_models()
    bind = op.get_bind()
    for table in reversed(Base.metadata.sorted_tables):
        if table.name in GOVERNANCE_TABLES:
            table.drop(bind, checkfirst=True)
