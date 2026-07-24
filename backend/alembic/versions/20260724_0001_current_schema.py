"""Create the immutable pre-platform Director OS schema.

Revision ID: 20260724_0001
Revises: None
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260724_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('director_memory_profiles',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('profile_key', sa.String(length=100), nullable=False),
    sa.Column('preferences', sa.JSON(), nullable=False),
    sa.Column('negative_preferences', sa.JSON(), nullable=False),
    sa.Column('evidence_count', sa.Integer(), nullable=False),
    sa.Column('performance_sample_count', sa.Integer(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'profile_key', name='uq_director_memory_user_profile')
    )
    op.create_index(op.f('ix_director_memory_profiles_profile_key'), 'director_memory_profiles', ['profile_key'], unique=False)
    op.create_index(op.f('ix_director_memory_profiles_user_id'), 'director_memory_profiles', ['user_id'], unique=False)
    op.create_table('projects',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('user_id', sa.Uuid(), nullable=False),
    sa.Column('status', sa.Enum('CREATED', 'UPLOADING', 'READY_TO_QUEUE', 'QUEUED', 'ANALYZING', 'NEEDS_PICKUPS', 'PLANNING', 'RENDERING', 'QUALITY_CHECK', 'READY', 'FAILED', name='projectstatus', native_enum=False, length=32), nullable=False),
    sa.Column('contract', sa.JSON(), nullable=False),
    sa.Column('task_id', sa.String(length=255), nullable=True),
    sa.Column('output_path', sa.Text(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_projects_status'), 'projects', ['status'], unique=False)
    op.create_index(op.f('ix_projects_task_id'), 'projects', ['task_id'], unique=False)
    op.create_index(op.f('ix_projects_user_id'), 'projects', ['user_id'], unique=False)
    op.create_table('director_camera_audits',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('mode', sa.String(length=24), nullable=False),
    sa.Column('readiness_score', sa.Float(), nullable=False),
    sa.Column('threshold', sa.Float(), nullable=False),
    sa.Column('ready', sa.Boolean(), nullable=False),
    sa.Column('report', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('project_id', 'version', name='uq_director_camera_project_version')
    )
    op.create_index(op.f('ix_director_camera_audits_project_id'), 'director_camera_audits', ['project_id'], unique=False)
    op.create_table('director_memory_evidence',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('profile_id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=True),
    sa.Column('revision_version', sa.Integer(), nullable=True),
    sa.Column('event_type', sa.String(length=40), nullable=False),
    sa.Column('verdict', sa.String(length=32), nullable=True),
    sa.Column('weight', sa.Float(), nullable=False),
    sa.Column('feedback_text', sa.Text(), nullable=True),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['profile_id'], ['director_memory_profiles.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_director_memory_evidence_event_type'), 'director_memory_evidence', ['event_type'], unique=False)
    op.create_index(op.f('ix_director_memory_evidence_profile_id'), 'director_memory_evidence', ['profile_id'], unique=False)
    op.create_index(op.f('ix_director_memory_evidence_project_id'), 'director_memory_evidence', ['project_id'], unique=False)
    op.create_table('edit_decision_graphs',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_edit_decision_graphs_project_id'), 'edit_decision_graphs', ['project_id'], unique=True)
    op.create_table('edit_graph_revisions',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('base_version', sa.Integer(), nullable=True),
    sa.Column('instruction', sa.Text(), nullable=True),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('task_id', sa.String(length=255), nullable=True),
    sa.Column('graph_payload', sa.JSON(), nullable=False),
    sa.Column('render_plan', sa.JSON(), nullable=False),
    sa.Column('critic_report', sa.JSON(), nullable=False),
    sa.Column('locked_ranges', sa.JSON(), nullable=False),
    sa.Column('output_path', sa.Text(), nullable=True),
    sa.Column('narration_cache_path', sa.Text(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('project_id', 'version', name='uq_edit_graph_revision_project_version')
    )
    op.create_index(op.f('ix_edit_graph_revisions_is_active'), 'edit_graph_revisions', ['is_active'], unique=False)
    op.create_index(op.f('ix_edit_graph_revisions_project_id'), 'edit_graph_revisions', ['project_id'], unique=False)
    op.create_index(op.f('ix_edit_graph_revisions_status'), 'edit_graph_revisions', ['status'], unique=False)
    op.create_index(op.f('ix_edit_graph_revisions_task_id'), 'edit_graph_revisions', ['task_id'], unique=False)
    op.create_index(op.f('ix_edit_graph_revisions_version'), 'edit_graph_revisions', ['version'], unique=False)
    op.create_table('project_analyses',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_project_analyses_project_id'), 'project_analyses', ['project_id'], unique=True)
    op.create_table('project_assets',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('kind', sa.Enum('SOURCE_VIDEO', 'PICKUP_VIDEO', 'REFERENCE_VIDEO', 'LOGO', 'BRAND_ASSET', 'MUSIC', name='assetkind', native_enum=False, length=32), nullable=False),
    sa.Column('original_filename', sa.String(length=512), nullable=False),
    sa.Column('stored_filename', sa.String(length=128), nullable=False),
    sa.Column('content_type', sa.String(length=255), nullable=False),
    sa.Column('size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('storage_path', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('stored_filename')
    )
    op.create_index(op.f('ix_project_assets_project_id'), 'project_assets', ['project_id'], unique=False)
    op.create_index(op.f('ix_project_assets_sha256'), 'project_assets', ['sha256'], unique=False)
    op.create_table('project_performance_signals',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('profile_id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('revision_version', sa.Integer(), nullable=True),
    sa.Column('platform', sa.String(length=80), nullable=False),
    sa.Column('normalized_score', sa.Float(), nullable=False),
    sa.Column('metrics', sa.JSON(), nullable=False),
    sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['profile_id'], ['director_memory_profiles.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_project_performance_signals_platform'), 'project_performance_signals', ['platform'], unique=False)
    op.create_index(op.f('ix_project_performance_signals_profile_id'), 'project_performance_signals', ['profile_id'], unique=False)
    op.create_index(op.f('ix_project_performance_signals_project_id'), 'project_performance_signals', ['project_id'], unique=False)
    op.create_table('pickup_missions',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('project_id', sa.Uuid(), nullable=False),
    sa.Column('audit_id', sa.Uuid(), nullable=False),
    sa.Column('mission_type', sa.String(length=40), nullable=False),
    sa.Column('priority', sa.String(length=24), nullable=False),
    sa.Column('title', sa.String(length=240), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('specification', sa.JSON(), nullable=False),
    sa.Column('target_terms', sa.JSON(), nullable=False),
    sa.Column('submitted_asset_id', sa.Uuid(), nullable=True),
    sa.Column('accepted_asset_id', sa.Uuid(), nullable=True),
    sa.Column('validation', sa.JSON(), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['accepted_asset_id'], ['project_assets.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['audit_id'], ['director_camera_audits.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['submitted_asset_id'], ['project_assets.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pickup_missions_accepted_asset_id'), 'pickup_missions', ['accepted_asset_id'], unique=False)
    op.create_index(op.f('ix_pickup_missions_audit_id'), 'pickup_missions', ['audit_id'], unique=False)
    op.create_index(op.f('ix_pickup_missions_mission_type'), 'pickup_missions', ['mission_type'], unique=False)
    op.create_index(op.f('ix_pickup_missions_priority'), 'pickup_missions', ['priority'], unique=False)
    op.create_index(op.f('ix_pickup_missions_project_id'), 'pickup_missions', ['project_id'], unique=False)
    op.create_index(op.f('ix_pickup_missions_status'), 'pickup_missions', ['status'], unique=False)
    op.create_index(op.f('ix_pickup_missions_submitted_asset_id'), 'pickup_missions', ['submitted_asset_id'], unique=False)


def downgrade() -> None:
    op.drop_table("pickup_missions")
    op.drop_table("project_performance_signals")
    op.drop_table("project_assets")
    op.drop_table("project_analyses")
    op.drop_table("edit_graph_revisions")
    op.drop_table("edit_decision_graphs")
    op.drop_table("director_memory_evidence")
    op.drop_table("director_camera_audits")
    op.drop_table("projects")
    op.drop_table("director_memory_profiles")
