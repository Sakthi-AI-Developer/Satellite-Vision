"""Persist daily satellite monitoring state and scenes.

Revision ID: 0005_satellite_monitoring
Revises: 0004_analysis_history_index
"""
from alembic import op
import sqlalchemy as sa

revision = "0005_satellite_monitoring"
down_revision = "0004_analysis_history_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "satellite_monitor_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("timezone_name", sa.String(length=80), nullable=False),
        sa.Column("schedule_time", sa.String(length=5), nullable=False),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("last_successful_scan_at", sa.DateTime(timezone=True)),
        sa.Column("next_scheduled_run_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.Text()),
        sa.Column("lease_owner", sa.String(length=128)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "monitored_satellite_scenes",
        sa.Column("scene_id", sa.String(length=512), primary_key=True),
        sa.Column("collection", sa.String(length=80), nullable=False),
        sa.Column("source", sa.String(length=120), nullable=False),
        sa.Column("observation_at", sa.DateTime(timezone=True)),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cloud_cover", sa.Float()),
        sa.Column("bbox", sa.JSON()),
        sa.Column("assets", sa.JSON(), nullable=False),
        sa.Column("acquisition_supported", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("next_retry_at", sa.DateTime(timezone=True)),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.Text()),
        sa.Column("storage_key", sa.String(length=512)),
        sa.Column("sha256", sa.String(length=64)),
        sa.Column("downloaded_bytes", sa.Integer()),
        sa.Column("artifact_bytes", sa.Integer(), nullable=False),
        sa.Column("validated_at", sa.DateTime(timezone=True)),
        sa.Column("model_name", sa.String(length=160)),
        sa.Column("model_version", sa.String(length=80)),
        sa.Column("model_sha256", sa.String(length=64)),
        sa.Column("analysis_status", sa.String(length=24)),
        sa.Column("processing_ms", sa.Integer()),
        sa.Column("classification_key", sa.String(length=512)),
        sa.Column("classification_sha256", sa.String(length=64)),
        sa.Column("confidence_key", sa.String(length=512)),
        sa.Column("confidence_sha256", sa.String(length=64)),
        sa.Column("class_distribution", sa.JSON()),
        sa.Column("processing_completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_monitor_scenes_observation",
        "monitored_satellite_scenes",
        ["observation_at"],
    )
    op.create_index(
        "ix_monitor_scenes_status_discovered",
        "monitored_satellite_scenes",
        ["status", "discovered_at"],
    )


def downgrade() -> None:
    raise RuntimeError(
        "This migration intentionally has no destructive downgrade. "
        "Back up and migrate production data explicitly before schema removal."
    )
