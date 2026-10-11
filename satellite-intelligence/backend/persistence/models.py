from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        Index(
            "ix_analyses_owner_status_created",
            "owner_id",
            "status",
            "created_at",
        ),
        Index(
            "uq_analyses_idempotency_scope_key",
            "idempotency_scope",
            "idempotency_key",
            unique=True,
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    analysis: Mapped[str] = mapped_column(String(80), nullable=False)
    input_parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    owner_id: Mapped[str | None] = mapped_column(String(128), index=True)
    idempotency_scope: Mapped[str | None] = mapped_column(String(128))
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    error: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    jobs: Mapped[list[ProcessingJob]] = relationship(back_populates="analysis_record")
    artifacts: Mapped[list[AnalysisArtifact]] = relationship(
        back_populates="analysis_record", cascade="all, delete-orphan"
    )


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    input_parameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    analysis_record: Mapped[AnalysisRecord] = relationship(back_populates="jobs")
    status_history: Mapped[list[JobStatusHistory]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobStatusHistory.id"
    )


class JobStatusHistory(Base):
    __tablename__ = "job_status_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    job: Mapped[ProcessingJob] = relationship(back_populates="status_history")


class AnalysisArtifact(Base):
    __tablename__ = "analysis_artifacts"
    __table_args__ = (
        UniqueConstraint("analysis_id", "artifact_name"),
        Index(
            "uq_analysis_artifacts_bucket_object_key",
            "bucket_name",
            "object_key",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_name: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    bucket_name: Mapped[str | None] = mapped_column(String(100))
    artifact_type: Mapped[str] = mapped_column(String(40), nullable=False, default="file")
    job_id: Mapped[str | None] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    analysis_record: Mapped[AnalysisRecord] = relationship(back_populates="artifacts")


class SatelliteMonitorState(Base):
    __tablename__ = "satellite_monitor_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="idle")
    timezone_name: Mapped[str] = mapped_column(
        String(80), nullable=False, default="Asia/Kolkata"
    )
    schedule_time: Mapped[str] = mapped_column(
        String(5), nullable=False, default="03:00"
    )
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_successful_scan_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    next_scheduled_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MonitoredSatelliteScene(Base):
    __tablename__ = "monitored_satellite_scenes"
    __table_args__ = (
        Index("ix_monitor_scenes_observation", "observation_at"),
        Index("ix_monitor_scenes_status_discovered", "status", "discovered_at"),
    )

    scene_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    collection: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[str] = mapped_column(String(120), nullable=False)
    observation_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    cloud_cover: Mapped[float | None] = mapped_column()
    bbox: Mapped[list[float] | None] = mapped_column(JSON)
    assets: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    acquisition_supported: Mapped[bool] = mapped_column(nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="discovered")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    storage_key: Mapped[str | None] = mapped_column(String(512))
    sha256: Mapped[str | None] = mapped_column(String(64))
    downloaded_bytes: Mapped[int | None] = mapped_column()
    artifact_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model_name: Mapped[str | None] = mapped_column(String(160))
    model_version: Mapped[str | None] = mapped_column(String(80))
    model_sha256: Mapped[str | None] = mapped_column(String(64))
    analysis_status: Mapped[str | None] = mapped_column(String(24))
    processing_ms: Mapped[int | None] = mapped_column()
    classification_key: Mapped[str | None] = mapped_column(String(512))
    classification_sha256: Mapped[str | None] = mapped_column(String(64))
    confidence_key: Mapped[str | None] = mapped_column(String(512))
    confidence_sha256: Mapped[str | None] = mapped_column(String(64))
    class_distribution: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
