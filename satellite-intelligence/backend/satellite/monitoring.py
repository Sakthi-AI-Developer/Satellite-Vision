from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from threading import Event, Thread
import time as monotonic_time
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import rasterio
from rasterio.errors import RasterioError
from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from config import Settings
from geoai import inference as geoai_inference
from persistence.models import MonitoredSatelliteScene, SatelliteMonitorState, utc_now
from pyproj import Transformer
from shapely import transform as transform_geometry
from shapely.geometry import box, shape
from satellite import acquisition as satellite_acquisition
from satellite.catalog import (
    CatalogError,
    CatalogRateLimitError,
    CatalogTimeoutError,
    search_sentinel2_catalog,
)
from satellite.models import (
    GeoJSONPolygon,
    SentinelCatalogSearchRequest,
    SentinelSceneAcquisitionRequest,
)

logger = logging.getLogger("satellite-intelligence.monitoring")
SOURCE_NAME = "Copernicus Data Space Ecosystem"
COLLECTION = "sentinel-2-l2a"
INPUT_BANDS = ("B02", "B03", "B04", "B08")
LEASE_DURATION = timedelta(minutes=60)
STALE_SCAN_DURATION = timedelta(minutes=65)
LEASE_RENEW_SECONDS = 120
POLL_SECONDS = 20


class MonitoringValidationError(Exception):
    pass


def next_daily_run(
    now: datetime, timezone_name: str, schedule_time: str
) -> datetime:
    zone = ZoneInfo(timezone_name)
    local_now = now.astimezone(zone)
    hour, minute = (int(part) for part in schedule_time.split(":"))
    scheduled = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if scheduled <= local_now:
        scheduled += timedelta(days=1)
    return scheduled.astimezone(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _json_safe(value: object) -> object:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return value


def _safe_failure(exc: BaseException) -> str:
    if isinstance(exc, MonitoringValidationError):
        return str(exc)[:500]
    detail = getattr(exc, "detail", None)
    if isinstance(detail, str) and detail:
        return detail[:500]
    if isinstance(exc, (httpx.TimeoutException, CatalogTimeoutError)):
        return "The catalog or imagery source timed out."
    if isinstance(exc, (httpx.RequestError, CatalogError)):
        return "The catalog or imagery source could not be reached."
    if isinstance(exc, OSError):
        return "Local storage could not read or write the scene."
    if isinstance(exc, (RasterioError, ValueError)):
        return "The downloaded scene failed raster validation."
    return f"Processing failed ({type(exc).__name__})."


class MonitoringStore:
    def __init__(self, engine: Engine):
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    def ensure_state(self, timezone_name: str, schedule_time: str) -> None:
        try:
            with self._sessions.begin() as session:
                state = session.get(SatelliteMonitorState, 1)
                if state is None:
                    session.add(
                        SatelliteMonitorState(
                            id=1,
                            status="idle",
                            timezone_name=timezone_name,
                            schedule_time=schedule_time,
                            next_scheduled_run_at=next_daily_run(
                                utc_now(), timezone_name, schedule_time
                            ),
                        )
                    )
        except IntegrityError:
            pass

    def initialize_schedule(self, timezone_name: str, schedule_time: str) -> None:
        self.ensure_state(timezone_name, schedule_time)
        with self._sessions.begin() as session:
            state = session.get(SatelliteMonitorState, 1)
            if state and (
                state.timezone_name != timezone_name
                or state.schedule_time != schedule_time
            ):
                state.timezone_name = timezone_name
                state.schedule_time = schedule_time
                state.next_scheduled_run_at = next_daily_run(
                    utc_now(), timezone_name, schedule_time
                )
            elif state and state.next_scheduled_run_at is None:
                state.next_scheduled_run_at = next_daily_run(
                    utc_now(), timezone_name, schedule_time
                )
            last_attempt = _as_utc(state.last_attempt_at) if state else None
            lease_expires = _as_utc(state.lease_expires_at) if state else None
            if (
                state
                and state.status == "running"
                and last_attempt is not None
                and last_attempt < utc_now() - STALE_SCAN_DURATION
                and (lease_expires is None or lease_expires < utc_now())
            ):
                state.status = "failed"
                state.last_error = "A previous scan stopped before completion."
                state.lease_owner = None
                state.lease_expires_at = None
            stale_scenes = session.scalars(
                select(MonitoredSatelliteScene).where(
                    MonitoredSatelliteScene.status.in_(
                        ("downloading", "validating", "analyzing")
                    )
                )
            ).all()
            for scene in stale_scenes:
                last_attempt = _as_utc(scene.last_attempt_at)
                if last_attempt and last_attempt < utc_now() - STALE_SCAN_DURATION:
                    scene.status = "failed"
                    scene.last_error = "A previous processing attempt stopped early."
                    scene.next_retry_at = utc_now()

    def acquire_lease(
        self,
        owner: str,
        *,
        timezone_name: str = "Asia/Kolkata",
        schedule_time: str = "03:00",
    ) -> bool:
        now = utc_now()
        self.ensure_state(timezone_name, schedule_time)
        with self._sessions.begin() as session:
            result = session.execute(
                update(SatelliteMonitorState)
                .where(
                    SatelliteMonitorState.id == 1,
                    (
                        SatelliteMonitorState.lease_expires_at.is_(None)
                        | (SatelliteMonitorState.lease_expires_at < now)
                    ),
                )
                .values(
                    lease_owner=owner,
                    lease_expires_at=now + LEASE_DURATION,
                    status="running",
                    last_attempt_at=now,
                    last_error=None,
                )
                .returning(SatelliteMonitorState.id)
            )
            return result.scalar_one_or_none() is not None

    def finish_scan(
        self,
        owner: str,
        *,
        status: str,
        error: str | None,
        catalog_succeeded: bool,
        timezone_name: str,
        schedule_time: str,
    ) -> None:
        now = utc_now()
        with self._sessions.begin() as session:
            state = session.get(SatelliteMonitorState, 1)
            if state is None or state.lease_owner != owner:
                return
            state.status = status
            state.last_error = error
            if catalog_succeeded:
                state.last_successful_scan_at = now
            state.next_scheduled_run_at = next_daily_run(
                now, timezone_name, schedule_time
            )
            state.lease_owner = None
            state.lease_expires_at = None

    def renew_lease(self, owner: str) -> bool:
        with self._sessions.begin() as session:
            result = session.execute(
                update(SatelliteMonitorState)
                .where(
                    SatelliteMonitorState.id == 1,
                    SatelliteMonitorState.lease_owner == owner,
                )
                .values(lease_expires_at=utc_now() + LEASE_DURATION)
                .returning(SatelliteMonitorState.id)
            )
            return result.scalar_one_or_none() is not None

    def add_scene(self, values: dict[str, Any]) -> bool:
        try:
            with self._sessions.begin() as session:
                session.add(MonitoredSatelliteScene(**values))
                session.flush()
            return True
        except IntegrityError:
            return False

    def update_scene(self, scene_id: str, **values: Any) -> None:
        allowed = {
            "status",
            "attempt_count",
            "next_retry_at",
            "last_attempt_at",
            "last_error",
            "storage_key",
            "sha256",
            "downloaded_bytes",
            "artifact_bytes",
            "validated_at",
            "model_name",
            "model_version",
            "model_sha256",
            "analysis_status",
            "processing_ms",
            "classification_key",
            "classification_sha256",
            "confidence_key",
            "confidence_sha256",
            "class_distribution",
            "processing_completed_at",
        }
        if values.keys() - allowed:
            raise ValueError("Unsupported satellite-monitor scene update.")
        with self._sessions.begin() as session:
            scene = session.get(MonitoredSatelliteScene, scene_id)
            if scene is not None:
                for key, value in values.items():
                    setattr(scene, key, value)

    def scenes_ready_for_processing(
        self, now: datetime, limit: int, max_attempts: int
    ) -> list[str]:
        with self._sessions() as session:
            return list(
                session.scalars(
                    select(MonitoredSatelliteScene.scene_id)
                    .where(
                        MonitoredSatelliteScene.status.in_(
                            ("discovered", "failed", "model_not_ready")
                        ),
                        (
                            (MonitoredSatelliteScene.attempt_count < max_attempts)
                            | (MonitoredSatelliteScene.status == "model_not_ready")
                        ),
                        (
                            MonitoredSatelliteScene.next_retry_at.is_(None)
                            | (MonitoredSatelliteScene.next_retry_at <= now)
                        ),
                    )
                    .order_by(MonitoredSatelliteScene.observation_at.desc())
                    .limit(limit)
                ).all()
            )

    def scene(self, scene_id: str) -> dict[str, Any] | None:
        with self._sessions() as session:
            row = session.get(MonitoredSatelliteScene, scene_id)
            return self._scene_dict(row) if row else None

    def list_scenes(self, limit: int, offset: int) -> list[dict[str, Any]]:
        with self._sessions() as session:
            rows = session.scalars(
                select(MonitoredSatelliteScene)
                .order_by(MonitoredSatelliteScene.discovered_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
            return [self._scene_dict(row) for row in rows]

    @staticmethod
    def _scene_dict(row: MonitoredSatelliteScene) -> dict[str, Any]:
        return {
            "scene_id": row.scene_id,
            "collection": row.collection,
            "source": row.source,
            "observation_at": _iso(row.observation_at),
            "published_at": _iso(row.published_at),
            "discovered_at": _iso(row.discovered_at),
            "cloud_cover": row.cloud_cover,
            "bbox": row.bbox,
            "assets": row.assets,
            "acquisition_supported": row.acquisition_supported,
            "status": row.status,
            "attempt_count": row.attempt_count,
            "next_retry_at": _iso(row.next_retry_at),
            "last_attempt_at": _iso(row.last_attempt_at),
            "last_error": row.last_error,
            "sha256": row.sha256,
            "downloaded_bytes": row.downloaded_bytes,
            "validated_at": _iso(row.validated_at),
            "analysis_status": row.analysis_status,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "model_sha256": row.model_sha256,
            "processing_ms": row.processing_ms,
            "classification_sha256": row.classification_sha256,
            "confidence_sha256": row.confidence_sha256,
            "class_distribution": row.class_distribution,
            "processing_completed_at": _iso(row.processing_completed_at),
            "classification_url": (
                f"/api/monitoring/scenes/{row.scene_id}/artifacts/classification.tif"
                if row.classification_key
                else None
            ),
            "confidence_url": (
                f"/api/monitoring/scenes/{row.scene_id}/artifacts/confidence.tif"
                if row.confidence_key
                else None
            ),
        }

    def artifact_key(self, scene_id: str, artifact: str) -> str | None:
        key_name = {
            "classification.tif": "classification_key",
            "confidence.tif": "confidence_key",
        }.get(artifact)
        if key_name is None:
            return None
        with self._sessions() as session:
            row = session.get(MonitoredSatelliteScene, scene_id)
            return getattr(row, key_name, None) if row else None

    def status(self, timezone_name: str, schedule_time: str) -> dict[str, Any]:
        self.ensure_state(timezone_name, schedule_time)
        with self._sessions() as session:
            state = session.get(SatelliteMonitorState, 1)
            counts = dict(
                session.execute(
                    select(
                        MonitoredSatelliteScene.status,
                        func.count(MonitoredSatelliteScene.scene_id),
                    ).group_by(MonitoredSatelliteScene.status)
                ).all()
            )
            downloaded_count = session.scalar(
                select(func.count(MonitoredSatelliteScene.scene_id)).where(
                    MonitoredSatelliteScene.downloaded_bytes.is_not(None)
                )
            )
            total_scenes = session.scalar(
                select(func.count(MonitoredSatelliteScene.scene_id))
            )
            latest_observation = session.scalar(
                select(func.max(MonitoredSatelliteScene.observation_at))
            )
        return {
            "status": state.status if state else "unavailable",
            "last_attempt_at": _iso(state.last_attempt_at) if state else None,
            "last_successful_scan_at": (
                _iso(state.last_successful_scan_at) if state else None
            ),
            "next_scheduled_run_at": (
                _iso(state.next_scheduled_run_at) if state else None
            ),
            "last_error": state.last_error if state else None,
            "lease_expires_at": _iso(state.lease_expires_at) if state else None,
            "latest_observation_at": _iso(latest_observation),
            "timezone": timezone_name,
            "schedule_time": schedule_time,
            "scene_counts": {
                "discovered": total_scenes or 0,
                "downloaded": downloaded_count or 0,
                "rejected": counts.get("rejected", 0),
                "analyzed": counts.get("analyzed", 0),
                "failed": counts.get("failed", 0),
            },
        }

    def stored_bytes(self) -> int:
        with self._sessions() as session:
            return int(
                session.scalar(
                    select(
                        func.coalesce(
                            func.sum(MonitoredSatelliteScene.downloaded_bytes), 0
                        )
                        + func.coalesce(
                            func.sum(MonitoredSatelliteScene.artifact_bytes), 0
                        )
                    )
                )
                    or 0
                )


def _iso(value: datetime | None) -> str | None:
    normalized = _as_utc(value)
    return normalized.isoformat() if normalized else None


class MonitoringPipeline:
    def __init__(
        self,
        configuration: Settings,
        engine: Engine,
        *,
        catalog_search: Callable = search_sentinel2_catalog,
        acquire_scene: Callable = satellite_acquisition.acquire_scene,
        model_listing: Callable = geoai_inference.list_models,
        inference: Callable = geoai_inference.run_landcover_inference,
    ):
        self.settings = configuration
        self.store = MonitoringStore(engine)
        self.catalog_search = catalog_search
        self._include_asset_urls = catalog_search is search_sentinel2_catalog
        self.acquire_scene = acquire_scene
        self.model_listing = model_listing
        self.inference = inference
        self.owner = f"{os.getpid()}-{uuid4().hex}"
        self.aoi = self._load_aoi(configuration.satellite_monitor_aoi_json)

    @staticmethod
    def _load_aoi(value: str | None) -> GeoJSONPolygon | None:
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict) and parsed.get("type") == "Feature":
                parsed = parsed.get("geometry")
            return GeoJSONPolygon.model_validate(parsed)
        except (ValueError, TypeError):
            return None

    @property
    def configured(self) -> bool:
        return bool(
            self._schedule_valid()
            and self.settings.satellite_monitor_enabled
            and self.settings.database_url
            and self.settings.cdse_username
            and self.settings.cdse_password
            and self.aoi is not None
            and self.aoi.estimate_area_km2()
            <= self.settings.max_satellite_acquisition_area_km2
        )

    def _schedule_valid(self) -> bool:
        if not re.fullmatch(
            r"(?:[01]\d|2[0-3]):[0-5]\d",
            self.settings.satellite_monitor_schedule_time,
        ):
            return False
        try:
            ZoneInfo(self.settings.satellite_monitor_timezone)
        except (ValueError, KeyError):
            return False
        return True

    def initialize(self) -> None:
        self.store.initialize_schedule(
            self.settings.satellite_monitor_timezone,
            self.settings.satellite_monitor_schedule_time,
        )

    def status(self) -> dict[str, Any]:
        result = self.store.status(
            self.settings.satellite_monitor_timezone,
            self.settings.satellite_monitor_schedule_time,
        )
        result.update(
            {
                "enabled": self.configured,
                "model": self._model_status(),
                "aoi_configured": self.aoi is not None,
                "data_source": SOURCE_NAME,
                "collection": COLLECTION,
            }
        )
        if not self.configured:
            result["status"] = "disabled"
            result["disabled_reason"] = self._disabled_reason()
        return result

    def _disabled_reason(self) -> str:
        if not self.settings.satellite_monitor_enabled:
            return "Monitoring is disabled by configuration."
        if not self._schedule_valid():
            return "The configured monitoring timezone or daily schedule is invalid."
        if self.aoi is None:
            return "A valid SATELLITE_MONITOR_AOI is required."
        if (
            self.aoi.estimate_area_km2()
            > self.settings.max_satellite_acquisition_area_km2
        ):
            return "The monitoring AOI exceeds MAX_SATELLITE_ACQUISITION_AREA_KM2."
        if not self.settings.database_url:
            return "DATABASE_URL is required for persistent monitoring."
        return "CDSE_USERNAME and CDSE_PASSWORD are required for scene acquisition."

    def _model_status(self) -> dict[str, Any]:
        model = self.model_listing(self.settings)["models"][0]
        return {
            "status": model["status"],
            "name": model.get("name") if model["status"] == "ready" else None,
            "version": model.get("version"),
        }

    def claim_scan(self) -> bool:
        if not self.configured:
            return False
        return self.store.acquire_lease(
            self.owner,
            timezone_name=self.settings.satellite_monitor_timezone,
            schedule_time=self.settings.satellite_monitor_schedule_time,
        )

    def run_scan(self, *, claimed: bool = False) -> None:
        if not claimed and not self.claim_scan():
            return
        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._renew_lease,
            args=(heartbeat_stop,),
            name="satellite-monitor-lease",
            daemon=True,
        )
        heartbeat.start()
        catalog_succeeded = False
        scan_error: str | None = None
        scan_status = "completed"
        try:
            self.initialize()
            discovered = self._discover()
            catalog_succeeded = True
            for scene_id in discovered:
                scene = self.store.scene(scene_id)
                if scene and not scene["acquisition_supported"]:
                    self.store.update_scene(
                        scene_id,
                        status="rejected",
                        last_error="Required 10 m Sentinel-2 assets are unavailable.",
                    )
            for scene_id in self.store.scenes_ready_for_processing(
                utc_now(),
                self.settings.satellite_monitor_max_scenes,
                self.settings.satellite_monitor_max_retries + 1,
            ):
                self._process_scene(scene_id)
            if self.store.status(
                self.settings.satellite_monitor_timezone,
                self.settings.satellite_monitor_schedule_time,
            )["scene_counts"]["failed"]:
                scan_status = "partial"
        except Exception as exc:
            scan_status = "failed"
            scan_error = _safe_failure(exc)
            logger.error(
                "Satellite monitoring scan failed.",
                extra={
                    "event": "monitor.scan_failed",
                    "error_type": type(exc).__name__,
                },
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=5)
            try:
                self.store.finish_scan(
                    self.owner,
                    status=scan_status,
                    error=scan_error,
                    catalog_succeeded=catalog_succeeded,
                    timezone_name=self.settings.satellite_monitor_timezone,
                    schedule_time=self.settings.satellite_monitor_schedule_time,
                )
            except SQLAlchemyError:
                logger.exception(
                    "Satellite monitoring scan state could not be finalized.",
                    extra={"event": "monitor.scan_finalize_failed"},
                )

    def _renew_lease(self, stop: Event) -> None:
        while not stop.wait(LEASE_RENEW_SECONDS):
            try:
                if not self.store.renew_lease(self.owner):
                    logger.error(
                        "Satellite monitoring lease ownership was lost.",
                        extra={"event": "monitor.lease_lost"},
                    )
                    return
            except SQLAlchemyError:
                logger.exception(
                    "Satellite monitoring lease renewal failed.",
                    extra={"event": "monitor.lease_renewal_failed"},
                )

    def _discover(self) -> list[str]:
        if self.aoi is None:
            raise ValueError("A valid monitoring area of interest is not configured.")
        today = utc_now().date()
        first_day = today - timedelta(days=self.settings.satellite_monitor_lookback_days - 1)
        cursor = None
        created: list[str] = []
        new_scene_count = 0
        page_count = 0
        seen_cursors: set[str] = set()
        max_attempts = self.settings.satellite_monitor_max_retries + 1
        while (
            new_scene_count < self.settings.satellite_monitor_max_scenes
            and page_count < 10
        ):
            request = SentinelCatalogSearchRequest(
                aoi=self.aoi,
                start_date=first_day,
                end_date=today,
                max_cloud_cover=self.settings.satellite_monitor_max_cloud_cover,
                limit=50,
                cursor=cursor,
            )
            page_count += 1
            for attempt in range(max_attempts):
                try:
                    response = (
                        self.catalog_search(request, include_asset_urls=True)
                        if self._include_asset_urls
                        else self.catalog_search(request)
                    )
                    break
                except CatalogRateLimitError:
                    if attempt + 1 >= max_attempts:
                        raise
                    monotonic_time.sleep(min(2**attempt, 8))
                except (CatalogTimeoutError, httpx.TimeoutException, CatalogError, httpx.RequestError):
                    if attempt + 1 >= max_attempts:
                        raise
                    monotonic_time.sleep(min(2**attempt, 8))
            else:
                raise CatalogError
            results = response.get("results")
            if not isinstance(results, list):
                raise CatalogError("Catalog returned an invalid result page.")
            for item in results:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    continue
                if new_scene_count >= self.settings.satellite_monitor_max_scenes:
                    break
                scene_id = item["id"]
                assets = item.get("available_assets")
                safe_assets = [
                    {
                        "key": asset.get("key"),
                        "href": asset.get("href"),
                        "media_type": asset.get("media_type"),
                        "size_bytes": asset.get("size_bytes"),
                        "checksum": asset.get("checksum"),
                    }
                    for asset in assets
                    if isinstance(asset, dict) and asset.get("href")
                ] if isinstance(assets, list) else []
                acquisition_supported = bool(
                    item.get("acquisition_supported") is True
                    and satellite_acquisition.ITEM_ID_PATTERN.fullmatch(scene_id)
                )
                added = self.store.add_scene(
                    {
                        "scene_id": scene_id,
                        "collection": COLLECTION,
                        "source": SOURCE_NAME,
                        "observation_at": _parse_datetime(item.get("acquisition_datetime")),
                        "published_at": _parse_datetime(item.get("published_datetime")),
                        "discovered_at": utc_now(),
                        "cloud_cover": item.get("cloud_cover"),
                        "bbox": _json_safe(item.get("bbox")),
                        "assets": safe_assets,
                        "acquisition_supported": acquisition_supported,
                        "status": (
                            "discovered"
                            if acquisition_supported
                            else "rejected"
                        ),
                        "attempt_count": 0,
                        "last_error": (
                            None
                            if acquisition_supported
                            else "Required 10 m Sentinel-2 assets or a supported scene identifier are unavailable."
                        ),
                    }
                )
                if added:
                    created.append(scene_id)
                    new_scene_count += 1
            cursor = response.get("next_cursor")
            if (
                not response.get("has_more")
                or not isinstance(cursor, str)
                or not cursor
                or cursor in seen_cursors
                or not results
            ):
                break
            seen_cursors.add(cursor)
        return created

    def _process_scene(self, scene_id: str) -> None:
        scene = self.store.scene(scene_id)
        if scene is None or not scene["acquisition_supported"]:
            return
        if scene["sha256"] and self._model_status()["status"] != "ready":
            self.store.update_scene(
                scene_id,
                status="model_not_ready",
                analysis_status="model_not_ready",
                last_error=None,
                processing_completed_at=utc_now(),
            )
            return
        now = utc_now()
        attempt = scene["attempt_count"] + 1
        self.store.update_scene(
            scene_id,
            attempt_count=attempt,
            last_attempt_at=now,
            status="downloading" if not scene["sha256"] else "analyzing",
            last_error=None,
        )
        root = self.settings.satellite_monitor_data_dir
        root.mkdir(parents=True, exist_ok=True)
        if root.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("Monitor storage must remain outside the project source tree.")
        current_bytes = self.store.stored_bytes()
        if current_bytes >= self.settings.satellite_monitor_max_storage_bytes:
            raise OSError("The configured satellite-monitor storage limit has been reached.")
        if shutil.disk_usage(root).free < min(
            self.settings.satellite_monitor_max_scene_bytes, 64 * 1024 * 1024
        ):
            raise OSError("Insufficient free disk space for a bounded scene acquisition.")

        scene_hash = hashlib.sha256(scene_id.encode("utf-8")).hexdigest()
        scene_dir = root / "scenes" / scene_hash
        scene_dir.mkdir(parents=True, exist_ok=True)
        source_path = scene_dir / "source.tif"
        started = monotonic_time.perf_counter()
        source_promoted = False
        source_recorded = bool(scene["sha256"])
        inference_started = False
        try:
            if not scene["sha256"]:
                if source_path.exists():
                    raise OSError("An unverified scene file already exists; refusing to overwrite it.")
                staging = root / "staging"
                staging.mkdir(parents=True, exist_ok=True)
                temporary_path = staging / f"{uuid4().hex}.tif"
                aoi = self.aoi
                if aoi is None:
                    raise MonitoringValidationError(
                        "The monitoring AOI is unavailable for scene acquisition."
                    )
                request = SentinelSceneAcquisitionRequest(
                    provider=SOURCE_NAME,
                    collection=COLLECTION,
                    scene_id=scene_id,
                    aoi=aoi,
                )
                acquisition = self.acquire_scene(
                    request,
                    temporary_path,
                    max_bytes=self.settings.satellite_monitor_max_scene_bytes,
                    max_pixels=min(
                        self.settings.max_imagery_pixels,
                        self.settings.max_ai_inference_pixels,
                    ),
                    aoi_max_area_km2=self.settings.max_satellite_acquisition_area_km2,
                )
                if (
                    not isinstance(acquisition, dict)
                    or acquisition.get("scene_id") != scene_id
                    or acquisition.get("collection") != COLLECTION
                    or acquisition.get("bands") != list(INPUT_BANDS)
                    or not _parse_datetime(acquisition.get("acquisition_datetime"))
                ):
                    raise MonitoringValidationError(
                        "CDSE acquisition metadata does not match the discovered scene."
                    )
                discovered_at = _parse_datetime(scene.get("observation_at"))
                acquired_at = _parse_datetime(acquisition.get("acquisition_datetime"))
                if (
                    discovered_at is not None
                    and acquired_at is not None
                    and abs((discovered_at - acquired_at).total_seconds()) > 1
                ):
                    raise MonitoringValidationError(
                        "The acquired observation time differs from the catalog item."
                    )
                self.store.update_scene(scene_id, status="validating")
                self._validate_raster(temporary_path)
                size_bytes = temporary_path.stat().st_size
                if size_bytes > self.settings.satellite_monitor_max_scene_bytes:
                    raise OSError("The validated raster exceeds the per-scene storage limit.")
                if current_bytes + size_bytes > self.settings.satellite_monitor_max_storage_bytes:
                    raise OSError("The configured satellite-monitor storage limit would be exceeded.")
                checksum = self._sha256(temporary_path)
                os.replace(temporary_path, source_path)
                source_promoted = True
                self.store.update_scene(
                    scene_id,
                    storage_key=f"scenes/{scene_hash}/source.tif",
                    sha256=checksum,
                    downloaded_bytes=size_bytes,
                    validated_at=utc_now(),
                )
                source_recorded = True
            else:
                self._validate_raster(source_path)
                if self._sha256(source_path) != scene["sha256"]:
                    raise MonitoringValidationError(
                        "The stored scene checksum no longer matches its manifest."
                    )
            model = self._model_status()
            if model["status"] != "ready":
                self.store.update_scene(
                    scene_id,
                    status="model_not_ready",
                    analysis_status="model_not_ready",
                    last_error=None,
                    processing_completed_at=utc_now(),
                    processing_ms=round((monotonic_time.perf_counter() - started) * 1000),
                )
                return
            classification_path = scene_dir / "classification.tif"
            confidence_path = scene_dir / "confidence.tif"
            if classification_path.exists() or confidence_path.exists():
                raise OSError("Unregistered inference output exists; refusing to overwrite it.")
            inference_started = True
            result = self.inference(
                source_path,
                classification_path,
                confidence_path,
                configuration=self.settings,
            )
            self._validate_inference_outputs(
                source_path, classification_path, confidence_path
            )
            classification_size = classification_path.stat().st_size
            confidence_size = confidence_path.stat().st_size
            if (
                classification_size <= 0
                or confidence_size <= 0
                or classification_size > self.settings.satellite_monitor_max_scene_bytes
                or confidence_size > self.settings.satellite_monitor_max_scene_bytes
                or self.store.stored_bytes()
                + classification_size
                + confidence_size
                > self.settings.satellite_monitor_max_storage_bytes
            ):
                classification_path.unlink(missing_ok=True)
                confidence_path.unlink(missing_ok=True)
                raise OSError("Inference outputs exceed the configured storage limits.")
            model_data = result.get("model", {})
            distribution = result.get("classes", [])
            self.store.update_scene(
                scene_id,
                status="analyzed",
                analysis_status="completed",
                model_name=model_data.get("name"),
                model_version=model_data.get("version"),
                model_sha256=model_data.get("weights_sha256"),
                processing_ms=round(
                    (monotonic_time.perf_counter() - started) * 1000
                ),
                classification_key=f"scenes/{scene_hash}/classification.tif",
                classification_sha256=self._sha256(classification_path),
                confidence_key=f"scenes/{scene_hash}/confidence.tif",
                confidence_sha256=self._sha256(confidence_path),
                artifact_bytes=classification_size + confidence_size,
                class_distribution={"classes": distribution},
                processing_completed_at=utc_now(),
            )
        except Exception as exc:
            for temporary in (locals().get("temporary_path"),):
                if isinstance(temporary, Path):
                    temporary.unlink(missing_ok=True)
            if source_promoted and not source_recorded:
                source_path.unlink(missing_ok=True)
            if inference_started:
                classification_path.unlink(missing_ok=True)
                confidence_path.unlink(missing_ok=True)
            self.store.update_scene(
                scene_id,
                status="failed",
                analysis_status="failed" if source_recorded else None,
                last_error=_safe_failure(exc),
                next_retry_at=utc_now()
                + timedelta(seconds=min(2 ** max(attempt - 1, 0), 300)),
                processing_completed_at=utc_now(),
                processing_ms=round((monotonic_time.perf_counter() - started) * 1000),
            )
            logger.warning(
                "Satellite scene processing failed.",
                extra={
                    "event": "monitor.scene_failed",
                    "error_type": type(exc).__name__,
                    "scene_id": scene_id[:128],
                },
            )

    def _validate_raster(self, path: Path) -> None:
        if not path.is_file() or path.is_symlink():
            raise OSError("The acquired scene is not a regular raster file.")
        with rasterio.open(path) as dataset:
            aoi = self.aoi
            if aoi is None:
                raise MonitoringValidationError(
                    "The monitoring AOI is unavailable for raster validation."
                )
            if dataset.crs is None:
                raise MonitoringValidationError(
                    "The downloaded scene has no coordinate reference system."
                )
            transformer = Transformer.from_crs(
                "EPSG:4326", dataset.crs, always_xy=True
            )

            def project_coordinates(coordinates: np.ndarray) -> np.ndarray:
                x, y = transformer.transform(
                    coordinates[:, 0], coordinates[:, 1]
                )
                return np.stack((x, y), axis=-1)

            projected_aoi = transform_geometry(
                shape(aoi.model_dump()),
                project_coordinates,
            )
            if (
                dataset.driver != "GTiff"
                or dataset.count != len(INPUT_BANDS)
                or not dataset.crs.is_projected
                or dataset.width < 1
                or dataset.height < 1
                or dataset.width * dataset.height
                > min(self.settings.max_imagery_pixels, self.settings.max_ai_inference_pixels)
                or tuple(dataset.descriptions) != INPUT_BANDS
                or any(not value for value in dataset.descriptions)
                or not all(value == "uint16" for value in dataset.dtypes)
                or not all(abs(value) == 10 for value in dataset.res)
                or not all(value == value and abs(value) < float("inf") for value in dataset.transform)
                or dataset.dataset_mask().sum() == 0
                or dataset.tags().get("ASSET_BANDS") != ",".join(INPUT_BANDS)
                or not box(*dataset.bounds).buffer(10).covers(projected_aoi)
            ):
                raise MonitoringValidationError(
                    "The downloaded scene failed CRS, AOI coverage, resolution, band, grid, or nodata validation."
                )
        if path.stat().st_size <= 0:
            raise OSError("The acquired scene is empty.")

    @staticmethod
    def _validate_inference_outputs(
        source_path: Path, classification_path: Path, confidence_path: Path
    ) -> None:
        with (
            rasterio.open(source_path) as source,
            rasterio.open(classification_path) as classification,
            rasterio.open(confidence_path) as confidence,
        ):
            for dataset, dtype, nodata in (
                (classification, "uint16", 0),
                (confidence, "float32", -1.0),
            ):
                if (
                    dataset.driver != "GTiff"
                    or dataset.count != 1
                    or dataset.dtypes != (dtype,)
                    or dataset.crs != source.crs
                    or dataset.width != source.width
                    or dataset.height != source.height
                    or dataset.transform != source.transform
                    or dataset.nodata != nodata
                    or dataset.dataset_mask().sum() == 0
                ):
                    raise MonitoringValidationError(
                        "A land-cover inference artifact does not match its source grid."
                    )
                for _, window in dataset.block_windows(1):
                    valid = dataset.read_masks(1, window=window) > 0
                    values = dataset.read(1, window=window)
                    if dataset is confidence:
                        if (
                            not np.isfinite(values[valid]).all()
                            or np.any(values[valid] < 0)
                            or np.any(values[valid] > 1)
                        ):
                            raise MonitoringValidationError(
                                "A confidence raster contains invalid probability values."
                            )
                    elif np.any(values[valid] == 0):
                        raise MonitoringValidationError(
                            "A classification raster contains a nodata class in valid pixels."
                        )

    @staticmethod
    def _sha256(path: Path) -> str:
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        return checksum.hexdigest()


class MonitoringScheduler:
    def __init__(self, pipeline: MonitoringPipeline):
        self.pipeline = pipeline
        self._stop = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._scan_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not self.pipeline.configured or self._loop_task is not None:
            return
        await asyncio.to_thread(self.pipeline.initialize)
        self._stop.clear()
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        if self._scan_tasks:
            await asyncio.gather(*tuple(self._scan_tasks), return_exceptions=True)

    async def request_manual_scan(self) -> bool:
        if not self.pipeline.configured:
            return False
        claimed = await asyncio.to_thread(self.pipeline.claim_scan)
        if not claimed:
            return False
        task = asyncio.create_task(
            asyncio.to_thread(self.pipeline.run_scan, claimed=True)
        )
        self._scan_tasks.add(task)
        task.add_done_callback(self._scan_tasks.discard)
        return True

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                current = datetime.now(timezone.utc)
                state = await asyncio.to_thread(
                    self.pipeline.store.status,
                    self.pipeline.settings.satellite_monitor_timezone,
                    self.pipeline.settings.satellite_monitor_schedule_time,
                )
                next_run = _parse_datetime(state.get("next_scheduled_run_at"))
                lease_expires = _parse_datetime(state.get("lease_expires_at"))
                if (
                    next_run is not None
                    and current >= next_run
                    and not self._scan_tasks
                    and (
                        state.get("status") != "running"
                        or lease_expires is None
                        or current >= lease_expires
                    )
                ):
                    task = asyncio.create_task(
                        asyncio.to_thread(self.pipeline.run_scan)
                    )
                    self._scan_tasks.add(task)
                    task.add_done_callback(self._scan_tasks.discard)
                await asyncio.wait_for(self._stop.wait(), timeout=POLL_SECONDS)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except SQLAlchemyError:
                logger.exception(
                    "Satellite monitoring scheduler could not read persistent state.",
                    extra={"event": "monitor.scheduler_state_failed"},
                )
                await asyncio.sleep(POLL_SECONDS)
