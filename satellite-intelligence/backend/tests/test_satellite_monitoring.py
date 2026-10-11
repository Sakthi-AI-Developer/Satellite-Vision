from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from sqlalchemy.orm import sessionmaker

from config import settings
from persistence.models import Base, MonitoredSatelliteScene, SatelliteMonitorState, utc_now
from satellite.acquisition import _sha256_checksum
from satellite import monitoring as monitoring_module
from satellite.monitoring import (
    MonitoringPipeline,
    MonitoringScheduler,
    MonitoringStore,
    next_daily_run,
)
from satellite.catalog import CatalogRateLimitError, CatalogTimeoutError


AOI = {
    "type": "Polygon",
    "coordinates": [[
        [77.0, 11.0],
        [77.00005, 11.0],
        [77.00005, 11.00005],
        [77.0, 11.00005],
        [77.0, 11.0],
    ]],
}
SCENE_ID = "S2A_MSIL2A_20240101T103021_N0500_R108_T30TXN_20240101T130000"


def _configuration(tmp_path: Path):
    return replace(
        settings,
        database_url=f"sqlite:///{tmp_path / 'monitor.db'}",
        satellite_monitor_enabled=True,
        satellite_monitor_timezone="Asia/Kolkata",
        satellite_monitor_schedule_time="03:00",
        satellite_monitor_aoi_json=json.dumps(AOI),
        satellite_monitor_lookback_days=14,
        satellite_monitor_max_cloud_cover=20,
        satellite_monitor_max_scenes=3,
        satellite_monitor_max_retries=2,
        satellite_monitor_max_scene_bytes=10 * 1024 * 1024,
        satellite_monitor_max_storage_bytes=30 * 1024 * 1024,
        satellite_monitor_data_dir=tmp_path / "outside-source" / "monitor",
        cdse_username="test-user",
        cdse_password="test-password",
    )


def _engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'monitor.db'}")
    Base.metadata.create_all(engine)
    return engine


def _item(scene_id: str = SCENE_ID, *, supported: bool = True):
    return {
        "id": scene_id,
        "acquisition_datetime": "2024-01-01T10:30:21Z",
        "published_datetime": "2024-01-02T12:00:00Z",
        "cloud_cover": 8.2,
        "bbox": [77.0, 11.0, 77.01, 11.01],
        "acquisition_supported": supported,
        "available_assets": [
            {
                "key": "B02_10m",
                "href": "s3://eodata/Sentinel-2/MSI/L2A/test_B02_10m.jp2",
                "media_type": "image/jp2",
                "size_bytes": 100,
            }
        ],
    }


def _write_raster(path: Path) -> None:
    x, y = Transformer.from_crs(
        "EPSG:4326", "EPSG:32643", always_xy=True
    ).transform(77.000025, 11.000025)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=4,
        dtype="uint16",
        crs="EPSG:32643",
        transform=from_origin(x - 10, y + 10, 10, 10),
        nodata=0,
    ) as output:
        for band, description in enumerate(("B02", "B03", "B04", "B08"), start=1):
            output.write(np.array([[100, 200], [300, 400]], dtype=np.uint16), band)
            output.set_band_description(band, description)
        output.update_tags(ASSET_BANDS="B02,B03,B04,B08")


def _catalog_page(*items, has_more=False, next_cursor=None):
    return {
        "results": list(items),
        "has_more": has_more,
        "next_cursor": next_cursor,
    }


def _acquisition_metadata():
    return {
        "scene_id": SCENE_ID,
        "collection": "sentinel-2-l2a",
        "acquisition_datetime": "2024-01-01T10:30:21Z",
        "bands": ["B02", "B03", "B04", "B08"],
    }


def test_daily_run_is_computed_in_configured_timezone():
    now = datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc)
    run_at = next_daily_run(now, "Asia/Kolkata", "03:00")
    assert run_at == datetime(2026, 5, 1, 21, 30, tzinfo=timezone.utc)


def test_source_checksum_metadata_accepts_sha256_and_multihash_encodings():
    digest = "a" * 64
    assert _sha256_checksum({"file:checksum": f"sha256:{digest}"}) == digest
    assert _sha256_checksum({"file:checksum": f"1220{digest}"}) == digest
    assert _sha256_checksum({"file:checksum": "unsupported"}) is None


def test_database_lease_prevents_duplicate_worker_scans(tmp_path):
    engine = _engine(tmp_path)
    store_a = MonitoringStore(engine)
    store_b = MonitoringStore(engine)
    store_a.initialize_schedule("Asia/Kolkata", "03:00")

    assert store_a.acquire_lease("worker-a")
    assert not store_b.acquire_lease("worker-b")
    store_a.finish_scan(
        "worker-a",
        status="completed",
        error=None,
        catalog_succeeded=True,
        timezone_name="Asia/Kolkata",
        schedule_time="03:00",
    )
    assert store_b.acquire_lease("worker-b")


def test_discovery_paginates_and_persists_first_seen_metadata(tmp_path):
    engine = _engine(tmp_path)
    pages = iter([
        _catalog_page(_item(), has_more=True, next_cursor="page-two"),
        _catalog_page(_item("S2B_MSIL2A_20240102T103021_N0500_R108_T30TXN_20240102T130000")),
    ])
    requests = []

    def search(request):
        requests.append(request.cursor)
        return next(pages)

    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=search,
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    pipeline.initialize()
    discovered = pipeline._discover()

    assert requests == [None, "page-two"]
    assert len(discovered) == 2
    scene = pipeline.store.scene(SCENE_ID)
    assert scene["observation_at"] == "2024-01-01T10:30:21+00:00"
    assert scene["published_at"] == "2024-01-02T12:00:00+00:00"
    assert scene["assets"][0]["href"].startswith("s3://eodata/")

    previous_first_seen = scene["discovered_at"]
    assert pipeline.store.add_scene({
        "scene_id": SCENE_ID,
        "collection": "sentinel-2-l2a",
        "source": "Copernicus Data Space Ecosystem",
        "observation_at": None,
        "published_at": None,
        "discovered_at": datetime.now(timezone.utc),
        "cloud_cover": None,
        "bbox": None,
        "assets": [],
        "acquisition_supported": False,
        "status": "rejected",
        "attempt_count": 0,
    }) is False
    assert pipeline.store.scene(SCENE_ID)["discovered_at"] == previous_first_seen


def test_empty_catalog_is_a_successful_empty_scan(tmp_path):
    engine = _engine(tmp_path)
    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=lambda _: _catalog_page(),
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    pipeline.run_scan()
    status = pipeline.status()
    assert status["last_successful_scan_at"] is not None
    assert status["scene_counts"]["discovered"] == 0
    assert status["status"] == "completed"


def test_download_is_validated_and_preserved_when_model_is_not_ready(tmp_path):
    engine = _engine(tmp_path)
    acquisitions = []

    def acquire(_request, destination, **_kwargs):
        acquisitions.append(destination)
        _write_raster(destination)
        return _acquisition_metadata()

    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=lambda _: _catalog_page(_item()),
        acquire_scene=acquire,
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    pipeline.run_scan()

    scene = pipeline.store.scene(SCENE_ID)
    assert len(acquisitions) == 1
    assert scene["status"] == "model_not_ready", scene["last_error"]
    assert scene["analysis_status"] == "model_not_ready"
    assert scene["sha256"] and len(scene["sha256"]) == 64
    assert scene["downloaded_bytes"] > 0
    storage_id = hashlib.sha256(SCENE_ID.encode("utf-8")).hexdigest()
    assert (
        pipeline.settings.satellite_monitor_data_dir
        / "scenes"
        / storage_id
        / "source.tif"
    ).is_file()
    assert acquisitions[0].parent.name == "staging"
    assert not acquisitions[0].exists()
    assert scene["classification_url"] is None
    assert pipeline.status()["scene_counts"]["downloaded"] == 1


def test_approved_model_path_persists_only_verified_mock_artifact_outputs(tmp_path):
    engine = _engine(tmp_path)
    inference_calls = []

    def acquire(_request, destination, **_kwargs):
        _write_raster(destination)
        return _acquisition_metadata()

    def inference(source, classification, confidence, *, configuration):
        inference_calls.append(source)
        with rasterio.open(source) as input_raster:
            class_profile = input_raster.profile.copy()
            class_profile.update(count=1, dtype="uint16", nodata=0)
            confidence_profile = input_raster.profile.copy()
            confidence_profile.update(count=1, dtype="float32", nodata=-1.0)
        with rasterio.open(classification, "w", **class_profile) as output:
            output.write(np.ones((2, 2), dtype=np.uint16), 1)
        with rasterio.open(confidence, "w", **confidence_profile) as output:
            output.write(np.full((2, 2), 0.75, dtype=np.float32), 1)
        return {
            "model": {
                "name": "mock-approved-model",
                "version": "test-only",
                "weights_sha256": "a" * 64,
            },
            "classes": [{"class_index": 1, "label": "test fixture"}],
        }

    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=lambda _: _catalog_page(_item()),
        acquire_scene=acquire,
        model_listing=lambda _: {
            "models": [{
                "status": "ready",
                "name": "mock-approved-model",
                "version": "test-only",
            }]
        },
        inference=inference,
    )
    pipeline.run_scan()

    scene = pipeline.store.scene(SCENE_ID)
    assert scene["status"] == "analyzed"
    assert scene["analysis_status"] == "completed"
    assert scene["model_name"] == "mock-approved-model"
    assert scene["model_sha256"] == "a" * 64
    assert len(scene["classification_sha256"]) == 64
    assert len(scene["confidence_sha256"]) == 64
    assert scene["classification_url"] and scene["confidence_url"]
    assert inference_calls


def test_unsupported_assets_are_rejected_without_acquisition(tmp_path):
    engine = _engine(tmp_path)
    acquisition_calls = []
    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=lambda _: _catalog_page(_item(supported=False)),
        acquire_scene=lambda *args, **kwargs: acquisition_calls.append(args),
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    pipeline.run_scan()

    scene = pipeline.store.scene(SCENE_ID)
    assert scene["status"] == "rejected"
    assert "assets are unavailable" in scene["last_error"]
    assert acquisition_calls == []


def test_corrupt_download_is_rejected_and_retry_count_is_bounded(tmp_path):
    engine = _engine(tmp_path)
    acquisition_calls = []

    def acquire(_request, destination, **_kwargs):
        acquisition_calls.append(destination)
        destination.write_bytes(b"not a raster")
        return _acquisition_metadata()

    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=lambda _: _catalog_page(_item()),
        acquire_scene=acquire,
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    pipeline.run_scan()
    assert pipeline.store.scene(SCENE_ID)["status"] == "failed"

    pipeline.store.update_scene(SCENE_ID, next_retry_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    pipeline.run_scan()
    pipeline.store.update_scene(SCENE_ID, next_retry_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    pipeline.run_scan()
    pipeline.store.update_scene(SCENE_ID, next_retry_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    pipeline.run_scan()

    scene = pipeline.store.scene(SCENE_ID)
    assert len(acquisition_calls) == 3
    assert scene["attempt_count"] == 3
    assert scene["status"] == "failed"
    assert list((pipeline.settings.satellite_monitor_data_dir / "staging").iterdir()) == []


@pytest.mark.parametrize(
    "catalog_error",
    [CatalogRateLimitError(), CatalogTimeoutError()],
)
def test_catalog_rate_limit_and_timeout_retries_are_bounded(
    tmp_path, monkeypatch, catalog_error
):
    engine = _engine(tmp_path)
    calls = []
    waits = []

    def search(_request):
        calls.append(1)
        if len(calls) == 1:
            raise catalog_error
        return _catalog_page()

    monkeypatch.setattr(monitoring_module.monotonic_time, "sleep", waits.append)
    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=search,
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    assert pipeline._discover() == []
    assert len(calls) == 2
    assert waits == [1]


def test_storage_limit_rejects_oversized_scene_without_promoting_file(tmp_path):
    engine = _engine(tmp_path)
    configuration = replace(
        _configuration(tmp_path),
        satellite_monitor_max_scene_bytes=10,
        satellite_monitor_max_storage_bytes=10,
    )

    def acquire(_request, destination, **_kwargs):
        _write_raster(destination)
        return _acquisition_metadata()

    pipeline = MonitoringPipeline(
        configuration,
        engine,
        catalog_search=lambda _: _catalog_page(_item()),
        acquire_scene=acquire,
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    pipeline.run_scan()

    scene = pipeline.store.scene(SCENE_ID)
    assert scene["status"] == "failed"
    assert scene["downloaded_bytes"] is None
    assert pipeline.store.stored_bytes() == 0
    assert list((configuration.satellite_monitor_data_dir / "staging").iterdir()) == []


def test_restart_recovers_expired_scan_and_interrupted_scene(tmp_path):
    engine = _engine(tmp_path)
    store = MonitoringStore(engine)
    store.initialize_schedule("Asia/Kolkata", "03:00")
    assert store.acquire_lease("crashed-worker")
    with sessionmaker(engine).begin() as session:
        state = session.get(SatelliteMonitorState, 1)
        state.last_attempt_at = utc_now() - timedelta(hours=2)
        state.lease_expires_at = utc_now() - timedelta(minutes=1)
        session.add(
            MonitoredSatelliteScene(
                scene_id=SCENE_ID,
                collection="sentinel-2-l2a",
                source="Copernicus Data Space Ecosystem",
                discovered_at=utc_now(),
                assets=[],
                acquisition_supported=True,
                status="downloading",
                attempt_count=1,
                last_attempt_at=utc_now() - timedelta(hours=2),
            )
        )

    store.initialize_schedule("Asia/Kolkata", "03:00")

    assert store.status("Asia/Kolkata", "03:00")["status"] == "failed"
    assert store.scene(SCENE_ID)["status"] == "failed"
    assert store.scene(SCENE_ID)["next_retry_at"] is not None


def test_scheduler_start_is_idempotent_and_stops_cleanly(tmp_path):
    engine = _engine(tmp_path)
    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=lambda _: _catalog_page(),
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    scheduler = MonitoringScheduler(pipeline)

    async def exercise():
        await scheduler.start()
        first_task = scheduler._loop_task
        await scheduler.start()
        assert scheduler._loop_task is first_task
        await scheduler.stop()
        assert scheduler._loop_task is None

    import asyncio

    asyncio.run(exercise())


def test_concurrent_manual_scan_requests_share_the_database_lease(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    pipeline = MonitoringPipeline(
        _configuration(tmp_path),
        engine,
        catalog_search=lambda _: _catalog_page(),
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    monkeypatch.setattr(pipeline, "run_scan", lambda **_kwargs: None)
    scheduler = MonitoringScheduler(pipeline)

    async def exercise():
        first = await scheduler.request_manual_scan()
        second = await scheduler.request_manual_scan()
        await asyncio.sleep(0)
        await scheduler.stop()
        pipeline.store.finish_scan(
            pipeline.owner,
            status="completed",
            error=None,
            catalog_succeeded=False,
            timezone_name="Asia/Kolkata",
            schedule_time="03:00",
        )
        return first, second

    import asyncio

    first, second = asyncio.run(exercise())
    assert first is True
    assert second is False
