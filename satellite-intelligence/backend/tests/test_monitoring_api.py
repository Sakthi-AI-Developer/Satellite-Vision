from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.routes as routes
from auth import AuthenticatedUser
from config import settings
from main import app
from persistence.database import PersistenceRepository
from satellite.monitoring import MonitoringPipeline


AOI = {
    "type": "Polygon",
    "coordinates": [[[77.0, 11.0], [77.01, 11.0], [77.01, 11.01], [77.0, 11.01], [77.0, 11.0]]],
}


def _pipeline(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'monitor-api.db'}"
    repository = PersistenceRepository(database_url)
    repository.create_schema_for_tests()
    local_settings = replace(
        settings,
        database_url=database_url,
        satellite_monitor_enabled=True,
        satellite_monitor_aoi_json=json.dumps(AOI),
        satellite_monitor_data_dir=tmp_path / "monitor-data",
        cdse_username="test-user",
        cdse_password="test-password",
    )
    pipeline = MonitoringPipeline(
        local_settings,
        repository.engine,
        catalog_search=lambda _: {"results": [], "has_more": False},
        model_listing=lambda _: {"models": [{"status": "model_not_ready"}]},
    )
    monkeypatch.setattr(routes, "monitoring_pipeline", pipeline)
    monkeypatch.setattr(routes, "settings", local_settings)
    return pipeline, repository


def test_monitoring_endpoints_require_authentication(tmp_path, monkeypatch):
    pipeline, repository = _pipeline(tmp_path, monkeypatch)
    app.dependency_overrides[routes.current_user] = lambda: None
    try:
        response = TestClient(app).get("/api/monitoring/status")
        assert response.status_code == 401
    finally:
        app.dependency_overrides.pop(routes.current_user, None)
        repository.close()


def test_monitoring_status_scene_history_and_manual_conflict(tmp_path, monkeypatch):
    pipeline, repository = _pipeline(tmp_path, monkeypatch)
    pipeline.initialize()
    pipeline.store.add_scene({
        "scene_id": "monitor-scene-1",
        "collection": "sentinel-2-l2a",
        "source": "Copernicus Data Space Ecosystem",
        "observation_at": None,
        "published_at": None,
        "discovered_at": datetime.now(timezone.utc),
        "cloud_cover": 2.5,
        "bbox": None,
        "assets": [],
        "acquisition_supported": False,
        "status": "rejected",
        "attempt_count": 0,
    })
    artifact = Path(pipeline.settings.satellite_monitor_data_dir) / "scenes" / "fixture" / "classification.tif"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"validated test artifact")
    pipeline.store.update_scene(
        "monitor-scene-1",
        classification_key="scenes/fixture/classification.tif",
    )

    class BusyScheduler:
        async def request_manual_scan(self):
            return False

        async def stop(self):
            return None

    monkeypatch.setattr(routes, "monitoring_scheduler", BusyScheduler())
    app.dependency_overrides[routes.current_user] = lambda: AuthenticatedUser("viewer")
    try:
        client = TestClient(app)
        status = client.get("/api/monitoring/status")
        history = client.get("/api/monitoring/scenes?limit=1&offset=0")
        invalid_page = client.get("/api/monitoring/scenes?limit=0")
        busy = client.post("/api/monitoring/scan")
        output = client.get(
            "/api/monitoring/scenes/monitor-scene-1/artifacts/classification.tif"
        )

        assert status.status_code == 200
        assert status.json()["enabled"] is True
        assert status.json()["model"]["status"] == "model_not_ready"
        assert history.status_code == 200
        assert history.json()["scenes"][0]["scene_id"] == "monitor-scene-1"
        assert history.json()["scenes"][0]["status"] == "rejected"
        assert invalid_page.status_code == 422
        assert busy.status_code == 409
        assert output.status_code == 200
        assert output.content == b"validated test artifact"
    finally:
        app.dependency_overrides.pop(routes.current_user, None)
        repository.close()
