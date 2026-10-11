from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pytest
import rasterio
from fastapi import HTTPException
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from sklearn.tree import DecisionTreeClassifier

import api.routes as routes
import main as application
from auth import AuthenticatedUser
from config import settings
from geoai import inference
from main import app
from persistence.artifacts import LocalArtifactStore
from persistence.database import PersistenceRepository
from persistence.manager import PersistenceManager

client = TestClient(app)
TRAINING_NOTE = "Synthetic training fixture only; not a satellite observation."


def training_model():
    model = DecisionTreeClassifier(max_depth=2, random_state=7)
    model.fit(
        np.asarray(
            [
                [0.1, 0.2, 0.4, 0.2],
                [0.2, 0.3, 0.5, 0.2],
                [0.1, 0.3, 0.2, 0.8],
                [0.2, 0.4, 0.2, 0.9],
            ],
            dtype=np.float32,
        ),
        ["water", "water", "vegetation", "vegetation"],
    )
    return model


def write_manifest(model_dir: Path, model=None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = model_dir / "classifier.joblib"
    joblib.dump(model or training_model(), artifact)
    manifest_path = model_dir / "sentinel2-landcover.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "Test four-band classifier",
                "version": "test-fixture-1",
                "license": "Test fixture; not a redistributable pretrained model.",
                "source": "Generated inside automated tests.",
                "attribution": TRAINING_NOTE,
                "artifact": "classifier.joblib",
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "deployment_approved": True,
                "bands": ["B02", "B03", "B04", "B08"],
                "normalization": "divide_by_10000",
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def inference_settings(tmp_path: Path):
    model_dir = tmp_path / "private-models"
    manifest_path = model_dir / "sentinel2-landcover.json"
    return replace(
        settings,
        ai_model_dir=model_dir,
        ai_landcover_manifest_path=manifest_path,
        max_ai_model_bytes=8 * 1024 * 1024,
        max_ai_inference_pixels=10_000,
        max_raster_bytes=8 * 1024 * 1024,
        max_artifact_bytes=8 * 1024 * 1024,
    )


def write_source(path: Path, *, crs="EPSG:32643", bands=("B02", "B03", "B04", "B08")):
    values = np.zeros((4, 6, 8), dtype=np.uint16)
    values[0] = 1000
    values[1] = 3000
    values[2] = 2000
    values[3] = 8000
    values[:, 0, 0] = 0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=6,
        count=4,
        dtype="uint16",
        nodata=0,
        crs=crs,
        transform=from_origin(500_000, 1_500_000, 10, 10),
    ) as destination:
        destination.write(values)
        for index, band in enumerate(bands, start=1):
            destination.set_band_description(index, band)


def make_upload_bytes() -> bytes:
    with tempfile.TemporaryDirectory(prefix="satellite-ai-test-upload-") as staging:
        path = Path(staging) / "test-scene.tif"
        write_source(path)
        return path.read_bytes()


@pytest.fixture
def ai_backend(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    output_dir = data_dir / "outputs"
    output_dir.mkdir(parents=True)
    local_settings = replace(
        inference_settings(tmp_path),
        data_dir=data_dir,
        output_dir=output_dir,
        database_url=None,
        supabase_url=None,
        supabase_anon_key=None,
        supabase_service_role_key=None,
        require_auth=False,
    )
    repository = PersistenceRepository(f"sqlite:///{tmp_path / 'ai-inference.sqlite'}")
    repository.create_schema_for_tests()
    manager = PersistenceManager(
        local_settings,
        repository=repository,
        artifact_store=LocalArtifactStore(output_dir),
    )
    write_manifest(local_settings.ai_model_dir)
    monkeypatch.setattr(routes, "settings", local_settings)
    monkeypatch.setattr(routes, "persistence_manager", manager)
    monkeypatch.setattr(application, "settings", local_settings)
    app.dependency_overrides[routes.current_user] = lambda: AuthenticatedUser("user-1")
    yield local_settings, repository, manager, output_dir
    app.dependency_overrides.pop(routes.current_user, None)
    manager.close()


def ingest_synthetic_scene() -> dict:
    return client.post(
        "/api/imagery/ingest",
        files={"file": ("synthetic-fixture.tif", make_upload_bytes(), "image/tiff")},
    ).json()


def test_model_registry_reports_missing_weights_without_loading_or_downloading(tmp_path):
    registry = inference.list_models(inference_settings(tmp_path))
    status = registry["models"][0]

    assert status["id"] == inference.LANDCOVER_MODEL_ID
    assert status["status"] == "model_not_ready"
    assert "Configure a local model manifest" in status["readiness_detail"]
    assert status["source"] is None
    assert status["artifact_size_bytes"] is None
    assert status["input_requirements"]["bands"] == ["B02", "B03", "B04", "B08"]

    candidate = registry["candidates"][0]
    assert candidate["id"] == "prithvi-eo-2.0-tiny-tl"
    assert candidate["status"] == "model_not_ready"
    assert candidate["license"].startswith("Apache-2.0")
    assert candidate["input_requirements"]["bands"] == [
        "B02", "B03", "B04", "B05", "B06", "B07"
    ]
    assert candidate["input_requirements"]["format"].startswith("four-frame")
    assert "segmentation head" in candidate["readiness_detail"]
    assert candidate["checkpoint_present"] is False


def test_model_registry_requires_explicit_approval_and_artifact_checksum(tmp_path):
    configuration = inference_settings(tmp_path)
    manifest_path = write_manifest(configuration.ai_model_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest.pop("deployment_approved")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status = inference.list_models(configuration)["models"][0]
    assert status["status"] == "model_not_ready"
    assert "deployment_approved must be true" in status["readiness_detail"]

    manifest["deployment_approved"] = True
    manifest.pop("artifact_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status = inference.list_models(configuration)["models"][0]
    assert status["status"] == "model_not_ready"
    assert "valid artifact SHA-256 checksum" in status["readiness_detail"]

    manifest["artifact_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status = inference.list_models(configuration)["models"][0]
    assert status["status"] == "model_not_ready"
    assert "checksum does not match" in status["readiness_detail"]


def test_local_classifier_inference_preserves_grid_masks_nodata_and_reports_confidence(
    tmp_path,
):
    configuration = inference_settings(tmp_path)
    write_manifest(configuration.ai_model_dir)
    source_path = tmp_path / "source.tif"
    classification_path = tmp_path / "classification.tif"
    confidence_path = tmp_path / "confidence.tif"
    write_source(source_path)

    result = inference.run_landcover_inference(
        source_path,
        classification_path,
        confidence_path,
        configuration=configuration,
    )

    assert result["status"] == "completed"
    assert result["model"]["framework"] == "scikit-learn"
    assert result["model"]["version"] == "test-fixture-1"
    assert result["input_requirements"]["normalization"] == "divide_by_10000"
    assert result["valid_pixels"] == 47
    assert result["confidence"]["valid_pixels"] == 47
    assert 0 <= result["confidence"]["minimum"] <= result["confidence"]["maximum"] <= 1
    with rasterio.open(source_path) as source:
        with rasterio.open(classification_path) as classes, rasterio.open(
            confidence_path
        ) as confidence:
            assert (classes.width, classes.height) == (source.width, source.height)
            assert classes.crs == source.crs
            assert classes.transform == source.transform
            assert confidence.crs == source.crs
            assert confidence.transform == source.transform
            assert classes.dataset_mask()[0, 0] == 0
            assert confidence.dataset_mask()[0, 0] == 0
            assert np.count_nonzero(classes.dataset_mask()) == 47
            assert classes.tags()["MODEL_ID"] == inference.LANDCOVER_MODEL_ID


@pytest.mark.parametrize(
    "crs,bands,expected",
    [
        (None, ("B02", "B03", "B04", "B08"), "georeferenced"),
        ("EPSG:32643", ("B02", "B03", "B04", "B11"), "labelled B02"),
    ],
)
def test_inference_rejects_unsupported_crs_or_incompatible_band_selection(
    tmp_path, crs, bands, expected
):
    configuration = inference_settings(tmp_path)
    write_manifest(configuration.ai_model_dir)
    source_path = tmp_path / "invalid-source.tif"
    write_source(source_path, crs=crs, bands=bands)

    with pytest.raises(inference.InferenceInputError, match=expected):
        inference.run_landcover_inference(
            source_path,
            tmp_path / "classification.tif",
            tmp_path / "confidence.tif",
            configuration=configuration,
        )


def test_inference_rejects_corrupt_input_and_corrupt_or_incompatible_model(
    tmp_path, monkeypatch
):
    configuration = inference_settings(tmp_path)
    write_manifest(configuration.ai_model_dir)
    corrupt_source = tmp_path / "corrupt.tif"
    corrupt_source.write_bytes(b"not a GeoTIFF")
    with pytest.raises(inference.InferenceInputError):
        inference.run_landcover_inference(
            corrupt_source,
            tmp_path / "classification.tif",
            tmp_path / "confidence.tif",
            configuration=configuration,
        )

    source_path = tmp_path / "source.tif"
    write_source(source_path)
    manifest = json.loads(configuration.ai_landcover_manifest_path.read_text())
    manifest["artifact"] = "corrupt.joblib"
    corrupt_artifact = configuration.ai_model_dir / "corrupt.joblib"
    corrupt_artifact.write_bytes(b"not joblib")
    manifest["artifact_sha256"] = hashlib.sha256(corrupt_artifact.read_bytes()).hexdigest()
    configuration.ai_landcover_manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(inference.ModelConfigurationError, match="could not be loaded"):
        inference.run_landcover_inference(
            source_path,
            tmp_path / "classification.tif",
            tmp_path / "confidence.tif",
            configuration=configuration,
        )

    manifest["artifact"] = "classifier.joblib"
    manifest["bands"] = ["B02", "B03", "B04", "B08"]
    configuration.ai_landcover_manifest_path.write_text(json.dumps(manifest))
    model = DecisionTreeClassifier().fit(np.ones((2, 2)), [0, 1])
    classifier_path = configuration.ai_model_dir / "classifier.joblib"
    joblib.dump(model, classifier_path)
    manifest["artifact_sha256"] = hashlib.sha256(classifier_path.read_bytes()).hexdigest()
    configuration.ai_landcover_manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(inference.ModelConfigurationError, match="feature count"):
        inference.run_landcover_inference(
            source_path,
            tmp_path / "classification.tif",
            tmp_path / "confidence.tif",
            configuration=configuration,
        )


def test_model_execution_errors_are_explicit_and_partial_outputs_stay_in_staging(
    tmp_path, monkeypatch
):
    configuration = inference_settings(tmp_path)
    write_manifest(configuration.ai_model_dir)
    source_path = tmp_path / "source.tif"
    write_source(source_path)

    def fail_prediction(_features):
        raise RuntimeError("private model error")

    class FailingClassifier:
        classes_ = np.asarray([0, 1])
        n_features_in_ = 4
        predict_proba = staticmethod(fail_prediction)

    monkeypatch.setattr(
        inference,
        "_load_estimator",
        lambda _spec: (FailingClassifier(), [0, 1]),
    )
    with pytest.raises(inference.InferenceExecutionError, match="failed while predicting"):
        inference.run_landcover_inference(
            source_path,
            tmp_path / "classification.tif",
            tmp_path / "confidence.tif",
            configuration=configuration,
        )
    assert not (tmp_path / "classification.tif").exists()
    assert not (tmp_path / "confidence.tif").exists()


def test_inference_bounds_pixels_preserves_existing_outputs_and_cleans_timeout_files(
    tmp_path,
):
    configuration = inference_settings(tmp_path)
    write_manifest(configuration.ai_model_dir)
    source_path = tmp_path / "source.tif"
    classification_path = tmp_path / "classification.tif"
    confidence_path = tmp_path / "confidence.tif"
    write_source(source_path)

    limited_configuration = replace(configuration, max_ai_inference_pixels=1)
    with pytest.raises(inference.InferenceInputError, match="bounded"):
        inference.run_landcover_inference(
            source_path,
            classification_path,
            confidence_path,
            configuration=limited_configuration,
        )
    assert not classification_path.exists()
    assert not confidence_path.exists()

    classification_path.write_bytes(b"existing output")
    with pytest.raises(inference.InferenceInputError, match="distinct new files"):
        inference.run_landcover_inference(
            source_path,
            classification_path,
            confidence_path,
            configuration=configuration,
        )
    assert classification_path.read_bytes() == b"existing output"

    classification_path.unlink()

    def time_out():
        raise inference.InferenceTimeoutError()

    with pytest.raises(inference.InferenceTimeoutError):
        inference.run_landcover_inference(
            source_path,
            classification_path,
            confidence_path,
            configuration=configuration,
            deadline_check=time_out,
        )
    assert not classification_path.exists()
    assert not confidence_path.exists()

    def http_timeout():
        raise HTTPException(status_code=504, detail="Analysis deadline elapsed.")

    with pytest.raises(HTTPException) as timeout:
        inference.run_landcover_inference(
            source_path,
            classification_path,
            confidence_path,
            configuration=configuration,
            deadline_check=http_timeout,
        )
    assert timeout.value.status_code == 504
    assert not classification_path.exists()
    assert not confidence_path.exists()


def test_api_models_inference_result_and_ownership_are_integrated(ai_backend):
    configuration, _repository, manager, _output_dir = ai_backend
    readiness = client.get("/api/imagery/ai-models")
    assert readiness.status_code == 200
    assert readiness.json()["models"][0]["status"] == "ready"
    assert readiness.json()["candidates"][0]["status"] == "model_not_ready"
    scene = ingest_synthetic_scene()
    assert scene["analysis"] == "imagery_ingestion"
    response = client.post(
        f"/api/imagery/scenes/{scene['id']}/ai-inference",
        json={"model_id": inference.LANDCOVER_MODEL_ID},
    )

    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert result["analysis"] == "ai_landcover"
    assert result["result"]["valid_pixels"] == 47
    assert {item["artifact_name"] for item in result["artifacts"]} == {
        "classification.tif",
        "confidence.tif",
    }
    fetched = client.get(f"/api/imagery/ai-inference/{result['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["result"]["model"]["version"] == "test-fixture-1"
    assert manager.get_analysis(result["id"], owner_id="user-1")["status"] == "completed"

    app.dependency_overrides[routes.current_user] = lambda: AuthenticatedUser("user-2")
    assert client.get(f"/api/imagery/ai-inference/{result['id']}").status_code == 404
    assert client.post(
        f"/api/imagery/scenes/{scene['id']}/ai-inference",
        json={"model_id": inference.LANDCOVER_MODEL_ID},
    ).status_code == 404


def test_inference_api_requires_authentication_and_reports_missing_model(
    ai_backend, monkeypatch
):
    configuration, _repository, _manager, _output_dir = ai_backend
    scene = ingest_synthetic_scene()
    app.dependency_overrides[routes.current_user] = lambda: None
    response = client.post(
        f"/api/imagery/scenes/{scene['id']}/ai-inference",
        json={"model_id": inference.LANDCOVER_MODEL_ID},
    )
    assert response.status_code == 401

    app.dependency_overrides[routes.current_user] = lambda: AuthenticatedUser("user-1")
    configuration.ai_landcover_manifest_path.unlink()
    response = client.post(
        f"/api/imagery/scenes/{scene['id']}/ai-inference",
        json={"model_id": inference.LANDCOVER_MODEL_ID},
    )
    assert response.status_code == 503
    assert "model manifest" in response.json()["detail"]


def test_failed_api_inference_cleans_temporary_staging(ai_backend, monkeypatch):
    _configuration, _repository, _manager, _output_dir = ai_backend
    scene = ingest_synthetic_scene()
    temporary_directories = []
    original_temporary_directory = routes.tempfile.TemporaryDirectory

    def track_temporary_directory(*args, **kwargs):
        directory = original_temporary_directory(*args, **kwargs)
        temporary_directories.append(directory)
        return directory

    def fail_inference(*_args, **_kwargs):
        raise inference.InferenceExecutionError()

    monkeypatch.setattr(routes.tempfile, "TemporaryDirectory", track_temporary_directory)
    monkeypatch.setattr(
        routes.geoai_inference,
        "run_landcover_inference",
        fail_inference,
    )
    response = client.post(
        f"/api/imagery/scenes/{scene['id']}/ai-inference",
        json={"model_id": inference.LANDCOVER_MODEL_ID},
    )

    assert response.status_code == 422
    assert temporary_directories
    assert all(not Path(directory.name).exists() for directory in temporary_directories)
