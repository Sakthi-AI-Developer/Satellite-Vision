from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from config import settings
from geoai import inference
from geoai.train_landcover import TrainingConfig, TrainingError, train_landcover_model
from geoai.training_data import RasterTrainingPair, prepare_training_dataset

TEST_ONLY_NOTE = (
    "Synthetic fixtures only; this statement is not evidence for real geographic independence."
)


def write_region(directory: Path, region_id: str) -> RasterTrainingPair:
    directory.mkdir(parents=True)
    image_path = directory / f"{region_id}-image.tif"
    label_path = directory / f"{region_id}-labels.tif"
    image = np.zeros((4, 8, 8), dtype=np.uint16)
    labels = np.where(np.indices((8, 8))[1] < 4, 10, 20).astype(np.uint8)
    image[0, :, :4] = 1_000
    image[1, :, :4] = 2_000
    image[2, :, :4] = 3_000
    image[3, :, :4] = 1_000
    image[0, :, 4:] = 8_000
    image[1, :, 4:] = 7_000
    image[2, :, 4:] = 6_000
    image[3, :, 4:] = 9_000
    image[:, 0, 0] = 0
    labels[0, 0] = 0
    transform = from_origin(500_000, 1_500_000, 10, 10)
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=4,
        dtype="uint16",
        nodata=0,
        crs="EPSG:32643",
        transform=transform,
    ) as destination:
        destination.write(image)
        for index, name in enumerate(("B02", "B03", "B04", "B08"), start=1):
            destination.set_band_description(index, name)
    with rasterio.open(
        label_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="uint8",
        nodata=0,
        crs="EPSG:32643",
        transform=transform,
    ) as destination:
        destination.write(labels, 1)
    return RasterTrainingPair(image_path, label_path, region_id, "2021-06-15")


def prepare_dataset(tmp_path: Path, *, geographic_evidence: str | None) -> Path:
    pairs = [
        write_region(tmp_path / "source" / region_id, region_id)
        for region_id in ("site-a", "site-b", "site-c")
    ]
    dataset = tmp_path / "prepared-dataset"
    prepare_training_dataset(
        pairs,
        dataset,
        chip_size=4,
        geographic_independence_evidence=geographic_evidence,
    )
    return dataset


def test_training_refuses_unattested_geographic_splits(tmp_path):
    dataset = prepare_dataset(tmp_path, geographic_evidence=None)

    with pytest.raises(TrainingError, match="metadata"):
        train_landcover_model(dataset, tmp_path / "model")

    assert not (tmp_path / "model").exists()


def test_training_rejects_changed_chip_after_dataset_index_creation(tmp_path):
    dataset = prepare_dataset(tmp_path, geographic_evidence=TEST_ONLY_NOTE)
    index_path = dataset / "dataset-index.jsonl"
    entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    sample_path = dataset / entries[0]["sample"]
    sample_path.write_bytes(sample_path.read_bytes() + b"tamper")

    with pytest.raises(TrainingError, match="checksum"):
        train_landcover_model(dataset, tmp_path / "model")

    assert not (tmp_path / "model").exists()


def test_training_writes_reproducible_weak_label_metrics_and_inference_manifest(tmp_path):
    dataset = prepare_dataset(tmp_path, geographic_evidence=TEST_ONLY_NOTE)
    model_directory = tmp_path / "private-model"
    config = TrainingConfig(
        seed=9,
        max_train_pixels=32,
        max_pixels_per_class=16,
        max_iterations=5,
        max_leaf_nodes=3,
        min_samples_leaf=2,
    )

    report = train_landcover_model(dataset, model_directory, config=config)

    assert report["framework"] == "scikit-learn HistGradientBoostingClassifier"
    assert report["training_config"] == {
        "seed": 9,
        "max_train_pixels": 32,
        "max_pixels_per_class": 16,
        "max_iterations": 5,
        "learning_rate": 0.1,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
    }
    for key in ("validation_metrics", "heldout_test_metrics"):
        metrics = report[key]
        assert metrics["comparison"] == "agreement_with_esa_worldcover_weak_labels"
        assert metrics["valid_pixels"] == 63
        assert 0 <= metrics["overall_accuracy"] <= 1
        assert 0 <= metrics["mean_iou_present_classes"] <= 1
        assert metrics["per_class"]["10"]["support"] > 0
        assert metrics["per_class"]["20"]["support"] > 0
        assert sum(
            sum(row.values()) for row in metrics["confusion_matrix"].values()
        ) == 63
    assert "not independent scientific accuracy" in report["evaluation_warning"]
    assert report["regions_by_split"]["train"]
    assert report["regions_by_split"]["validation"]
    assert report["regions_by_split"]["test"]

    manifest = json.loads(
        (model_directory / "sentinel2-landcover.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_sha256"] == report["model_sha256"]
    assert manifest["weak_supervision_only"] is True
    assert manifest["class_mapping"] == {"10": "tree_cover", "20": "shrubland"} | {
        "30": "grassland",
        "40": "cropland",
        "50": "built_up",
        "60": "bare_or_sparse_vegetation",
        "70": "snow_and_ice",
        "80": "permanent_water_bodies",
        "90": "herbaceous_wetland",
        "95": "mangroves",
        "100": "moss_and_lichen",
    }
    assert manifest["deployment_approved"] is False

    local_settings = replace(
        settings,
        ai_model_dir=model_directory,
        ai_landcover_manifest_path=model_directory / "sentinel2-landcover.json",
        max_ai_model_bytes=8 * 1024 * 1024,
        max_ai_inference_pixels=1_000,
        max_raster_bytes=8 * 1024 * 1024,
        max_artifact_bytes=8 * 1024 * 1024,
    )
    status = inference.list_models(local_settings)["models"][0]
    assert status["status"] == "model_not_ready"
    assert "not approved" in status["readiness_detail"]
    manifest["deployment_approved"] = True
    (model_directory / "sentinel2-landcover.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assert inference.list_models(local_settings)["models"][0]["status"] == "ready"
    source_path = next((dataset.parent / "source").glob("site-*/site-*-image.tif"))
    classification = tmp_path / "class-index.tif"
    confidence = tmp_path / "confidence.tif"
    result = inference.run_landcover_inference(
        source_path,
        classification,
        confidence,
        configuration=local_settings,
    )
    assert result["status"] == "completed"
    assert {entry["label"] for entry in result["classes"]} == {"tree_cover", "shrubland"}
    with rasterio.open(source_path) as source, rasterio.open(classification) as output:
        assert output.crs == source.crs
        assert output.transform == source.transform
        assert output.dataset_mask()[0, 0] == 0

    artifact_path = model_directory / manifest["artifact"]
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
    status = inference.list_models(local_settings)["models"][0]
    assert status["status"] == "model_not_ready"
    assert "checksum" in status["readiness_detail"]


@pytest.mark.parametrize(
    "config",
    [
        TrainingConfig(max_train_pixels=0),
        TrainingConfig(max_pixels_per_class=0),
        TrainingConfig(max_iterations=501),
        TrainingConfig(learning_rate=float("nan")),
    ],
)
def test_training_rejects_invalid_configuration(tmp_path, config):
    with pytest.raises(TrainingError):
        train_landcover_model(tmp_path / "missing-data", tmp_path / "model", config=config)
