from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from geoai.training_data import (
    SENTINEL2_BANDS,
    WORLD_COVER_CLASSES,
    WORLD_COVER_LABEL_SOURCE,
    WORLD_COVER_LICENSE,
    WORLD_COVER_NODATA,
)

DEFAULT_MAX_TRAIN_PIXELS = 50_000
MAX_TRAIN_PIXELS = 100_000
DEFAULT_MAX_PIXELS_PER_CLASS = 5_000
MAX_EVALUATION_PIXELS = 1_000_000
MODEL_ARTIFACT_NAME = "sentinel2-landcover.joblib"
MODEL_MANIFEST_NAME = "sentinel2-landcover.json"
TRAINING_REPORT_NAME = "training-report.json"


class TrainingError(ValueError):
    """A safe explanation for invalid training inputs or failed training."""


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    max_train_pixels: int = DEFAULT_MAX_TRAIN_PIXELS
    max_pixels_per_class: int = DEFAULT_MAX_PIXELS_PER_CLASS
    max_iterations: int = 100
    learning_rate: float = 0.1
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: TrainingConfig) -> None:
    if not 0 <= config.seed <= 4_294_967_295:
        raise TrainingError("The random seed must be an unsigned 32-bit integer.")
    if not 1 <= config.max_train_pixels <= MAX_TRAIN_PIXELS:
        raise TrainingError(f"Training pixel cap must be between 1 and {MAX_TRAIN_PIXELS}.")
    if not 1 <= config.max_pixels_per_class <= config.max_train_pixels:
        raise TrainingError("Per-class pixel cap must be positive and no larger than the total cap.")
    if not 1 <= config.max_iterations <= 500:
        raise TrainingError("Training iterations must be between 1 and 500.")
    if not math.isfinite(config.learning_rate) or not 0 < config.learning_rate <= 1:
        raise TrainingError("Learning rate must be finite and in the interval (0, 1].")
    if not 2 <= config.max_leaf_nodes <= 63:
        raise TrainingError("Maximum leaf nodes must be between 2 and 63.")
    if not 2 <= config.min_samples_leaf <= 1_000:
        raise TrainingError("Minimum samples per leaf must be between 2 and 1000.")
    if (
        not math.isfinite(config.l2_regularization)
        or config.l2_regularization < 0
        or config.l2_regularization > 1_000_000
    ):
        raise TrainingError("L2 regularization must be finite and between 0 and 1000000.")


def _load_dataset(dataset_directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    supplied_root = dataset_directory.expanduser()
    if supplied_root.is_symlink():
        raise TrainingError("The prepared dataset directory may not be a symbolic link.")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise TrainingError("The prepared dataset directory must be an existing local directory.")
    metadata_path = root / "metadata.json"
    index_path = root / "dataset-index.jsonl"
    if (
        metadata_path.is_symlink()
        or index_path.is_symlink()
        or not metadata_path.is_file()
        or not index_path.is_file()
        or metadata_path.stat().st_size > 128 * 1024
        or index_path.stat().st_size > 16 * 1024 * 1024
    ):
        raise TrainingError("The prepared dataset must contain metadata.json and dataset-index.jsonl.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_entries = index_path.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in raw_entries if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError("The prepared dataset index or metadata is invalid.") from exc

    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset") != WORLD_COVER_LABEL_SOURCE
        or metadata.get("label_license") != WORLD_COVER_LICENSE
        or metadata.get("input_bands") != list(SENTINEL2_BANDS)
        or metadata.get("normalization") != "uint16_reflectance_divide_by_10000"
        or metadata.get("nodata_code") != WORLD_COVER_NODATA
        or metadata.get("label_codes")
        != {str(code): name for code, name in WORLD_COVER_CLASSES.items()}
        or metadata.get("geographic_independence_attested") is not True
        or not isinstance(metadata.get("geographic_independence_evidence"), str)
        or not metadata["geographic_independence_evidence"].strip()
        or not isinstance(entries, list)
        or not entries
        or len(entries) > 5_000
    ):
        raise TrainingError("The prepared dataset metadata does not match the supported contract.")

    region_splits: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TrainingError("The prepared dataset index contains an invalid sample entry.")
        region_id = entry.get("region_id")
        split = entry.get("split")
        if (
            not isinstance(region_id, str)
            or not region_id.strip()
            or split not in {"train", "validation", "test"}
        ):
            raise TrainingError("Each sample must declare a region ID and a supported split.")
        previous_split = region_splits.setdefault(region_id, split)
        if previous_split != split:
            raise TrainingError("A geographic region appears in more than one data split.")
        sample_name = entry.get("sample")
        if (
            not isinstance(sample_name, str)
            or Path(sample_name).name != sample_name
            or not sample_name.endswith(".npz")
        ):
            raise TrainingError("Sample files must be relative NPZ files inside the dataset directory.")
        sample_path = root / sample_name
        if (
            sample_path.is_symlink()
            or not sample_path.is_file()
            or sample_path.stat().st_size > 2 * 1024 * 1024
        ):
            raise TrainingError("A prepared sample file is missing or is not a regular file.")
        if (
            entry.get("bands") != list(SENTINEL2_BANDS)
            or entry.get("normalization") != "uint16_reflectance_divide_by_10000"
            or not isinstance(entry.get("valid_pixels"), int)
            or isinstance(entry.get("valid_pixels"), bool)
            or entry["valid_pixels"] < 1
            or not isinstance(entry.get("image_sha256"), str)
            or len(entry["image_sha256"]) != 64
            or not isinstance(entry.get("label_sha256"), str)
            or len(entry["label_sha256"]) != 64
            or not isinstance(entry.get("sample_sha256"), str)
            or len(entry["sample_sha256"]) != 64
        ):
            raise TrainingError("A sample entry is missing its preprocessing or provenance fields.")
        if _sha256(sample_path) != entry["sample_sha256"]:
            raise TrainingError("A prepared sample checksum does not match the dataset index.")

    if len(region_splits) < 3 or set(region_splits.values()) != {
        "train",
        "validation",
        "test",
    }:
        raise TrainingError("Training requires distinct regions in train, validation, and test splits.")
    return entries, metadata, index_path


def _read_sample(
    root: Path,
    entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    sample_path = root / entry["sample"]
    try:
        with np.load(sample_path, allow_pickle=False) as sample:
            features = np.asarray(sample["image"])
            labels = np.asarray(sample["labels"])
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise TrainingError("A prepared sample is corrupt or incompatible.") from exc
    if (
        features.ndim != 3
        or features.shape[0] != len(SENTINEL2_BANDS)
        or features.shape[1] < 1
        or features.shape[1] != features.shape[2]
        or features.shape[1] > 256
        or features.dtype != np.float32
        or labels.shape != features.shape[1:]
        or labels.dtype != np.uint8
        or not np.isfinite(features).all()
        or np.any(features < 0)
        or np.any(features > 6.5535)
        or not np.isin(labels, (WORLD_COVER_NODATA, *WORLD_COVER_CLASSES)).all()
        or not np.any(labels != WORLD_COVER_NODATA)
        or np.any(features[:, labels == WORLD_COVER_NODATA] != 0)
    ):
        raise TrainingError("A prepared sample violates the validated chip contract.")
    return features, labels


def _training_quotas(
    class_counts: dict[int, int],
    config: TrainingConfig,
) -> dict[int, int]:
    present = [code for code in WORLD_COVER_CLASSES if class_counts[code] > 0]
    if len(present) < 2:
        raise TrainingError("Training requires valid pixels from at least two land-cover classes.")
    per_class_caps = {
        code: min(class_counts[code], config.max_pixels_per_class)
        for code in present
    }
    total = sum(per_class_caps.values())
    if total <= config.max_train_pixels:
        return per_class_caps

    quotas = {
        code: config.max_train_pixels * per_class_caps[code] // total
        for code in present
    }
    remainder = config.max_train_pixels - sum(quotas.values())
    order = sorted(
        present,
        key=lambda code: (
            -(config.max_train_pixels * per_class_caps[code] % total),
            code,
        ),
    )
    for code in order:
        if remainder == 0:
            break
        if quotas[code] < per_class_caps[code]:
            quotas[code] += 1
            remainder -= 1
    return quotas


def _sample_training_pixels(
    root: Path,
    entries: list[dict[str, Any]],
    quotas: dict[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_entries = [entry for entry in entries if entry["split"] == "train"]
    counts_by_entry: list[dict[int, int]] = []
    totals = {code: 0 for code in WORLD_COVER_CLASSES}
    for entry in train_entries:
        _, labels = _read_sample(root, entry)
        counts = {
            code: int(np.count_nonzero(labels == code))
            for code in WORLD_COVER_CLASSES
        }
        counts_by_entry.append(counts)
        for code, count in counts.items():
            totals[code] += count

    per_entry_quotas: list[dict[int, int]] = [
        {code: 0 for code in WORLD_COVER_CLASSES} for _ in train_entries
    ]
    for code, quota in quotas.items():
        exact = [
            quota * counts_by_entry[index][code] / totals[code]
            for index in range(len(train_entries))
        ]
        allocated = [math.floor(value) for value in exact]
        remainder = quota - sum(allocated)
        order = sorted(
            range(len(train_entries)),
            key=lambda index: (-(exact[index] - allocated[index]), index),
        )
        for index in order:
            if remainder == 0:
                break
            if allocated[index] < counts_by_entry[index][code]:
                allocated[index] += 1
                remainder -= 1
        for index, count in enumerate(allocated):
            per_entry_quotas[index][code] = count

    rng = np.random.default_rng(seed)
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for entry, entry_quotas in zip(train_entries, per_entry_quotas):
        features, labels = _read_sample(root, entry)
        flattened_labels = labels.reshape(-1)
        flattened_features = np.moveaxis(features, 0, -1).reshape(-1, len(SENTINEL2_BANDS))
        for code, quota in entry_quotas.items():
            if quota == 0:
                continue
            indexes = np.flatnonzero(flattened_labels == code)
            selected = (
                indexes
                if quota == len(indexes)
                else rng.choice(indexes, size=quota, replace=False)
            )
            feature_batches.append(flattened_features[selected])
            label_batches.append(np.full(quota, code, dtype=np.uint8))
    if not feature_batches:
        raise TrainingError("No valid training pixels were selected.")
    return np.concatenate(feature_batches), np.concatenate(label_batches)


def _evaluate(
    estimator: HistGradientBoostingClassifier,
    root: Path,
    entries: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    split_entries = [entry for entry in entries if entry["split"] == split]
    pixel_count = sum(int(entry["valid_pixels"]) for entry in split_entries)
    if not split_entries or not 0 < pixel_count <= MAX_EVALUATION_PIXELS:
        raise TrainingError(
            f"The {split} split must contain between 1 and {MAX_EVALUATION_PIXELS} valid pixels."
        )
    class_codes = list(WORLD_COVER_CLASSES)
    matrix = np.zeros((len(class_codes), len(class_codes)), dtype=np.int64)
    batch_size = 8_192
    processed = 0
    for entry in split_entries:
        features, labels = _read_sample(root, entry)
        flat_labels = labels.reshape(-1)
        valid_indexes = np.flatnonzero(flat_labels != WORLD_COVER_NODATA)
        flat_features = np.moveaxis(features, 0, -1).reshape(-1, len(SENTINEL2_BANDS))
        for start in range(0, len(valid_indexes), batch_size):
            batch_indexes = valid_indexes[start : start + batch_size]
            truth = flat_labels[batch_indexes].astype(np.int64)
            prediction = np.asarray(estimator.predict(flat_features[batch_indexes]), dtype=np.int64)
            matrix += confusion_matrix(truth, prediction, labels=class_codes)
            processed += len(truth)

    if processed != pixel_count:
        raise TrainingError("The evaluation index pixel count does not match its sample rasters.")
    true_positive = np.diag(matrix)
    predicted_count = matrix.sum(axis=0)
    actual_count = matrix.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(len(class_codes), dtype=np.float64),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        actual_count,
        out=np.zeros(len(class_codes), dtype=np.float64),
        where=actual_count > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(class_codes), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    union = actual_count + predicted_count - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.zeros(len(class_codes), dtype=np.float64),
        where=union > 0,
    )
    matrix_dict = {
        str(actual): {
            str(predicted): int(matrix[row, column])
            for column, predicted in enumerate(class_codes)
        }
        for row, actual in enumerate(class_codes)
    }
    metrics = {
        "comparison": "agreement_with_esa_worldcover_weak_labels",
        "split": split,
        "valid_pixels": processed,
        "overall_accuracy": float(true_positive.sum() / processed),
        "mean_iou_present_classes": float(iou[actual_count > 0].mean()),
        "confusion_matrix": matrix_dict,
        "per_class": {
            str(code): {
                "name": WORLD_COVER_CLASSES[code],
                "support": int(actual_count[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "iou": float(iou[index]),
            }
            for index, code in enumerate(class_codes)
        },
    }
    return metrics


def train_landcover_model(
    dataset_directory: Path,
    output_directory: Path,
    *,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, Any]:
    """Train a bounded pixel-wise classifier and evaluate only on disjoint region splits."""
    _validate_config(config)
    supplied_target = output_directory.expanduser()
    if supplied_target.is_symlink():
        raise TrainingError("The model output directory may not be a symbolic link.")
    target = supplied_target.resolve()
    project_root = _project_root()
    if _within(target, project_root):
        raise TrainingError("Model artifacts and reports must be stored outside the project repository.")
    if target.exists() or target.is_symlink():
        raise TrainingError("The model output directory must not already exist.")

    entries, dataset_metadata, index_path = _load_dataset(dataset_directory)
    root = dataset_directory.expanduser().resolve()
    region_splits = {
        entry["region_id"]: entry["split"]
        for entry in entries
    }
    counts = {code: 0 for code in WORLD_COVER_CLASSES}
    for entry in entries:
        _, labels = _read_sample(root, entry)
        if int(np.count_nonzero(labels != WORLD_COVER_NODATA)) != int(entry["valid_pixels"]):
            raise TrainingError("A sample valid-pixel count does not match the training index.")
        if entry["split"] == "train":
            for code in WORLD_COVER_CLASSES:
                counts[code] += int(np.count_nonzero(labels == code))
    quotas = _training_quotas(counts, config)
    features, labels = _sample_training_pixels(root, entries, quotas, config.seed)
    class_sample_counts = {
        int(code): int(np.count_nonzero(labels == code))
        for code in np.unique(labels)
    }
    sample_weights = np.asarray(
        [
            len(labels) / (len(class_sample_counts) * class_sample_counts[int(code)])
            for code in labels
        ],
        dtype=np.float64,
    )
    estimator = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.learning_rate,
        max_iter=config.max_iterations,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.seed,
    )
    try:
        estimator.fit(features, labels, sample_weight=sample_weights)
    except (ValueError, MemoryError) as exc:
        raise TrainingError("The bounded training run could not fit the selected pixel samples.") from exc

    validation = _evaluate(estimator, root, entries, "validation")
    test = _evaluate(estimator, root, entries, "test")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sentinel2-model-training-", dir=target.parent))
    try:
        artifact_path = staging / MODEL_ARTIFACT_NAME
        joblib.dump(estimator, artifact_path, compress=3)
        artifact_sha256 = _sha256(artifact_path)
        attribution = (
            "Contains ESA WorldCover 2021 v200 labels, accessed via the AWS Open Data Registry "
            "(CC-BY-4.0); DOI: 10.5281/zenodo.7254221. Sentinel-2 input data are Copernicus "
            "Sentinel data; see the Sentinel Data Legal Notice."
        )
        manifest = {
            "name": "Satellite Vision WorldCover weak-label pixel classifier",
            "version": "1.0.0",
            "license": "No separate model-weight license determination; see source-label terms.",
            "source": "Satellite Vision offline training pipeline",
            "attribution": attribution,
            "deployment_approved": False,
            "artifact": MODEL_ARTIFACT_NAME,
            "artifact_sha256": artifact_sha256,
            "bands": list(SENTINEL2_BANDS),
            "normalization": "divide_by_10000",
            "class_mapping": {
                str(code): name for code, name in WORLD_COVER_CLASSES.items()
            },
            "label_source": WORLD_COVER_LABEL_SOURCE,
            "label_license": WORLD_COVER_LICENSE,
            "label_attribution": dataset_metadata.get("label_attribution"),
            "weak_supervision_only": True,
            "training_index_sha256": _sha256(index_path),
            "training_metadata_sha256": _sha256(root / "metadata.json"),
            "training_sample_count": len(entries),
            "training_valid_pixel_count": sum(counts.values()),
            "selected_training_pixel_count": int(len(labels)),
            "training_config": asdict(config),
            "sklearn_version": sklearn.__version__,
            "random_seed": config.seed,
        }
        report = {
            "model": MODEL_ARTIFACT_NAME,
            "model_sha256": artifact_sha256,
            "framework": "scikit-learn HistGradientBoostingClassifier",
            "framework_version": sklearn.__version__,
            "training_config": asdict(config),
            "training_class_sample_counts": {
                str(code): count for code, count in class_sample_counts.items()
            },
            "regions_by_split": {
                split: sorted(region for region, assigned in region_splits.items() if assigned == split)
                for split in ("train", "validation", "test")
            },
            "validation_metrics": validation,
            "heldout_test_metrics": test,
            "evaluation_warning": (
                "These values measure agreement with ESA WorldCover 2021 v200 weak labels only; "
                "they are not independent scientific accuracy."
            ),
        }
        (staging / MODEL_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / TRAINING_REPORT_NAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise TrainingError("The model output directory became unavailable.")
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {**report, "output_directory": str(target)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a local Sentinel-2 pixel classifier from prepared chips."
    )
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument(
        "--max-train-pixels",
        type=int,
        default=TrainingConfig.max_train_pixels,
    )
    parser.add_argument(
        "--max-pixels-per-class",
        type=int,
        default=TrainingConfig.max_pixels_per_class,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=TrainingConfig.max_iterations,
    )
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--max-leaf-nodes", type=int, default=TrainingConfig.max_leaf_nodes)
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=TrainingConfig.min_samples_leaf,
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=TrainingConfig.l2_regularization,
    )
    arguments = parser.parse_args(argv)
    config = TrainingConfig(
        seed=arguments.seed,
        max_train_pixels=arguments.max_train_pixels,
        max_pixels_per_class=arguments.max_pixels_per_class,
        max_iterations=arguments.max_iterations,
        learning_rate=arguments.learning_rate,
        max_leaf_nodes=arguments.max_leaf_nodes,
        min_samples_leaf=arguments.min_samples_leaf,
        l2_regularization=arguments.l2_regularization,
    )
    try:
        report = train_landcover_model(
            arguments.dataset_directory,
            arguments.output_directory,
            config=config,
        )
    except TrainingError as exc:
        parser.exit(2, f"training error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
