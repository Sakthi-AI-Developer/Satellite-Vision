from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.io import DatasetReader
from rasterio.windows import Window

SENTINEL2_BANDS = ("B02", "B03", "B04", "B08")
WORLD_COVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_or_sparse_vegetation",
    70: "snow_and_ice",
    80: "permanent_water_bodies",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_and_lichen",
}
WORLD_COVER_NODATA = 0
WORLD_COVER_LABEL_SOURCE = "ESA WorldCover 2021 v200"
WORLD_COVER_LICENSE = "CC-BY-4.0"
DEFAULT_CHIP_SIZE = 128
MAX_CHIP_SIZE = 256
DEFAULT_MAX_SAMPLES = 5_000


class TrainingDataError(ValueError):
    """A safe explanation for an invalid local training raster or dataset."""


@dataclass(frozen=True)
class RasterTrainingPair:
    image_path: Path
    label_path: Path
    region_id: str
    acquisition_date: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_pair(pair: RasterTrainingPair) -> tuple[Path, Path]:
    image_path = Path(pair.image_path).expanduser().resolve()
    label_path = Path(pair.label_path).expanduser().resolve()
    if not image_path.is_file() or not label_path.is_file():
        raise TrainingDataError("Training image and label rasters must be local files.")
    if not pair.region_id.strip() or len(pair.region_id) > 160:
        raise TrainingDataError("Each raster pair must have a non-empty region identifier.")
    try:
        acquired = date.fromisoformat(pair.acquisition_date)
    except (TypeError, ValueError) as exc:
        raise TrainingDataError("Acquisition dates must use YYYY-MM-DD format.") from exc
    if acquired.isoformat() != pair.acquisition_date:
        raise TrainingDataError("Acquisition dates must use YYYY-MM-DD format.")
    if acquired.year != 2021:
        raise TrainingDataError(
            "ESA WorldCover 2021 v200 preparation requires a 2021 Sentinel-2 acquisition date."
        )
    return image_path, label_path


def _split_regions(region_ids: set[str], seed: int) -> dict[str, str]:
    if len(region_ids) < 3:
        raise TrainingDataError(
            "At least three distinct geographic region IDs are required for spatial train, "
            "validation, and test splits."
        )
    ordered = sorted(
        region_ids,
        key=lambda region: hashlib.sha256(f"{seed}:{region}".encode("utf-8")).digest(),
    )
    train_end = min(len(ordered) - 2, max(1, int(len(ordered) * 0.70)))
    validation_end = min(
        len(ordered) - 1,
        max(train_end + 1, round(len(ordered) * 0.85)),
    )
    return {
        region: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, region in enumerate(ordered)
    }


def _open_aligned_pair(
    image_path: Path,
    label_path: Path,
) -> tuple[DatasetReader, DatasetReader]:
    try:
        image = rasterio.open(image_path)
    except (OSError, RasterioError) as exc:
        raise TrainingDataError("A local training raster could not be opened.") from exc

    try:
        labels = rasterio.open(label_path)
    except (OSError, RasterioError) as exc:
        image.close()
        raise TrainingDataError("A local training raster could not be opened.") from exc

    try:
        if image.driver != "GTiff" or labels.driver != "GTiff":
            raise TrainingDataError("Training imagery and labels must be GeoTIFF files.")
        if image.count != len(SENTINEL2_BANDS):
            raise TrainingDataError("Training imagery must contain exactly four Sentinel-2 bands.")
        band_names = tuple((name or "").strip().upper() for name in image.descriptions)
        if band_names != SENTINEL2_BANDS:
            raise TrainingDataError(
                "Training imagery bands must be labelled and ordered B02, B03, B04, B08."
            )
        if any(dtype != "uint16" for dtype in image.dtypes):
            raise TrainingDataError("Sentinel-2 source bands must use unsigned 16-bit reflectance.")
        if (
            labels.count != 1
            or labels.dtypes[0] not in {"uint8", "uint16"}
        ):
            raise TrainingDataError(
                "WorldCover labels must be a single-band unsigned 8-bit or 16-bit GeoTIFF."
            )
        if image.crs is None or labels.crs is None:
            raise TrainingDataError("Both training rasters must declare a CRS.")
        if (
            image.width != labels.width
            or image.height != labels.height
            or image.crs != labels.crs
            or image.transform != labels.transform
        ):
            raise TrainingDataError(
                "Label and Sentinel-2 rasters must have identical CRS, dimensions, and pixel alignment."
            )
        return image, labels
    except Exception:
        image.close()
        labels.close()
        raise


def prepare_training_dataset(
    pairs: list[RasterTrainingPair],
    output_directory: Path,
    *,
    chip_size: int = DEFAULT_CHIP_SIZE,
    seed: int = 42,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    geographic_independence_evidence: str | None = None,
) -> dict[str, Any]:
    """Prepare local, spatially grouped Sentinel-2/WorldCover chips without downloading data."""
    if not pairs:
        raise TrainingDataError("Provide at least three aligned image/label region pairs.")
    if not 1 <= chip_size <= MAX_CHIP_SIZE:
        raise TrainingDataError(f"Chip size must be between 1 and {MAX_CHIP_SIZE} pixels.")
    if not 1 <= max_samples <= DEFAULT_MAX_SAMPLES:
        raise TrainingDataError(
            f"Maximum sample count must be between 1 and {DEFAULT_MAX_SAMPLES}."
        )
    if not 0 <= seed <= 4_294_967_295:
        raise TrainingDataError("The random seed must be an unsigned 32-bit integer.")
    if geographic_independence_evidence is not None and (
        not isinstance(geographic_independence_evidence, str)
        or not geographic_independence_evidence.strip()
        or len(geographic_independence_evidence) > 2_000
    ):
        raise TrainingDataError(
            "Geographic split evidence must be a non-empty statement of at most 2000 characters."
        )

    supplied_target = Path(output_directory).expanduser()
    if supplied_target.is_symlink():
        raise TrainingDataError("The training output directory may not be a symbolic link.")
    target = supplied_target.resolve()
    if _within(target, _project_root()):
        raise TrainingDataError(
            "Prepared training data must be stored outside the project repository."
        )
    if target.exists() or target.is_symlink():
        raise TrainingDataError("The training output directory must not already exist.")

    validated: list[tuple[RasterTrainingPair, Path, Path]] = []
    region_ids: set[str] = set()
    for pair in pairs:
        image_path, label_path = _validate_pair(pair)
        validated.append((pair, image_path, label_path))
        region_ids.add(pair.region_id)
    split_by_region = _split_regions(region_ids, seed)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sentinel2-landcover-prep-", dir=target.parent))
    entries: list[dict[str, Any]] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    class_counts = {str(code): 0 for code in WORLD_COVER_CLASSES}

    try:
        index_path = staging / "dataset-index.jsonl"
        with index_path.open("w", encoding="utf-8", newline="\n") as index_file:
            for pair, image_path, label_path in validated:
                image_sha256 = _sha256(image_path)
                label_sha256 = _sha256(label_path)
                try:
                    image, labels = _open_aligned_pair(image_path, label_path)
                except RasterioError as exc:
                    raise TrainingDataError("A training raster could not be inspected.") from exc
                with image, labels:
                    for row in range(0, image.height - chip_size + 1, chip_size):
                        for column in range(0, image.width - chip_size + 1, chip_size):
                            window = Window.from_slices(
                                (row, row + chip_size),
                                (column, column + chip_size),
                            )
                            image_chip = image.read(
                                indexes=list(range(1, len(SENTINEL2_BANDS) + 1)),
                                window=window,
                                masked=True,
                            )
                            label_chip = labels.read(1, window=window, masked=True)
                            raw_labels = np.asarray(label_chip.data)
                            label_mask = np.ma.getmaskarray(label_chip)
                            unknown = (
                                ~label_mask
                                & (raw_labels != WORLD_COVER_NODATA)
                                & ~np.isin(raw_labels, tuple(WORLD_COVER_CLASSES))
                            )
                            if np.any(unknown):
                                raise TrainingDataError(
                                    "The label raster contains class codes outside the ESA WorldCover 2021 v200 mapping."
                                )
                            valid = (
                                ~np.ma.getmaskarray(image_chip).any(axis=0)
                                & ~label_mask
                                & (raw_labels != WORLD_COVER_NODATA)
                            )
                            if not np.any(valid):
                                continue
                            if len(entries) >= max_samples:
                                raise TrainingDataError(
                                    "Prepared sample count would exceed the configured safety limit."
                                )

                            features = (
                                np.asarray(image_chip.data, dtype=np.float32) / np.float32(10_000)
                            )
                            features[:, ~valid] = 0
                            target_labels = np.where(valid, raw_labels, WORLD_COVER_NODATA).astype(
                                np.uint8
                            )
                            unique, counts = np.unique(target_labels[valid], return_counts=True)
                            sample_name = f"sample-{len(entries) + 1:06d}.npz"
                            np.savez_compressed(
                                staging / sample_name,
                                image=features,
                                labels=target_labels,
                            )
                            sample_sha256 = _sha256(staging / sample_name)
                            counts_by_class = {
                                str(int(code)): int(count)
                                for code, count in zip(unique, counts)
                            }
                            for code, count in counts_by_class.items():
                                class_counts[code] += count

                            entry = {
                                "sample": sample_name,
                                "sample_sha256": sample_sha256,
                                "region_id": pair.region_id,
                                "split": split_by_region[pair.region_id],
                                "acquisition_date": pair.acquisition_date,
                                "image_file": image_path.name,
                                "image_sha256": image_sha256,
                                "label_file": label_path.name,
                                "label_sha256": label_sha256,
                                "crs": image.crs.to_string(),
                                "transform": list(image.window_transform(window).to_gdal()),
                                "bands": list(SENTINEL2_BANDS),
                                "normalization": "uint16_reflectance_divide_by_10000",
                                "valid_pixels": int(np.count_nonzero(valid)),
                                "class_pixel_counts": counts_by_class,
                            }
                            index_file.write(json.dumps(entry, sort_keys=True) + "\n")
                            entries.append(entry)
                            split_counts[entry["split"]] += 1

        if not entries:
            raise TrainingDataError("No valid, non-nodata chips were found in the supplied rasters.")
        if any(split_counts[name] == 0 for name in ("train", "validation", "test")):
            raise TrainingDataError(
                "Each spatial split must contain at least one valid chip; provide valid data in all three splits."
            )

        metadata = {
            "dataset": WORLD_COVER_LABEL_SOURCE,
            "label_license": WORLD_COVER_LICENSE,
            "label_attribution": (
                "ESA WorldCover 2021 v200 (CC-BY-4.0), DOI: 10.5281/zenodo.7254221; "
                "accessed via the AWS Open Data Registry."
            ),
            "label_codes": {
                str(code): name for code, name in WORLD_COVER_CLASSES.items()
            },
            "nodata_code": WORLD_COVER_NODATA,
            "input_bands": list(SENTINEL2_BANDS),
            "normalization": "uint16_reflectance_divide_by_10000",
            "chip_size": chip_size,
            "split_strategy": "all observations sharing a region_id stay in one deterministic split",
            "geographic_independence_attested": geographic_independence_evidence is not None,
            "geographic_independence_evidence": geographic_independence_evidence,
            "seed": seed,
            "sample_count": len(entries),
            "samples_by_split": split_counts,
            "valid_pixels_by_class_code": class_counts,
            "training_label_warning": (
                "WorldCover labels are derived map-product labels, not independent ground truth. "
                "Use an independently licensed reference dataset for final evaluation."
            ),
            "split_warning": (
                "The utility groups region IDs but cannot verify that distinct IDs are spatially "
                "separate. Geographic independence evidence is an operator attestation."
            ),
            "samples": "dataset-index.jsonl",
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise TrainingDataError("The training output directory became unavailable.")
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return {**metadata, "output_directory": str(target)}
