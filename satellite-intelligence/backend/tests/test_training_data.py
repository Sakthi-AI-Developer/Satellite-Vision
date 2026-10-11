import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geoai.training_data import (
    RasterTrainingPair,
    TrainingDataError,
    prepare_training_dataset,
)


def write_pair(
    directory: Path,
    region_id: str,
    *,
    image_transform=None,
    label_transform=None,
    band_names=("B02", "B03", "B04", "B08"),
    label_codes=None,
) -> RasterTrainingPair:
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"{region_id}-image.tif"
    label_path = directory / f"{region_id}-worldcover.tif"
    transform = from_origin(500_000, 1_500_000, 10, 10)
    image_values = np.full((4, 8, 8), 2_000, dtype=np.uint16)
    image_values[:, 0, 0] = 0
    labels = np.where(np.indices((8, 8))[1] < 4, 10, 20).astype(np.uint8)
    labels[0, 0] = 0
    if label_codes is not None:
        labels[1, 1] = label_codes

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
        transform=image_transform or transform,
    ) as raster:
        raster.write(image_values)
        for index, name in enumerate(band_names, start=1):
            raster.set_band_description(index, name)
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
        transform=label_transform or transform,
    ) as raster:
        raster.write(labels, 1)
    return RasterTrainingPair(image_path, label_path, region_id, "2021-06-15")


def three_region_pairs(tmp_path: Path):
    return [
        write_pair(tmp_path / region_id, region_id)
        for region_id in ("region-a", "region-b", "region-c")
    ]


def test_preparation_normalizes_data_preserves_labels_and_keeps_regions_separate(tmp_path):
    pairs = three_region_pairs(tmp_path / "inputs")
    output = tmp_path / "private-training" / "prepared-v1"

    result = prepare_training_dataset(pairs, output, chip_size=4, seed=8)

    assert result["sample_count"] == 12
    assert set(result["samples_by_split"]) == {"train", "validation", "test"}
    assert all(count > 0 for count in result["samples_by_split"].values())
    assert result["normalization"] == "uint16_reflectance_divide_by_10000"
    assert result["label_license"] == "CC-BY-4.0"
    assert "not independent ground truth" in result["training_label_warning"]

    entries = [
        json.loads(line)
        for line in (output / "dataset-index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    region_splits = {}
    has_nodata_sample = False
    for entry in entries:
        region_splits.setdefault(entry["region_id"], set()).add(entry["split"])
        with np.load(output / entry["sample"]) as sample:
            assert sample["image"].shape == (4, 4, 4)
            assert np.allclose(sample["image"][:, 1:, 1:], 0.2)
            assert sample["labels"].shape == (4, 4)
            assert np.count_nonzero(sample["labels"]) == entry["valid_pixels"]
            assert np.all(np.isin(sample["labels"], (0, 10, 20)))
            if entry["valid_pixels"] == 15:
                assert sample["labels"][0, 0] == 0
                assert np.all(sample["image"][:, 0, 0] == 0)
                has_nodata_sample = True
            else:
                assert entry["valid_pixels"] == 16
        assert entry["crs"] == "EPSG:32643"
        assert len(entry["image_sha256"]) == 64
        assert len(entry["label_sha256"]) == 64
        assert len(entry["sample_sha256"]) == 64
    assert all(len(splits) == 1 for splits in region_splits.values())
    assert has_nodata_sample
    assert (output / "metadata.json").is_file()


@pytest.mark.parametrize(
    "pair_options,match",
    [
        ({"image_transform": from_origin(500_000, 1_500_000, 20, 20)}, "identical CRS"),
        ({"band_names": ("B03", "B02", "B04", "B08")}, "ordered B02"),
        ({"label_codes": 11}, "class codes"),
    ],
)
def test_preparation_rejects_misaligned_or_incompatible_rasters(tmp_path, pair_options, match):
    pair = write_pair(tmp_path / "invalid", "invalid", **pair_options)
    pairs = [
        pair,
        write_pair(tmp_path / "second", "second"),
        write_pair(tmp_path / "third", "third"),
    ]

    with pytest.raises(TrainingDataError, match=match):
        prepare_training_dataset(pairs, tmp_path / "prepared", chip_size=4)

    assert not (tmp_path / "prepared").exists()


def test_preparation_requires_2021_labels_and_distinct_regions(tmp_path):
    first = write_pair(tmp_path / "first", "same")
    second = write_pair(tmp_path / "second", "same")
    third = write_pair(tmp_path / "third", "same")
    pairs = [first, second, third]
    pairs[0] = RasterTrainingPair(
        first.image_path,
        first.label_path,
        first.region_id,
        "2020-06-15",
    )
    with pytest.raises(TrainingDataError, match="2021 Sentinel-2"):
        prepare_training_dataset(pairs, tmp_path / "bad-date", chip_size=4)

    invalid_calendar_date = [
        RasterTrainingPair(pair.image_path, pair.label_path, pair.region_id, "2021-02-30")
        for pair in [first, second, third]
    ]
    with pytest.raises(TrainingDataError, match="YYYY-MM-DD"):
        prepare_training_dataset(
            invalid_calendar_date,
            tmp_path / "invalid-calendar-date",
            chip_size=4,
        )

    valid_dates = [
        RasterTrainingPair(pair.image_path, pair.label_path, pair.region_id, "2021-06-15")
        for pair in [first, second, third]
    ]
    with pytest.raises(TrainingDataError, match="three distinct geographic"):
        prepare_training_dataset(valid_dates, tmp_path / "bad-split", chip_size=4)


def test_preparation_never_overwrites_existing_or_repository_data(tmp_path):
    pairs = three_region_pairs(tmp_path / "inputs")
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(TrainingDataError, match="must not already exist"):
        prepare_training_dataset(pairs, existing, chip_size=4)
    with pytest.raises(TrainingDataError, match="outside the project repository"):
        prepare_training_dataset(pairs, Path(__file__).resolve().parents[2], chip_size=4)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
