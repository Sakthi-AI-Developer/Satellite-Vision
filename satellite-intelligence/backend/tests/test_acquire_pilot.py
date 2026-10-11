import numpy as np
import pytest

from geoai.acquire_pilot import (
    AcquisitionError,
    _sha256,
    _parse_range_response,
    _validate_item,
    _validate_manifest,
    _validate_worldcover_codes,
    _window_gap_meters,
)
from geoai.training_data import SENTINEL2_BANDS


def make_stac_item():
    band_names = dict(zip(SENTINEL2_BANDS, ("blue", "green", "red", "nir")))
    return {
        "id": "S2B_43PGN_20211224_0_L2A",
        "properties": {
            "datetime": "2021-12-24T05:25:44.093Z",
            "platform": "sentinel-2b",
            "eo:cloud_cover": 0.69,
            "s2:product_uri": "S2B_MSIL2A_20211224T052129_N0301_R019_T43PGN_20211224T080233.SAFE",
        },
        "assets": {
            name: {
                "href": (
                    "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
                    f"sentinel-s2-l2a-cogs/43/P/GN/2021/12/{name}.tif"
                ),
                "gsd": 10,
                "raster:bands": [
                    {
                        "data_type": "uint16",
                        "nodata": 0,
                        "scale": 0.0001,
                        "offset": 0,
                        "spatial_resolution": 10,
                    }
                ],
                "eo:bands": [{"common_name": name}],
            }
            for name in band_names.values()
        },
    }


def test_validates_real_scene_metadata_and_expected_band_assets():
    metadata = _validate_item(make_stac_item(), "S2B_43PGN_20211224_0_L2A")

    assert metadata["acquisition_date"] == "2021-12-24"
    assert metadata["platform"] == "sentinel-2b"
    assert tuple(metadata["assets"]) == SENTINEL2_BANDS
    assert metadata["assets"]["B02"].endswith("/blue.tif")
    assert metadata["band_metadata"]["B02"]["scale"] == 0.0001


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda item: item.update(id="unexpected"), "unexpected Sentinel-2 item"),
        (
            lambda item: item["properties"].update(datetime="2020-12-24T00:00:00Z"),
            "not from 2021",
        ),
        (
            lambda item: item["properties"].pop("datetime"),
            "invalid acquisition time",
        ),
        (
            lambda item: item["assets"]["blue"].update(gsd=20),
            "incompatible B02 asset metadata",
        ),
        (
            lambda item: item["assets"]["blue"]["raster:bands"][0].update(scale=1),
            "incompatible B02 asset metadata",
        ),
        (
            lambda item: item["assets"]["blue"].update(
                href="https://example.invalid/blue.tif"
            ),
            "unapproved asset host",
        ),
    ],
)
def test_rejects_incompatible_or_untrusted_scene_metadata(mutate, match):
    item = make_stac_item()
    mutate(item)

    with pytest.raises(AcquisitionError, match=match):
        _validate_item(item, "S2B_43PGN_20211224_0_L2A")


def test_range_probe_records_object_size_and_etag_without_full_download():
    metadata = _parse_range_response(206, "bytes 0-0/75590338", '"multipart-etag-10"')

    assert metadata == {"size_bytes": 75_590_338, "etag": '"multipart-etag-10"'}

    with pytest.raises(AcquisitionError, match="one-byte range"):
        _parse_range_response(200, None, '"etag"')


def test_sha256_records_exact_file_content(tmp_path):
    source = tmp_path / "source.tif"
    source.write_bytes(b"bounded-raster-window")

    assert _sha256(source) == "debd616adf6efa784a58c6976f103fee36b3b60db1aa7a3be245d268fdf920ba"


def test_spatial_window_gap_uses_edge_to_edge_distance():
    first = (500_000, 1_200_000, 505_120, 1_205_120)
    second = (520_120, 1_200_000, 525_240, 1_205_120)
    overlapping = (504_000, 1_204_000, 509_120, 1_209_120)

    assert _window_gap_meters(first, second) == pytest.approx(15_000)
    assert _window_gap_meters(first, overlapping) == 0


def test_validates_worldcover_nodata_and_documented_class_codes():
    _validate_worldcover_codes(np.array([[0, 10, 20, 100]], dtype=np.uint8))

    with pytest.raises(AcquisitionError, match="unsupported class codes"):
        _validate_worldcover_codes(np.array([[0, 11]], dtype=np.uint8))


def test_manifest_requires_checksums_spatial_gap_and_all_data_splits():
    manifest = {
        "phase": 21,
        "geographic_split": {
            "region_count": 3,
            "edge_to_edge_minimum_gap_m": 12_000,
            "regions": [
                {"image_sha256": "a" * 64, "label_sha256": "b" * 64}
                for _ in range(3)
            ],
        },
        "label_source": {"license": "CC-BY-4.0"},
        "preparation": {
            "prepared_metadata": {
                "sample_count": 3,
                "samples_by_split": {"train": 1, "validation": 1, "test": 1},
            }
        },
        "full_source_objects_downloaded": False,
        "training_performed": False,
        "model_checkpoint_created": False,
    }

    _validate_manifest(manifest)

    manifest["training_performed"] = True
    with pytest.raises(AcquisitionError, match="must not claim downloads or training"):
        _validate_manifest(manifest)
