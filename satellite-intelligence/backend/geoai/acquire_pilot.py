from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import httpx
import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.io import DatasetReader
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject
from rasterio.windows import Window

from geoai.training_data import (
    MAX_CHIP_SIZE,
    RasterTrainingPair,
    SENTINEL2_BANDS,
    WORLD_COVER_CLASSES,
    WORLD_COVER_NODATA,
    prepare_training_dataset,
)

STAC_ITEM_URL = (
    "https://earth-search.aws.element84.com/v1/collections/"
    "sentinel-2-l2a/items/{item_id}"
)
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N09E075_Map.tif"
)
STAC_ASSET_BANDS = dict(zip(SENTINEL2_BANDS, ("blue", "green", "red", "nir")))
WORLD_COVER_TILE = "ESA_WorldCover_10m_2021_v200_N09E075_Map.tif"
WINDOW_SIZE = 512
SPATIAL_GAP_METERS = 10_000
_RANGE_RE = re.compile(r"bytes 0-0/(\d+)")


class AcquisitionError(ValueError):
    """A safe explanation for invalid or unavailable public acquisition data."""


PILOT_REGIONS = (
    {
        "region_id": "tamilnadu-erode",
        "center_lon_lat": [77.72, 11.34],
        "item_id": "S2B_43PGN_20211224_0_L2A",
    },
    {
        "region_id": "tamilnadu-mettur",
        "center_lon_lat": [77.80, 11.79],
        "item_id": "S2B_43PGP_20211224_0_L2A",
    },
    {
        "region_id": "tamilnadu-sathyamangalam",
        "center_lon_lat": [77.24, 11.51],
        "item_id": "S2B_43PGN_20211224_0_L2A",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_item(item: dict[str, Any], expected_id: str) -> dict[str, Any]:
    if item.get("id") != expected_id:
        raise AcquisitionError(f"STAC returned an unexpected Sentinel-2 item for {expected_id}.")
    properties = item.get("properties")
    assets = item.get("assets")
    if not isinstance(properties, dict) or not isinstance(assets, dict):
        raise AcquisitionError(f"STAC item {expected_id} is missing properties or assets.")
    acquired_at = properties.get("datetime")
    if not isinstance(acquired_at, str):
        raise AcquisitionError(f"STAC item {expected_id} has an invalid acquisition time.")
    try:
        acquired = datetime.fromisoformat(acquired_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcquisitionError(f"STAC item {expected_id} has an invalid acquisition time.") from exc
    if acquired.year != 2021:
        raise AcquisitionError(f"STAC item {expected_id} is not from 2021.")
    cloud_cover = properties.get("eo:cloud_cover")
    if not isinstance(cloud_cover, (int, float)) or not 0 <= cloud_cover <= 100:
        raise AcquisitionError(f"STAC item {expected_id} has invalid cloud-cover metadata.")
    if not isinstance(properties.get("platform"), str) or not properties["platform"].startswith(
        "sentinel-2"
    ):
        raise AcquisitionError(f"STAC item {expected_id} is not identified as Sentinel-2.")
    band_metadata: dict[str, dict[str, Any]] = {}
    for band in SENTINEL2_BANDS:
        asset_name = STAC_ASSET_BANDS[band]
        asset = assets.get(asset_name)
        raster_bands = asset.get("raster:bands") if isinstance(asset, dict) else None
        eo_bands = asset.get("eo:bands") if isinstance(asset, dict) else None
        if (
            not isinstance(asset, dict)
            or not isinstance(asset.get("href"), str)
            or asset.get("gsd") != 10
            or not isinstance(raster_bands, list)
            or len(raster_bands) != 1
            or not isinstance(raster_bands[0], dict)
            or raster_bands[0].get("data_type") != "uint16"
            or raster_bands[0].get("nodata") != 0
            or raster_bands[0].get("scale") != 0.0001
            or raster_bands[0].get("offset") != 0
            or raster_bands[0].get("spatial_resolution") != 10
            or not isinstance(eo_bands, list)
            or len(eo_bands) != 1
            or not isinstance(eo_bands[0], dict)
            or eo_bands[0].get("common_name") != asset_name
        ):
            raise AcquisitionError(
                f"STAC item {expected_id} has incompatible {band} asset metadata."
            )
        if not asset["href"].startswith(
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/"
        ):
            raise AcquisitionError(f"STAC item {expected_id} has an unapproved asset host.")
        band_metadata[band] = raster_bands[0]
    return {
        "item_id": expected_id,
        "acquisition_datetime": acquired.isoformat(),
        "acquisition_date": acquired.date().isoformat(),
        "platform": properties["platform"],
        "cloud_cover_percent": float(cloud_cover),
        "product_uri": properties.get("s2:product_uri"),
        "assets": {
            band: assets[STAC_ASSET_BANDS[band]]["href"] for band in SENTINEL2_BANDS
        },
        "band_metadata": band_metadata,
    }


def _parse_range_response(status_code: int, content_range: str | None, etag: str | None) -> dict:
    match = _RANGE_RE.fullmatch(content_range or "")
    if status_code != 206 or match is None:
        raise AcquisitionError("Public COG did not honor the bounded one-byte range request.")
    return {"size_bytes": int(match.group(1)), "etag": etag}


def _window_gap_meters(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_min_x, first_min_y, first_max_x, first_max_y = first
    second_min_x, second_min_y, second_max_x, second_max_y = second
    dx = max(first_min_x - second_max_x, second_min_x - first_max_x, 0)
    dy = max(first_min_y - second_max_y, second_min_y - first_max_y, 0)
    return math.hypot(dx, dy)


def _validate_worldcover_codes(labels: np.ndarray) -> None:
    invalid = ~np.isin(
        labels,
        (WORLD_COVER_NODATA, *WORLD_COVER_CLASSES.keys()),
    )
    if np.any(invalid):
        codes = sorted(int(code) for code in np.unique(labels[invalid]))
        raise AcquisitionError(f"WorldCover raster contains unsupported class codes: {codes}.")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    regions = manifest.get("geographic_split", {}).get("regions", [])
    if (
        manifest.get("phase") != 21
        or len(regions) != 3
        or manifest["geographic_split"].get("region_count") != 3
        or manifest["geographic_split"].get("edge_to_edge_minimum_gap_m", 0)
        < SPATIAL_GAP_METERS
    ):
        raise AcquisitionError("The acquisition manifest does not establish three separated regions.")
    if (
        manifest.get("full_source_objects_downloaded") is not False
        or manifest.get("training_performed") is not False
        or manifest.get("model_checkpoint_created") is not False
    ):
        raise AcquisitionError("The acquisition manifest must not claim downloads or training.")
    if manifest.get("label_source", {}).get("license") != "CC-BY-4.0":
        raise AcquisitionError("The acquisition manifest must record the WorldCover license.")
    for region in regions:
        for key in ("image_sha256", "label_sha256"):
            digest = region.get(key)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise AcquisitionError("Every acquired raster needs a SHA-256 checksum.")
    prepared = manifest.get("preparation", {}).get("prepared_metadata", {})
    split_counts = prepared.get("samples_by_split", {})
    if (
        prepared.get("sample_count", 0) < 1
        or any(split_counts.get(split, 0) < 1 for split in ("train", "validation", "test"))
    ):
        raise AcquisitionError("The acquisition manifest needs chips in all three splits.")


def _check_output_directory(output_directory: Path) -> Path:
    supplied = output_directory.expanduser()
    if supplied.is_symlink():
        raise AcquisitionError("The output directory may not be a symbolic link.")
    target = supplied.resolve()
    try:
        target.relative_to(_project_root())
    except ValueError:
        pass
    else:
        raise AcquisitionError("Acquisition data must be stored outside the project repository.")
    if target.exists():
        raise AcquisitionError("The output directory must not already exist.")
    return target


def _probe_asset(client: httpx.Client, url: str) -> dict:
    try:
        with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as response:
            metadata = _parse_range_response(
                response.status_code,
                response.headers.get("Content-Range"),
                response.headers.get("ETag"),
            )
    except httpx.HTTPError as exc:
        raise AcquisitionError("A public raster asset could not be range-probed.") from exc
    return metadata


def acquire_pilot_dataset(output_directory: Path, *, chip_size: int = 128) -> dict[str, Any]:
    """Acquire three small public COG windows and prepare aligned weak-label chips."""
    target = _check_output_directory(Path(output_directory))
    if not 1 <= chip_size <= MAX_CHIP_SIZE:
        raise AcquisitionError(f"Chip size must be between 1 and {MAX_CHIP_SIZE} pixels.")
    if len(PILOT_REGIONS) != 3:
        raise AcquisitionError("The pilot requires exactly three declared geographic regions.")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phase21-pilot-", dir=target.parent))
    try:
        with httpx.Client(timeout=45, follow_redirects=True) as client:
            item_metadata: dict[str, dict[str, Any]] = {}
            for region in PILOT_REGIONS:
                item_id = region["item_id"]
                if item_id not in item_metadata:
                    try:
                        response = client.get(STAC_ITEM_URL.format(item_id=item_id))
                        response.raise_for_status()
                        item = response.json()
                    except (httpx.HTTPError, ValueError) as exc:
                        raise AcquisitionError(
                            f"Could not retrieve public STAC metadata for {item_id}."
                        ) from exc
                    metadata = _validate_item(item, item_id)
                    for band, url in metadata["assets"].items():
                        metadata.setdefault("asset_objects", {})[band] = {
                            **_probe_asset(client, url),
                            "url": url,
                        }
                    item_metadata[item_id] = metadata
            worldcover_object = {
                **_probe_asset(client, WORLDCOVER_URL),
                "url": WORLDCOVER_URL,
            }

        image_dir = staging / "images"
        label_dir = staging / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        pairs: list[RasterTrainingPair] = []
        windows_by_region: dict[str, tuple[float, float, float, float]] = {}
        region_metadata: list[dict[str, Any]] = []
        opened: dict[tuple[str, str], DatasetReader] = {}
        worldcover: DatasetReader | None = None
        try:
            for item_id, metadata in item_metadata.items():
                for band, url in metadata["assets"].items():
                    try:
                        source = rasterio.open(url)
                    except (OSError, RasterioError) as exc:
                        raise AcquisitionError(
                            f"Could not open public {band} COG for {item_id}."
                        ) from exc
                    opened[(item_id, band)] = source
                sources = [opened[(item_id, band)] for band in SENTINEL2_BANDS]
                reference = sources[0]
                if (
                    reference.crs is None
                    or reference.res != (10.0, 10.0)
                    or reference.dtypes != ("uint16",)
                    or reference.nodata != 0
                    or any(
                        source.crs != reference.crs
                        or source.transform != reference.transform
                        or source.width != reference.width
                        or source.height != reference.height
                        or source.dtypes != ("uint16",)
                        or source.nodata != 0
                        for source in sources
                    )
                ):
                    raise AcquisitionError(
                        f"Sentinel-2 bands for {item_id} do not share a 10 m uint16 grid."
                    )
                metadata["raster_grid"] = {
                    "crs": reference.crs.to_string(),
                    "width": reference.width,
                    "height": reference.height,
                    "resolution_m": list(reference.res),
                    "nodata": reference.nodata,
                    "transform": list(reference.transform)[:6],
                }

            try:
                worldcover_source = rasterio.open(WORLDCOVER_URL)
            except (OSError, RasterioError) as exc:
                raise AcquisitionError("Could not open the public WorldCover COG.") from exc
            worldcover = worldcover_source
            if (
                worldcover_source.crs is None
                or worldcover_source.dtypes != ("uint8",)
                or worldcover_source.nodata != WORLD_COVER_NODATA
                or worldcover_source.res != (1 / 12000, 1 / 12000)
            ):
                raise AcquisitionError("The WorldCover source grid or nodata metadata is unexpected.")

            for region in PILOT_REGIONS:
                item_id = region["item_id"]
                metadata = item_metadata[item_id]
                sources = [opened[(item_id, band)] for band in SENTINEL2_BANDS]
                reference = sources[0]
                transformer = Transformer.from_crs(
                    "EPSG:4326", reference.crs, always_xy=True
                )
                x, y = transformer.transform(*region["center_lon_lat"])
                center_row, center_column = reference.index(x, y)
                row = center_row - WINDOW_SIZE // 2
                column = center_column - WINDOW_SIZE // 2
                if (
                    row < 0
                    or column < 0
                    or row + WINDOW_SIZE > reference.height
                    or column + WINDOW_SIZE > reference.width
                ):
                    raise AcquisitionError(
                        f"Pilot region {region['region_id']} is outside its Sentinel-2 tile."
                    )
                window = Window.from_slices(
                    (row, row + WINDOW_SIZE),
                    (column, column + WINDOW_SIZE),
                )
                transform = reference.window_transform(window)
                imagery = np.stack(
                    [source.read(1, window=window) for source in sources],
                    axis=0,
                )
                labels = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
                reproject(
                    source=rasterio.band(worldcover_source, 1),
                    destination=labels,
                    src_transform=worldcover_source.transform,
                    src_crs=worldcover_source.crs,
                    src_nodata=worldcover_source.nodata,
                    dst_transform=transform,
                    dst_crs=reference.crs,
                    dst_nodata=WORLD_COVER_NODATA,
                    resampling=Resampling.nearest,
                )
                _validate_worldcover_codes(labels)

                region_id = region["region_id"]
                image_path = image_dir / f"{region_id}-sentinel2.tif"
                label_path = label_dir / f"{region_id}-worldcover.tif"
                image_profile = {
                    "driver": "GTiff",
                    "width": WINDOW_SIZE,
                    "height": WINDOW_SIZE,
                    "count": len(SENTINEL2_BANDS),
                    "dtype": "uint16",
                    "crs": reference.crs,
                    "transform": transform,
                    "nodata": 0,
                    "compress": "deflate",
                    "tiled": True,
                }
                with rasterio.open(image_path, "w", **image_profile) as output:
                    output.write(imagery)
                    output.scales = [
                        metadata["band_metadata"][band]["scale"]
                        for band in SENTINEL2_BANDS
                    ]
                    output.offsets = [
                        metadata["band_metadata"][band]["offset"]
                        for band in SENTINEL2_BANDS
                    ]
                    for band_index, band in enumerate(SENTINEL2_BANDS, start=1):
                        output.set_band_description(band_index, band)
                    output.update_tags(
                        stac_item_id=item_id,
                        acquisition_date=metadata["acquisition_date"],
                        source="Copernicus Sentinel-2 L2A COG via AWS Open Data",
                        normalization="raw uint16 values divided by 10000 during preparation",
                    )
                with rasterio.open(
                    label_path,
                    "w",
                    driver="GTiff",
                    width=WINDOW_SIZE,
                    height=WINDOW_SIZE,
                    count=1,
                    dtype="uint8",
                    crs=reference.crs,
                    transform=transform,
                    nodata=WORLD_COVER_NODATA,
                    compress="deflate",
                    tiled=True,
                ) as output:
                    output.write(labels, 1)
                    output.update_tags(
                        product="ESA WorldCover 2021 v200",
                        license="CC-BY-4.0",
                        source_tile=WORLD_COVER_TILE,
                    )

                image_invalid = np.any(imagery == 0, axis=0)
                label_invalid = labels == WORLD_COVER_NODATA
                valid = ~(image_invalid | label_invalid)
                class_codes, class_counts = np.unique(labels[~label_invalid], return_counts=True)
                left, bottom, right, top = array_bounds(WINDOW_SIZE, WINDOW_SIZE, transform)
                bounds = (left, bottom, right, top)
                windows_by_region[region_id] = bounds
                region_metadata.append(
                    {
                        "region_id": region_id,
                        "center_lon_lat": region["center_lon_lat"],
                        "item_id": item_id,
                        "acquisition_date": metadata["acquisition_date"],
                        "source_grid_crs": reference.crs.to_string(),
                        "window_size_pixels": WINDOW_SIZE,
                        "window_bounds": list(bounds),
                        "image_file": image_path.relative_to(staging).as_posix(),
                        "image_sha256": _sha256(image_path),
                        "label_file": label_path.relative_to(staging).as_posix(),
                        "label_sha256": _sha256(label_path),
                        "image_nodata_pixels": int(np.count_nonzero(image_invalid)),
                        "label_nodata_pixels": int(np.count_nonzero(label_invalid)),
                        "valid_joint_pixels": int(np.count_nonzero(valid)),
                        "class_pixel_counts": {
                            str(int(code)): int(count)
                            for code, count in zip(class_codes, class_counts)
                        },
                    }
                )
                pairs.append(
                    RasterTrainingPair(
                        image_path=image_path,
                        label_path=label_path,
                        region_id=region_id,
                        acquisition_date=metadata["acquisition_date"],
                    )
                )
        finally:
            if worldcover is not None:
                worldcover.close()
            for source in opened.values():
                source.close()

        if (
            len(
                {
                    item_metadata[region["item_id"]]["raster_grid"]["crs"]
                    for region in PILOT_REGIONS
                }
            )
            != 1
            or next(iter(item_metadata.values()))["raster_grid"]["crs"] != "EPSG:32643"
        ):
            raise AcquisitionError(
                "Pilot windows use different CRSs; spatial separation cannot be validated."
            )
        region_ids = sorted(windows_by_region)
        minimum_gap = min(
            _window_gap_meters(windows_by_region[first], windows_by_region[second])
            for index, first in enumerate(region_ids)
            for second in region_ids[index + 1 :]
        )
        if minimum_gap < SPATIAL_GAP_METERS:
            raise AcquisitionError(
                f"Pilot regions are too close for the {SPATIAL_GAP_METERS} m spatial gap."
            )
        independence_evidence = (
            f"Three fixed 5.12 km windows in EPSG:32643 have a measured minimum "
            f"axis-aligned edge-to-edge gap of {minimum_gap:.0f} m; required gap is "
            f"{SPATIAL_GAP_METERS} m. Region windows, not random chips, are assigned "
            "to separate deterministic splits. Labels remain WorldCover weak supervision."
        )
        prepared_path = staging / "prepared"
        prepared_metadata = prepare_training_dataset(
            pairs,
            prepared_path,
            chip_size=chip_size,
            seed=42,
            geographic_independence_evidence=independence_evidence,
        )
        prepared_metadata["output_directory"] = "prepared"

        rejected_chips: list[dict[str, str]] = []
        chip_counts: dict[str, int] = {region["region_id"]: 0 for region in PILOT_REGIONS}
        for pair in pairs:
            with rasterio.open(pair.image_path) as image, rasterio.open(pair.label_path) as labels:
                for row in range(0, image.height - chip_size + 1, chip_size):
                    for column in range(0, image.width - chip_size + 1, chip_size):
                        window = Window.from_slices(
                            (row, row + chip_size),
                            (column, column + chip_size),
                        )
                        image_chip = image.read(window=window)
                        label_chip = labels.read(1, window=window)
                        valid_chip = np.any(
                            (image_chip != 0).all(axis=0)
                            & (label_chip != WORLD_COVER_NODATA)
                        )
                        if valid_chip:
                            chip_counts[pair.region_id] += 1
                        else:
                            rejected_chips.append(
                                {
                                    "region_id": pair.region_id,
                                    "row": str(row),
                                    "column": str(column),
                                    "reason": "no joint non-nodata Sentinel-2 and WorldCover pixels",
                                }
                            )

        manifest = {
            "phase": 21,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sentinel_source": {
                "provider": "Copernicus Sentinel-2 L2A",
                "access_path": "AWS Open Data COG registry / Element 84 Earth Search STAC",
                "official_terms": "https://dataspace.copernicus.eu/terms-and-conditions",
                "legal_notice": "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice",
                "license_note": (
                    "Copernicus Sentinel data are free, full and open under the Sentinel Data "
                    "Legal Notice; the Earth Search STAC license field is not treated as the "
                    "authoritative data license."
                ),
                "items": list(item_metadata.values()),
            },
            "label_source": {
                "product": "ESA WorldCover 2021 v200",
                "version": "v200",
                "resolution_m": 10,
                "license": "CC-BY-4.0",
                "attribution": (
                    "ESA WorldCover 2021 v200 (CC-BY-4.0), DOI: 10.5281/zenodo.7254221; "
                    "accessed via the AWS Open Data Registry."
                ),
                "source": "https://esa-worldcover.org/en/data-access",
                "registry": "https://registry.opendata.aws/esa-worldcover-vito/",
                "source_tile": WORLD_COVER_TILE,
                "source_object": worldcover_object,
                "class_codes": {
                    str(code): name for code, name in WORLD_COVER_CLASSES.items()
                },
                "nodata_code": WORLD_COVER_NODATA,
                "label_warning": (
                    "WorldCover is weak supervision and is not independent ground truth. "
                    "The annual map and December 2021 Sentinel scenes may differ temporally."
                ),
            },
            "geographic_split": {
                "region_count": len(region_metadata),
                "edge_to_edge_minimum_gap_m": round(minimum_gap, 1),
                "required_gap_m": SPATIAL_GAP_METERS,
                "evidence": independence_evidence,
                "method": "distinct fixed-region windows assigned wholly to deterministic splits",
                "regions": region_metadata,
                "limitation": (
                    "This separation supports a spatially held-out pilot, but does not establish "
                    "independence from the WorldCover training labels or scientific accuracy."
                ),
            },
            "preparation": {
                "bands": list(SENTINEL2_BANDS),
                "normalization": "uint16_reflectance_divide_by_10000",
                "chip_size_pixels": chip_size,
                "chips_by_region": chip_counts,
                "rejected_chips": rejected_chips,
                "prepared_dataset": "prepared/",
                "prepared_metadata": prepared_metadata,
            },
            "full_source_objects_downloaded": False,
            "training_performed": False,
            "model_checkpoint_created": False,
        }
        _validate_manifest(manifest)
        (staging / "acquisition-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise AcquisitionError("The output directory became unavailable during acquisition.")
        os.rename(staging, target)
        return {**manifest, "output_directory": str(target)}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire a bounded Sentinel-2/WorldCover pilot dataset (no training)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory outside the repository; it must not already exist.",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=128,
        help="Prepared chip edge in pixels (1-256; defaults to 128).",
    )
    args = parser.parse_args()
    try:
        result = acquire_pilot_dataset(args.output_dir, chip_size=args.chip_size)
    except (AcquisitionError, OSError, RasterioError, httpx.HTTPError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
