from __future__ import annotations

from datetime import date, datetime
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import RasterioError
from rasterio.io import DatasetReader
from rasterio.transform import Affine, array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds

from processing.preprocessing import BAND_NAMES, identify_band


class ImageryValidationError(ValueError):
    """A safe, user-facing explanation for an unsupported or invalid scene."""


_ACQUISITION_DATE_TAGS = {
    "acquisition_date",
    "acquisitiondate",
    "date_acquired",
    "dateacquired",
    "sensing_time",
    "sensingtime",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y%m%d",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y%m%dT%H%M%S",
)
_SENTINEL_PRODUCT_DATE = re.compile(r"S2[ABC]_[A-Z0-9_]*?_(\d{8})T\d{6}", re.IGNORECASE)
_LANDSAT_PRODUCT_DATE = re.compile(r"LC0[89]_L[12][A-Z0-9_]*?_(\d{8})", re.IGNORECASE)
_SAFE_DESCRIPTION = re.compile(r"[\x00-\x1f\x7f]")
_MAX_BANDS = 64
_PREVIEW_MAX_SIDE = 512


def _safe_band_description(value: str, index: int) -> str:
    description = _SAFE_DESCRIPTION.sub("", value).strip()
    return description[:80] if description else f"Band {index}"


def _identity_tag(tags: dict[str, str], candidates: set[str]) -> str | None:
    values = {
        _SAFE_DESCRIPTION.sub("", value).strip()[:80]
        for key, value in tags.items()
        if key.casefold() in candidates and value.strip()
    }
    return values.pop() if len(values) == 1 else None


def _acquisition_date(tags: dict[str, str], filename: str) -> str | None:
    for key, value in tags.items():
        if key.lower() not in _ACQUISITION_DATE_TAGS:
            continue
        candidate = value.strip()
        for date_format in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(candidate, date_format)
            except ValueError:
                continue
            return parsed.date().isoformat()
        try:
            return date.fromisoformat(candidate[:10]).isoformat()
        except ValueError:
            continue

    for pattern in (_SENTINEL_PRODUCT_DATE, _LANDSAT_PRODUCT_DATE):
        match = pattern.search(filename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
            except ValueError:
                continue
    return None


def _band_metadata(dataset: DatasetReader, filename: str) -> list[dict[str, Any]]:
    bands = []
    for index in range(1, dataset.count + 1):
        description = dataset.descriptions[index - 1] or ""
        tags = dataset.tags(index)
        label = " ".join((description, tags.get("NAME", ""), tags.get("BAND", "")))
        code = identify_band(label)
        if dataset.count == 1 and code is None:
            code = identify_band(filename)
        bands.append(
            {
                "index": index,
                "code": code,
                "name": (
                    BAND_NAMES[code]
                    if code in BAND_NAMES
                    else _safe_band_description(description, index)
                ),
            }
        )
    return bands


def _preview_band_indexes(bands: list[dict[str, Any]]) -> tuple[list[int], str]:
    recognized = {
        band["code"]: band["index"]
        for band in bands
        if band["code"] in {"B02", "B03", "B04"}
    }
    if all(code in recognized for code in ("B04", "B03", "B02")):
        return [recognized["B04"], recognized["B03"], recognized["B02"]], (
            "RGB preview using bands labelled B04, B03, and B02; labels and satellite "
            "provenance are not independently verified; linear 2nd-to-98th percentile stretch."
        )
    if len(bands) >= 3:
        return [1, 2, 3], (
            "RGB preview using raster bands 1, 2, and 3 in file order; "
            "band semantics are not verified."
        )
    return [1], "Grayscale preview using raster band 1."


def inspect_geotiff(
    source_path: Path,
    preview_path: Path,
    *,
    original_filename: str,
    sha256: str,
    max_pixels: int,
) -> dict[str, Any]:
    try:
        with rasterio.open(source_path) as source:
            if source.driver != "GTiff":
                raise ImageryValidationError("Only GeoTIFF imagery is supported.")
            if source.width < 1 or source.height < 1 or source.count < 1:
                raise ImageryValidationError("The GeoTIFF contains no raster pixels or bands.")
            if source.width * source.height > max_pixels:
                raise ImageryValidationError(
                    f"GeoTIFF exceeds the configured {max_pixels:,}-pixel imagery limit."
                )
            if source.count > _MAX_BANDS:
                raise ImageryValidationError(
                    f"GeoTIFFs with more than {_MAX_BANDS} bands are not supported."
                )
            if source.crs is None:
                raise ImageryValidationError(
                    "The GeoTIFF has no coordinate reference system; georeferenced imagery is required."
                )
            transform = source.transform
            resolution = [
                math.hypot(transform.a, transform.d),
                math.hypot(transform.b, transform.e),
            ]
            if not all(math.isfinite(value) and value > 0 for value in resolution):
                raise ImageryValidationError("The GeoTIFF geotransform is invalid.")

            try:
                west, south, east, north = transform_bounds(
                    source.crs,
                    "EPSG:4326",
                    *source.bounds,
                    densify_pts=21,
                )
            except (RasterioError, ValueError) as exc:
                raise ImageryValidationError(
                    "The GeoTIFF extent could not be transformed to geographic coordinates."
                ) from exc
            if not all(math.isfinite(value) for value in (west, south, east, north)):
                raise ImageryValidationError("The GeoTIFF has an invalid spatial extent.")

            tags = source.tags()
            catalog_item_id = tags.get("STAC_ITEM_ID", "")
            catalog_item_url = tags.get("STAC_ITEM_URL", "")
            catalog_source = None
            if (
                tags.get("STAC_PROVIDER") == "Copernicus Data Space Ecosystem"
                and tags.get("STAC_COLLECTION") == "sentinel-2-l2a"
                and re.fullmatch(
                    r"S2[ABC]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T[0-9A-Z]{5}_\d{8}T\d{6}",
                    catalog_item_id,
                )
                and catalog_item_url
                == (
                    "https://stac.dataspace.copernicus.eu/v1/collections/"
                    f"sentinel-2-l2a/items/{catalog_item_id}"
                )
                and re.fullmatch(
                    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                    tags.get("CDSE_PRODUCT_UUID", ""),
                )
            ):
                catalog_source = {
                    "provider": tags["STAC_PROVIDER"],
                    "collection": tags["STAC_COLLECTION"],
                    "item_id": catalog_item_id,
                    "catalog_url": catalog_item_url,
                    "product_uuid": tags["CDSE_PRODUCT_UUID"],
                    "processing_level": tags.get("PROCESSING_LEVEL") or None,
                    "asset_bands": [
                        band
                        for band in tags.get("ASSET_BANDS", "").split(",")
                        if band in {"B02", "B03", "B04", "B08"}
                    ],
                    "provenance_verified": False,
                }
            bands = _band_metadata(source, original_filename)
            indexes, preview_description = _preview_band_indexes(bands)
            preview_transform, preview_grid_width, preview_grid_height = calculate_default_transform(
                source.crs,
                "EPSG:4326",
                source.width,
                source.height,
                *source.bounds,
            )
            scale = min(
                _PREVIEW_MAX_SIDE / preview_grid_width,
                _PREVIEW_MAX_SIDE / preview_grid_height,
                1,
            )
            output_width = max(1, round(preview_grid_width * scale))
            output_height = max(1, round(preview_grid_height * scale))
            preview_transform = preview_transform @ Affine.scale(
                preview_grid_width / output_width,
                preview_grid_height / output_height,
            )
            preview_bounds = [
                float(value)
                for value in array_bounds(output_height, output_width, preview_transform)
            ]
            with WarpedVRT(
                source,
                crs="EPSG:4326",
                transform=preview_transform,
                width=output_width,
                height=output_height,
                resampling=Resampling.average,
                nodata=source.nodata,
            ) as warped:
                sample = warped.read(indexes, masked=True)
            channels: list[np.ndarray] = []
            valid_masks: list[np.ndarray] = []
            for channel in sample:
                values = np.asarray(channel.data, dtype=np.float32)
                valid = ~np.ma.getmaskarray(channel) & np.isfinite(values)
                if not np.any(valid):
                    raise ImageryValidationError(
                        "The selected preview bands contain no valid pixels."
                    )
                low, high = np.percentile(values[valid], (2, 98))
                if not math.isfinite(float(low)) or not math.isfinite(float(high)):
                    raise ImageryValidationError("The GeoTIFF preview values are invalid.")
                if high <= low:
                    stretched = np.where(valid, 0.5, 0.0)
                else:
                    stretched = np.clip((values - low) / (high - low), 0.0, 1.0)
                channels.append(stretched)
                valid_masks.append(valid)

            valid_preview = np.logical_and.reduce(valid_masks)
            if not np.any(valid_preview):
                raise ImageryValidationError(
                    "The selected preview bands do not share any valid pixels."
                )
            if len(channels) == 1:
                rgb = np.repeat(channels[0][..., None], 3, axis=2)
            else:
                rgb = np.stack(channels, axis=2)
            rgba = np.zeros((output_height, output_width, 4), dtype=np.uint8)
            rgba[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            rgba[..., 3] = np.where(valid_preview, 255, 0).astype(np.uint8)
            Image.fromarray(rgba, mode="RGBA").save(preview_path, format="PNG", optimize=True)

            return {
                "original_filename": original_filename,
                "sha256": sha256,
                "format": "GeoTIFF",
                "acquisition_date": _acquisition_date(tags, original_filename),
                "platform": _identity_tag(
                    tags,
                    {"platform", "satellite", "spacecraft_name", "spacecraft"},
                ),
                **({"catalog_source": catalog_source} if catalog_source else {}),
                "sensor": _identity_tag(
                    tags,
                    {"sensor", "instrument", "sensor_name", "instrument_name"},
                ),
                "crs": source.crs.to_string(),
                "extent_wgs84": [
                    float(west),
                    float(south),
                    float(east),
                    float(north),
                ],
                "width": int(source.width),
                "height": int(source.height),
                "band_count": int(source.count),
                "resolution": resolution,
                "resolution_units": source.crs.linear_units,
                "bands": bands,
                "preview": {
                    "filename": "preview.png",
                    "width": output_width,
                    "height": output_height,
                    "crs": "EPSG:4326",
                    "bounds_wgs84": preview_bounds,
                    "description": preview_description,
                    "cloud_mask_applied": False,
                },
                "metadata_source": "GeoTIFF tags, filename pattern, and raster georeferencing",
                "provenance_verified": False,
                "provenance_note": (
                    "Metadata, band labels, and satellite origin are not independently authenticated."
                ),
            }
    except ImageryValidationError:
        raise
    except MemoryError as exc:
        raise ImageryValidationError(
            "The GeoTIFF could not be previewed within the available memory."
        ) from exc
    except (RasterioError, OSError, ValueError, OverflowError) as exc:
        raise ImageryValidationError(
            "The uploaded file is corrupt or is not a readable georeferenced GeoTIFF."
        ) from exc
