from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import socket
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote, urljoin, urlsplit
from uuid import UUID

import httpx
import numpy as np
import rasterio
from pyproj import Transformer
from pyproj.exceptions import ProjError
from rasterio.errors import RasterioError, WindowError
from rasterio.features import geometry_mask, geometry_window
from rasterio.windows import Window
from shapely import transform as transform_geometry
from shapely.errors import GEOSException
from shapely.geometry import mapping, shape

from config import settings
from satellite.models import GeoJSONPolygon, SentinelSceneAcquisitionRequest

STAC_ITEM_URL = "https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items/{scene_id}"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
ODATA_ROOT = "https://download.dataspace.copernicus.eu/odata/v1/Products"
PROVIDER = "Copernicus Data Space Ecosystem"
COLLECTION = "sentinel-2-l2a"
OFFICIAL_HOSTS = {
    "stac.dataspace.copernicus.eu",
    "identity.dataspace.copernicus.eu",
    "download.dataspace.copernicus.eu",
    "zipper.dataspace.copernicus.eu",
    "eodata.dataspace.copernicus.eu",
}
TIMEOUT = httpx.Timeout(45.0, connect=8.0, read=30.0, write=10.0, pool=8.0)
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 4
MAX_RETRIES = 2
CHUNK_BYTES = 1024 * 1024
BANDS = ("B02", "B03", "B04", "B08")
ITEM_ID_PATTERN = re.compile(
    r"^S2[ABC]_MSIL2A_(\d{8}T\d{6})_N\d{4}_R\d{3}_T[0-9A-Z]{5}_\d{8}T\d{6}$"
)
JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \r\n\x87\n"


class AcquisitionError(Exception):
    status_code = 502
    detail = "The selected Copernicus scene could not be acquired."


class AcquisitionConfigurationError(AcquisitionError):
    status_code = 503
    detail = "Configure CDSE_USERNAME and CDSE_PASSWORD on the backend to acquire imagery."


class AcquisitionUnavailableError(AcquisitionError):
    status_code = 503
    detail = "The Copernicus imagery service is temporarily unavailable."


class AcquisitionTimeoutError(AcquisitionError):
    status_code = 504
    detail = "The Copernicus imagery request timed out. Retry the selected scene."


class AcquisitionRateLimitError(AcquisitionError):
    status_code = 429
    detail = "Copernicus is rate limiting imagery requests. Retry shortly."


class AcquisitionAuthorizationError(AcquisitionError):
    status_code = 502
    detail = "Copernicus rejected the configured account credentials or imagery access."


class AcquisitionResponseError(AcquisitionError):
    status_code = 502
    detail = "Copernicus returned invalid or unsupported scene metadata."


class AcquisitionSizeError(AcquisitionError):
    status_code = 413
    detail = "The selected scene exceeds the configured imagery size limit."


class AcquisitionRasterError(AcquisitionError):
    status_code = 422
    detail = "The downloaded Sentinel-2 bands could not be aligned as a safe GeoTIFF."


def _assert_official_public_host(host: str) -> None:
    if host not in OFFICIAL_HOSTS:
        raise AcquisitionResponseError
    try:
        records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise AcquisitionUnavailableError from exc
    addresses = {record[4][0] for record in records}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise AcquisitionResponseError


def _validate_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise AcquisitionResponseError from exc
    host = parsed.hostname
    if (
        parsed.scheme != "https"
        or host not in OFFICIAL_HOSTS
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise AcquisitionResponseError
    _assert_official_public_host(host)
    return value


def _response_error(status_code: int) -> AcquisitionError:
    if status_code == 429:
        return AcquisitionRateLimitError()
    if status_code in {401, 403}:
        return AcquisitionAuthorizationError()
    if status_code >= 500:
        return AcquisitionUnavailableError()
    return AcquisitionResponseError()


def _check_content_length(value: str | None, maximum: int) -> None:
    if value is None:
        return
    try:
        length = int(value)
    except ValueError as exc:
        raise AcquisitionResponseError from exc
    if length < 0:
        raise AcquisitionResponseError()
    if length > maximum:
        raise AcquisitionSizeError()


def _read_json(url: str, *, headers: dict[str, str] | None = None) -> dict:
    _validate_url(url)
    try:
        with httpx.stream(
            "GET",
            url,
            headers=headers,
            timeout=TIMEOUT,
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise _response_error(response.status_code)
            _check_content_length(
                response.headers.get("content-length"),
                MAX_METADATA_BYTES,
            )
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_METADATA_BYTES:
                    raise AcquisitionResponseError()
    except httpx.TimeoutException as exc:
        raise AcquisitionTimeoutError from exc
    except httpx.RequestError as exc:
        raise AcquisitionUnavailableError from exc
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionResponseError from exc
    if not isinstance(payload, dict):
        raise AcquisitionResponseError()
    return payload


def _access_token() -> str:
    if not settings.cdse_username or not settings.cdse_password:
        raise AcquisitionConfigurationError()
    _validate_url(TOKEN_URL)
    try:
        response = httpx.post(
            TOKEN_URL,
            data={
                "client_id": "cdse-public",
                "grant_type": "password",
                "username": settings.cdse_username,
                "password": settings.cdse_password,
            },
            timeout=TIMEOUT,
            follow_redirects=False,
        )
    except httpx.TimeoutException as exc:
        raise AcquisitionTimeoutError from exc
    except httpx.RequestError as exc:
        raise AcquisitionUnavailableError from exc
    if response.status_code != 200:
        raise _response_error(response.status_code)
    try:
        token = response.json().get("access_token")
    except (ValueError, AttributeError) as exc:
        raise AcquisitionResponseError from exc
    if not isinstance(token, str) or not token or len(token) > 16_384:
        raise AcquisitionResponseError()
    return token


def _item_id(request: SentinelSceneAcquisitionRequest) -> str:
    if request.provider != PROVIDER or request.collection != COLLECTION:
        raise AcquisitionResponseError()
    if not ITEM_ID_PATTERN.fullmatch(request.scene_id):
        raise AcquisitionResponseError()
    return request.scene_id


def _datetime(item: dict, scene_id: str) -> str:
    if not ITEM_ID_PATTERN.fullmatch(scene_id):
        raise AcquisitionResponseError()
    properties = item.get("properties")
    value = properties.get("datetime") if isinstance(properties, dict) else None
    if not isinstance(value, str):
        raise AcquisitionResponseError()
    from datetime import datetime, timezone

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AcquisitionResponseError from exc
    if parsed.tzinfo is None:
        raise AcquisitionResponseError()
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _asset_references(item: dict, scene_id: str) -> tuple[str, dict[str, str]]:
    assets = item.get("assets")
    if not isinstance(assets, dict):
        raise AcquisitionResponseError()
    product_uuid: UUID | None = None
    product = assets.get("Product")
    if isinstance(product, dict) and isinstance(product.get("href"), str):
        parsed = urlsplit(product["href"])
        if (
            parsed.scheme == "https"
            and parsed.hostname
            in {"download.dataspace.copernicus.eu", "zipper.dataspace.copernicus.eu"}
            and parsed.port in {None, 443}
            and parsed.username is None
            and parsed.password is None
            and re.fullmatch(r"/odata/v1/Products\([0-9a-fA-F-]{36}\)/\$value", parsed.path)
        ):
            try:
                product_uuid = UUID(parsed.path.split("(", 1)[1].split(")", 1)[0])
            except ValueError:
                product_uuid = None
    if product_uuid is None:
        raise AcquisitionResponseError()

    base_path = None
    asset_paths: dict[str, str] = {}
    for band in BANDS:
        asset = assets.get(f"{band}_10m")
        href = asset.get("href") if isinstance(asset, dict) else None
        if not isinstance(href, str):
            raise AcquisitionResponseError()
        parsed = urlsplit(href)
        if (
            parsed.scheme != "s3"
            or parsed.netloc != "eodata"
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise AcquisitionResponseError()
        components = parsed.path.strip("/").split("/")
        if (
            len(components) < 10
            or components[:3] != ["Sentinel-2", "MSI", "L2A"]
            or not re.fullmatch(r"\d{4}/\d{2}/\d{2}", "/".join(components[3:6]))
            or components[6] != f"{scene_id}.SAFE"
            or components[7] != "GRANULE"
            or components[9] != "IMG_DATA"
            or len(components) != 12
            or components[10] != "R10m"
            or not re.fullmatch(r"[A-Za-z0-9_-]+", components[8])
            or not re.fullmatch(r"[A-Za-z0-9_-]+_" + band + r"_10m\.jp2", components[11])
        ):
            raise AcquisitionResponseError()
        shared_path = components[:9]
        if base_path is not None and shared_path != base_path:
            raise AcquisitionResponseError()
        base_path = shared_path
        asset_paths[band] = "/".join(components)

    if base_path is None:
        raise AcquisitionResponseError()
    return str(product_uuid), asset_paths


def _sha256_checksum(asset: object) -> str | None:
    if not isinstance(asset, dict):
        return None
    checksum = asset.get("file:checksum") or asset.get("checksum:multihash")
    if not isinstance(checksum, str):
        return None
    normalized = checksum.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[7:]
    elif normalized.startswith("1220") and len(normalized) == 68:
        normalized = normalized[4:]
    if re.fullmatch(r"[0-9a-f]{64}", normalized):
        return normalized
    return None


def _file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _node_url(product_uuid: str, parents: tuple[str, ...]) -> str:
    path = f"{ODATA_ROOT}({product_uuid})/Nodes"
    for parent in parents:
        path += f"({quote(parent, safe='')})/Nodes"
    return path


def _nodes(
    product_uuid: str,
    parents: tuple[str, ...],
    token: str,
) -> list[dict]:
    result = _read_json(
        _node_url(product_uuid, parents),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    nodes = result.get("result")
    if not isinstance(nodes, list) or any(not isinstance(node, dict) for node in nodes):
        raise AcquisitionResponseError()
    return nodes


def _node_by_name(nodes: list[dict], name: str) -> dict:
    matches = [node for node in nodes if node.get("Name") == name]
    if len(matches) != 1:
        raise AcquisitionResponseError()
    return matches[0]


def _file_url(product_uuid: str, node_path: str) -> str:
    parents = node_path.split("/")
    encoded = [quote(part, safe="") for part in parents]
    path = f"{ODATA_ROOT}({product_uuid})"
    for parent in encoded[:-1]:
        path += f"/Nodes({parent})"
    return f"{path}/Nodes({encoded[-1]})/$value"


def _download_file(
    product_uuid: str,
    node_path: str,
    destination: Path,
    *,
    token: str,
    max_bytes: int,
) -> int:
    url = _file_url(product_uuid, node_path)
    for attempt in range(MAX_RETRIES + 1):
        current_url = url
        try:
            for _ in range(MAX_REDIRECTS + 1):
                _validate_url(current_url)
                if urlsplit(current_url).hostname not in {
                    "download.dataspace.copernicus.eu",
                    "zipper.dataspace.copernicus.eu",
                    "eodata.dataspace.copernicus.eu",
                }:
                    raise AcquisitionResponseError()
                with httpx.stream(
                    "GET",
                    current_url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=TIMEOUT,
                    follow_redirects=False,
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise AcquisitionResponseError()
                        current_url = urljoin(current_url, location)
                        continue
                    if response.status_code != 200:
                        raise _response_error(response.status_code)
                    _check_content_length(
                        response.headers.get("content-length"),
                        max_bytes,
                    )
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type not in {"image/jp2", "application/octet-stream"}:
                        raise AcquisitionResponseError()
                    total = 0
                    signature = bytearray()
                    with destination.open("wb") as output:
                        for chunk in response.iter_bytes(CHUNK_BYTES):
                            if len(signature) < len(JP2_SIGNATURE):
                                signature.extend(chunk[: len(JP2_SIGNATURE) - len(signature)])
                            total += len(chunk)
                            if total > max_bytes:
                                raise AcquisitionSizeError()
                            output.write(chunk)
                    if total < 12 or not (
                        bytes(signature).startswith(JP2_SIGNATURE)
                        or bytes(signature).startswith(b"\xff\x4f\xff\x51")
                    ):
                        raise AcquisitionResponseError()
                    return total
            else:
                raise AcquisitionResponseError()
        except httpx.TimeoutException as exc:
            if attempt >= MAX_RETRIES:
                raise AcquisitionTimeoutError from exc
            time.sleep(0.2 * (attempt + 1))
        except httpx.RequestError as exc:
            if attempt >= MAX_RETRIES:
                raise AcquisitionUnavailableError from exc
            time.sleep(0.2 * (attempt + 1))
        except AcquisitionRateLimitError:
            if attempt >= MAX_RETRIES:
                raise
            time.sleep(0.2 * (attempt + 1))
    raise AcquisitionUnavailableError()


def _source_bands(
    item: dict,
    asset_paths: dict[str, str],
    product_uuid: str,
    token: str,
    staging: Path,
    max_bytes: int,
) -> dict[str, Path]:
    root = _node_by_name(_nodes(product_uuid, (), token), f"{item['id']}.SAFE")
    parent = (root["Name"],)
    granule_root = _node_by_name(_nodes(product_uuid, parent, token), "GRANULE")
    parent = (*parent, granule_root["Name"])
    first_asset = next(iter(asset_paths.values())).split("/")
    granule = _node_by_name(_nodes(product_uuid, parent, token), first_asset[8])
    parent = (*parent, granule["Name"])
    imagery = _node_by_name(_nodes(product_uuid, parent, token), "IMG_DATA")
    parent = (*parent, imagery["Name"])
    resolution = _node_by_name(_nodes(product_uuid, parent, token), "R10m")
    parent = (*parent, resolution["Name"])
    available = _nodes(product_uuid, parent, token)

    downloaded: dict[str, Path] = {}
    total_bytes = 0
    for band, asset_path in asset_paths.items():
        name = asset_path.split("/")[-1]
        node = _node_by_name(available, name)
        size = node.get("ContentLength")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise AcquisitionResponseError()
        if size > max_bytes - total_bytes:
            raise AcquisitionSizeError()
        destination = staging / f"{band}.jp2"
        downloaded_bytes = _download_file(
            product_uuid,
            "/".join((*parent, node["Name"])),
            destination,
            token=token,
            max_bytes=max_bytes - total_bytes,
        )
        if downloaded_bytes != size:
            raise AcquisitionResponseError()
        total_bytes += downloaded_bytes
        downloaded[band] = destination
    return downloaded


def _same_grid(first: rasterio.DatasetReader, other: rasterio.DatasetReader) -> bool:
    return (
        first.width == other.width
        and first.height == other.height
        and first.crs == other.crs
        and first.dtypes == other.dtypes
        and first.dtypes[0] == "uint16"
        and all(
            math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
            for a, b in zip(first.transform, other.transform, strict=True)
        )
    )


def _compose_geotiff(
    downloaded: dict[str, Path],
    output_path: Path,
    *,
    item: dict,
    aoi: GeoJSONPolygon,
    acquisition_datetime: str,
    scene_id: str,
    product_uuid: str,
    max_pixels: int,
) -> None:
    try:
        with rasterio.open(downloaded["B02"]) as reference:
            width, height = reference.width, reference.height
            if (
                reference.crs is None
                or width < 1
                or height < 1
                or reference.driver != "JP2OpenJPEG"
                or reference.count != 1
                or reference.dtypes[0] != "uint16"
                or not reference.crs.is_projected
                or reference.crs.linear_units != "metre"
                or not all(
                    math.isclose(abs(resolution), 10, rel_tol=0, abs_tol=1e-6)
                    for resolution in reference.res
                )
            ):
                raise AcquisitionRasterError()
            sources = {band: rasterio.open(downloaded[band]) for band in BANDS}
            try:
                if any(
                    not _same_grid(reference, source)
                    or source.count != 1
                    or source.driver != "JP2OpenJPEG"
                    for source in sources.values()
                ):
                    raise AcquisitionRasterError()
                try:
                    transformer = Transformer.from_crs(
                        "EPSG:4326",
                        reference.crs,
                        always_xy=True,
                    )

                    def project_coordinates(coordinates: np.ndarray) -> np.ndarray:
                        x, y = transformer.transform(
                            coordinates[:, 0],
                            coordinates[:, 1],
                        )
                        return np.stack((x, y), axis=-1)

                    projected_aoi = transform_geometry(
                        shape(aoi.model_dump()),
                        project_coordinates,
                    )
                    crop_window = geometry_window(
                        reference,
                        [mapping(projected_aoi)],
                    ).round_offsets().round_lengths()
                except (
                    RasterioError,
                    WindowError,
                    ProjError,
                    GEOSException,
                    ValueError,
                    TypeError,
                ) as exc:
                    raise AcquisitionRasterError from exc
                crop_width = int(crop_window.width)
                crop_height = int(crop_window.height)
                if (
                    projected_aoi.is_empty
                    or not projected_aoi.is_valid
                    or crop_width < 1
                    or crop_height < 1
                    or crop_width * crop_height > max_pixels
                ):
                    raise AcquisitionRasterError()
                profile = reference.profile.copy()
                profile.update(
                    width=crop_width,
                    height=crop_height,
                    transform=reference.window_transform(crop_window),
                    driver="GTiff",
                    count=len(BANDS),
                    dtype="uint16",
                    nodata=0,
                    compress="deflate",
                    predictor=2,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                    BIGTIFF="IF_SAFER",
                )
                properties = item.get("properties")
                properties = properties if isinstance(properties, dict) else {}
                platform = properties.get("platform")
                processing_level = properties.get("product:type")
                with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                    with rasterio.open(output_path, "w", **profile) as destination:
                        for band_index, band in enumerate(BANDS, start=1):
                            destination.set_band_description(band_index, band)
                        destination.update_tags(
                            PLATFORM=platform if isinstance(platform, str) else "Sentinel-2",
                            ACQUISITION_DATE=acquisition_datetime[:10],
                            STAC_PROVIDER=PROVIDER,
                            STAC_COLLECTION=COLLECTION,
                            STAC_ITEM_ID=scene_id,
                            STAC_ITEM_URL=STAC_ITEM_URL.format(scene_id=quote(scene_id, safe="")),
                            CDSE_PRODUCT_UUID=product_uuid,
                            PROCESSING_LEVEL=(
                                processing_level if isinstance(processing_level, str) else ""
                            ),
                            ASSET_BANDS=",".join(BANDS),
                        )
                        any_valid = False
                        crop_row_offset = int(crop_window.row_off)
                        crop_column_offset = int(crop_window.col_off)
                        for row in range(0, crop_height, 256):
                            for column in range(0, crop_width, 256):
                                block_height = min(256, crop_height - row)
                                block_width = min(256, crop_width - column)
                                source_window = Window.from_slices(
                                    (
                                        crop_row_offset + row,
                                        crop_row_offset + row + block_height,
                                    ),
                                    (
                                        crop_column_offset + column,
                                        crop_column_offset + column + block_width,
                                    ),
                                )
                                destination_window = Window.from_slices(
                                    (row, row + block_height),
                                    (column, column + block_width),
                                )
                                arrays = [
                                    sources[band].read(
                                        1,
                                        window=source_window,
                                        masked=True,
                                    )
                                    for band in BANDS
                                ]
                                within_aoi = geometry_mask(
                                    [mapping(projected_aoi)],
                                    out_shape=(block_height, block_width),
                                    transform=reference.window_transform(source_window),
                                    all_touched=False,
                                    invert=True,
                                )
                                valid = np.logical_and.reduce(
                                    [
                                        ~np.ma.getmaskarray(array)
                                        & within_aoi
                                        & (np.asarray(array.data) != 0)
                                        & np.isfinite(np.asarray(array.data))
                                        for array in arrays
                                    ]
                                )
                                if np.any(valid):
                                    any_valid = True
                                    for index, array in enumerate(arrays, start=1):
                                        destination.write(
                                            np.where(valid, array.data, 0).astype(np.uint16),
                                            index,
                                            window=destination_window,
                                        )
                                destination.write_mask(
                                    valid.astype(np.uint8) * 255,
                                    window=destination_window,
                                )
                        if not any_valid:
                            raise AcquisitionRasterError()
            finally:
                for source in sources.values():
                    source.close()
    except AcquisitionError:
        raise
    except (OSError, RasterioError, ValueError, TypeError) as exc:
        raise AcquisitionRasterError from exc


def acquire_scene(
    request: SentinelSceneAcquisitionRequest,
    destination: Path,
    *,
    max_bytes: int,
    max_pixels: int,
    aoi_max_area_km2: float,
) -> dict:
    scene_id = _item_id(request)
    if request.aoi.estimate_area_km2() > aoi_max_area_km2:
        raise AcquisitionResponseError(
            f"Acquisition AOI exceeds the configured {aoi_max_area_km2:g} km² limit."
        )
    token = _access_token()
    item_url = STAC_ITEM_URL.format(scene_id=quote(scene_id, safe=""))
    item = _read_json(
        item_url,
        headers={"Accept": "application/geo+json"},
    )
    if (
        item.get("type") != "Feature"
        or item.get("id") != scene_id
        or item.get("collection") != COLLECTION
    ):
        raise AcquisitionResponseError()
    acquisition_datetime = _datetime(item, scene_id)
    product_uuid, asset_paths = _asset_references(item, scene_id)
    item_assets = item.get("assets")
    expected_checksums = {
        band: checksum
        for band in BANDS
        if isinstance(item_assets, dict)
        and (checksum := _sha256_checksum(item_assets.get(f"{band}_10m")))
    }
    with TemporaryDirectory(prefix="cdse-acquisition-") as staging:
        staging_path = Path(staging)
        downloaded = _source_bands(
            item,
            asset_paths,
            product_uuid,
            token,
            staging_path,
            max_bytes,
        )
        downloaded_bytes = sum(path.stat().st_size for path in downloaded.values())
        for band, expected_checksum in expected_checksums.items():
            checksum = _file_sha256(downloaded[band])
            if checksum != expected_checksum:
                raise AcquisitionResponseError(
                    "A Sentinel band checksum does not match the STAC asset metadata."
                )
        _compose_geotiff(
            downloaded,
            destination,
            item=item,
            aoi=request.aoi,
            acquisition_datetime=acquisition_datetime,
            scene_id=scene_id,
            product_uuid=product_uuid,
            max_pixels=max_pixels,
        )
    return {
        "provider": PROVIDER,
        "collection": COLLECTION,
        "scene_id": scene_id,
        "product_uuid": product_uuid,
        "catalog_url": item_url,
        "acquisition_datetime": acquisition_datetime,
        "bands": list(BANDS),
        "downloaded_bytes": downloaded_bytes,
        "verified_source_checksums": sorted(expected_checksums),
    }
