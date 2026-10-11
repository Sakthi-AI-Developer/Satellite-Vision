from __future__ import annotations

import base64
import binascii
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import parse_qs, quote, urlsplit

import httpx
from shapely.geometry import shape
from shapely.errors import GEOSException

from satellite.models import SentinelCatalogSearchRequest

STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
STAC_COLLECTION = "sentinel-2-l2a"
PROVIDER_NAME = "Copernicus Data Space Ecosystem"
REQUEST_TIMEOUT = httpx.Timeout(12.0, connect=5.0)
MAX_PAGE_SIZE = 50
MAX_CURSOR_BYTES = 2048
MAX_FEATURE_GEOMETRY_COORDINATES = 10_000
SPECTRAL_ASSET_PATTERN = re.compile(r"^(B(?:0[1-9]|1[0-2]|8A))_(10|20|60)m$")
ACQUISITION_ASSET_KEYS = ("B02_10m", "B03_10m", "B04_10m", "B08_10m")


class CatalogError(Exception):
    status_code = 503
    detail = "The Copernicus catalog is temporarily unavailable."


class CatalogRateLimitError(CatalogError):
    status_code = 429
    detail = "The Copernicus catalog is rate limited. Retry shortly."


class CatalogTimeoutError(CatalogError):
    status_code = 504
    detail = "The Copernicus catalog request timed out."


class CatalogResponseError(CatalogError):
    status_code = 502
    detail = "The Copernicus catalog returned an invalid response."


def _decode_cursor(cursor: str | None) -> str | None:
    if cursor is None:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        token = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Catalog cursor is invalid.") from exc
    if not token or len(token) > MAX_CURSOR_BYTES:
        raise ValueError("Catalog cursor is invalid.")
    return token.decode("utf-8", errors="strict")


def _encode_cursor(token: str) -> str:
    if not token or len(token.encode("utf-8")) > MAX_CURSOR_BYTES:
        raise CatalogResponseError
    return base64.urlsafe_b64encode(token.encode("utf-8")).decode("ascii").rstrip("=")


def _next_cursor(links: object) -> str | None:
    if not isinstance(links, list):
        return None
    next_link = next(
        (link for link in links if isinstance(link, dict) and link.get("rel") == "next"),
        None,
    )
    if next_link is None:
        return None
    href = next_link.get("href")
    if not isinstance(href, str):
        raise CatalogResponseError
    try:
        parsed = urlsplit(href)
        port = parsed.port
    except ValueError as exc:
        raise CatalogResponseError from exc
    method = next_link.get("method", "GET")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "stac.dataspace.copernicus.eu"
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v1/search"
        or not isinstance(method, str)
        or method.upper() != "POST"
    ):
        raise CatalogResponseError

    body = next_link.get("body")
    token = body.get("token") if isinstance(body, dict) else None
    if token is None:
        token = parse_qs(parsed.query).get("token", [None])[0]
    if not isinstance(token, str) or not token:
        raise CatalogResponseError
    return _encode_cursor(token)


def _utc_datetime(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _cloud_cover(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) and 0 <= normalized <= 100 else None


def _bbox(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    coordinates = value[:4]
    if any(
        isinstance(coordinate, bool)
        or not isinstance(coordinate, (int, float))
        or not math.isfinite(float(coordinate))
        for coordinate in coordinates
    ):
        return None
    west, south, east, north = map(float, coordinates)
    if (
        west < -180
        or east > 180
        or south < -90
        or north > 90
        or west > east
        or south > north
    ):
        return None
    return [west, south, east, north]


def _geometry(value: object) -> dict | None:
    if not isinstance(value, dict) or value.get("type") not in {
        "Polygon",
        "MultiPolygon",
    }:
        return None
    coordinates = value.get("coordinates")
    stack = [(coordinates, 0)]
    coordinate_count = 0
    visited_nodes = 0
    while stack:
        current, depth = stack.pop()
        visited_nodes += 1
        if visited_nodes > MAX_FEATURE_GEOMETRY_COORDINATES * 4 or depth > 12:
            return None
        if not isinstance(current, list):
            return None
        if len(current) >= 2 and all(
            isinstance(component, (int, float)) and not isinstance(component, bool)
            for component in current[:2]
        ):
            coordinate_count += 1
            if coordinate_count > MAX_FEATURE_GEOMETRY_COORDINATES:
                return None
        else:
            stack.extend((child, depth + 1) for child in current)
    try:
        geometry = shape(value)
        if geometry.is_empty or not geometry.is_valid:
            return None
    except (TypeError, ValueError, GEOSException):
        return None
    return value


def _spectral_assets(
    value: object, *, include_asset_urls: bool = False
) -> tuple[list[dict], bool, int | None]:
    if not isinstance(value, dict):
        return [], False, None

    normalized: list[dict] = []
    asset_by_key: dict[str, dict] = {}
    for key, asset in value.items():
        match = SPECTRAL_ASSET_PATTERN.fullmatch(key) if isinstance(key, str) else None
        if not match or not isinstance(asset, dict):
            continue
        size = asset.get("file:size")
        size_bytes = (
            size
            if isinstance(size, int) and not isinstance(size, bool) and size > 0
            else None
        )
        gsd = asset.get("gsd")
        if (
            isinstance(gsd, bool)
            or not isinstance(gsd, (int, float))
            or not math.isfinite(float(gsd))
            or gsd <= 0
        ):
            gsd = None
        roles = asset.get("roles")
        media_type = asset.get("type")
        href = asset.get("href")
        checksum = asset.get("file:checksum") or asset.get("checksum:multihash")
        try:
            parsed_href = urlsplit(href) if isinstance(href, str) else None
        except ValueError:
            parsed_href = None
        listed_for_acquisition = (
            key in ACQUISITION_ASSET_KEYS
            and media_type == "image/jp2"
            and parsed_href is not None
            and parsed_href.scheme == "s3"
            and parsed_href.netloc == "eodata"
            and not parsed_href.query
            and not parsed_href.fragment
            and parsed_href.path.startswith("/Sentinel-2/MSI/L2A/")
            and parsed_href.path.endswith(f"_{match.group(1)}_{match.group(2)}m.jp2")
        )
        metadata = {
            "key": key,
            "title": asset.get("title") if isinstance(asset.get("title"), str) else None,
            "media_type": media_type if isinstance(media_type, str) else None,
            "roles": [role for role in roles if isinstance(role, str)]
            if isinstance(roles, list)
            else [],
            "gsd": float(gsd) if gsd is not None else None,
            "size_bytes": size_bytes,
            "listed_for_acquisition": listed_for_acquisition,
        }
        if isinstance(checksum, str):
            metadata["checksum"] = checksum[:256]
        if (
            include_asset_urls
            and listed_for_acquisition
            and isinstance(href, str)
            and len(href) <= 2048
        ):
            metadata["href"] = href
        normalized.append(metadata)
        asset_by_key[key] = metadata

    normalized.sort(key=lambda asset: asset["key"])
    required = [asset_by_key.get(key) for key in ACQUISITION_ASSET_KEYS]
    acquisition_supported = bool(
        all(
            asset is not None
            and asset["listed_for_acquisition"]
            for asset in required
        )
    )
    acquisition_bytes = (
        sum(asset["size_bytes"] for asset in required if asset is not None)
        if acquisition_supported
        and all(asset["size_bytes"] is not None for asset in required if asset)
        else None
    )
    return normalized, acquisition_supported, acquisition_bytes


def _normalize_feature(
    feature: object, *, include_asset_urls: bool = False
) -> dict:
    if not isinstance(feature, dict):
        raise CatalogResponseError
    item_id = feature.get("id")
    collection = feature.get("collection")
    properties = feature.get("properties")
    if (
        not isinstance(item_id, str)
        or not item_id
        or len(item_id) > 512
        or not isinstance(properties, dict)
    ):
        raise CatalogResponseError
    if collection is not None and collection != STAC_COLLECTION:
        raise CatalogResponseError

    product_type = properties.get("product:type")
    platform = properties.get("platform")
    available_assets, acquisition_supported, acquisition_bytes = _spectral_assets(
        feature.get("assets"), include_asset_urls=include_asset_urls
    )
    normalized = {
        "id": item_id,
        "collection": STAC_COLLECTION,
        "satellite": platform if isinstance(platform, str) else None,
        "acquisition_datetime": _utc_datetime(properties.get("datetime")),
        "cloud_cover": _cloud_cover(properties.get("eo:cloud_cover")),
        "bbox": _bbox(feature.get("bbox")),
        "footprint": _geometry(feature.get("geometry")),
        "processing_level": product_type if isinstance(product_type, str) else None,
        "available_assets": available_assets,
        "acquisition_supported": acquisition_supported,
        "acquisition_bytes": acquisition_bytes,
        "catalog_url": (
            f"https://stac.dataspace.copernicus.eu/v1/collections/"
            f"{STAC_COLLECTION}/items/{quote(item_id, safe='')}"
        ),
        "provider": PROVIDER_NAME,
    }
    published_datetime = _utc_datetime(
        properties.get("published") or properties.get("created")
    )
    if published_datetime is not None:
        normalized["published_datetime"] = published_datetime
    return normalized


def search_sentinel2_catalog(
    request: SentinelCatalogSearchRequest,
    *,
    include_asset_urls: bool = False,
) -> dict:
    token = _decode_cursor(request.cursor)
    end_datetime = (
        datetime.combine(request.end_date, time.max, timezone.utc)
        if request.end_date == date.max
        else datetime.combine(
            request.end_date + timedelta(days=1),
            time.min,
            timezone.utc,
        )
    )
    payload: dict = {
        "collections": [STAC_COLLECTION],
        "intersects": request.aoi.model_dump(),
        "datetime": (
            f"{datetime.combine(request.start_date, time.min, timezone.utc).isoformat()}"
            f"/{end_datetime.isoformat()}"
        ),
        "query": {"eo:cloud_cover": {"lte": request.max_cloud_cover}},
        "limit": min(request.limit, MAX_PAGE_SIZE),
    }
    if token is not None:
        payload["token"] = token

    try:
        response = httpx.post(
            STAC_SEARCH_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            follow_redirects=False,
        )
    except httpx.TimeoutException as exc:
        raise CatalogTimeoutError from exc
    except httpx.RequestError as exc:
        raise CatalogError from exc

    if response.status_code == 429:
        raise CatalogRateLimitError
    if response.status_code >= 500:
        raise CatalogError
    if response.status_code != 200:
        raise CatalogResponseError
    try:
        body = response.json()
    except (ValueError, TypeError) as exc:
        raise CatalogResponseError from exc
    if not isinstance(body, dict) or body.get("type") != "FeatureCollection":
        raise CatalogResponseError
    features = body.get("features")
    if not isinstance(features, list) or len(features) > request.limit:
        raise CatalogResponseError

    results = [
        _normalize_feature(feature, include_asset_urls=include_asset_urls)
        for feature in features
    ]
    next_cursor = _next_cursor(body.get("links"))
    return {
        "provider": PROVIDER_NAME,
        "collection": STAC_COLLECTION,
        "result_count": len(results),
        "limit": request.limit,
        "has_more": next_cursor is not None,
        "next_cursor": next_cursor,
        "results": results,
    }
