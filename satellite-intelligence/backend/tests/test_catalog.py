from dataclasses import replace
from datetime import date
import math

import httpx
import pytest
from fastapi.testclient import TestClient

import auth
import main
from api import routes
from config import settings
from main import app
from satellite import catalog, models
from satellite.models import GeoJSONPolygon, SentinelCatalogSearchRequest

client = TestClient(app)


def sample_aoi() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [77.0, 12.0],
            [77.0, 12.1],
            [77.1, 12.1],
            [77.1, 12.0],
            [77.0, 12.0],
        ]],
    }


def sample_request(**changes) -> SentinelCatalogSearchRequest:
    values = {
        "aoi": GeoJSONPolygon(**sample_aoi()),
        "start_date": date(2025, 1, 1),
        "end_date": date(2025, 1, 31),
        "max_cloud_cover": 20,
        "limit": 2,
    }
    values.update(changes)
    return SentinelCatalogSearchRequest(**values)


def sample_feature(item_id: str = "S2A_TEST") -> dict:
    return {
        "type": "Feature",
        "id": item_id,
        "collection": "sentinel-2-l2a",
        "geometry": sample_aoi(),
        "bbox": [77.0, 12.0, 77.1, 12.1],
        "properties": {
            "datetime": "2025-01-27T05:10:09.024+01:00",
            "eo:cloud_cover": 6.94,
            "platform": "sentinel-2a",
            "product:type": "S2MSI2A",
        },
    }


def stac_response(features: list, links: list | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={"type": "FeatureCollection", "features": features, "links": links or []},
    )


def test_catalog_search_posts_valid_bounded_search_and_normalizes_metadata(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return stac_response([sample_feature()])

    monkeypatch.setattr(catalog.httpx, "post", post)
    result = catalog.search_sentinel2_catalog(sample_request())

    assert captured["url"] == catalog.STAC_SEARCH_URL
    assert captured["json"]["collections"] == ["sentinel-2-l2a"]
    assert captured["json"]["query"] == {"eo:cloud_cover": {"lte": 20}}
    assert captured["json"]["limit"] == 2
    assert captured["follow_redirects"] is False
    assert captured["timeout"] == catalog.REQUEST_TIMEOUT
    assert result["provider"] == "Copernicus Data Space Ecosystem"
    assert result["results"][0] == {
        "id": "S2A_TEST",
        "collection": "sentinel-2-l2a",
        "satellite": "sentinel-2a",
        "acquisition_datetime": "2025-01-27T04:10:09.024Z",
        "cloud_cover": 6.94,
        "bbox": [77.0, 12.0, 77.1, 12.1],
        "footprint": sample_aoi(),
        "processing_level": "S2MSI2A",
        "available_assets": [],
        "acquisition_supported": False,
        "acquisition_bytes": None,
        "catalog_url": (
            "https://stac.dataspace.copernicus.eu/v1/collections/"
            "sentinel-2-l2a/items/S2A_TEST"
        ),
        "provider": "Copernicus Data Space Ecosystem",
    }


def test_catalog_exposes_listed_spectral_assets_and_acquisition_estimate(monkeypatch):
    feature = sample_feature("S2C_LIVE_FORMAT")
    feature["assets"] = {
        "B02_10m": {
            "title": "Blue (band 2) - 10m",
            "type": "image/jp2",
            "roles": ["data", "reflectance"],
            "gsd": 10,
            "file:size": 100,
            "href": "s3://eodata/Sentinel-2/MSI/L2A/scene/T43PHN_20250201T051101_B02_10m.jp2",
        },
        "B03_10m": {
            "title": "Green (band 3) - 10m",
            "type": "image/jp2",
            "roles": ["data", "reflectance"],
            "gsd": 10,
            "file:size": 200,
            "href": "s3://eodata/Sentinel-2/MSI/L2A/scene/T43PHN_20250201T051101_B03_10m.jp2",
        },
        "B04_10m": {
            "title": "Red (band 4) - 10m",
            "type": "image/jp2",
            "roles": ["data", "reflectance"],
            "gsd": 10,
            "file:size": 300,
            "href": "s3://eodata/Sentinel-2/MSI/L2A/scene/T43PHN_20250201T051101_B04_10m.jp2",
        },
        "B08_10m": {
            "title": "NIR (band 8) - 10m",
            "type": "image/jp2",
            "roles": ["data", "reflectance"],
            "gsd": 10,
            "file:size": 400,
            "href": "s3://eodata/Sentinel-2/MSI/L2A/scene/T43PHN_20250201T051101_B08_10m.jp2",
        },
        "B11_20m": {
            "title": "SWIR (band 11) - 20m",
            "type": "image/jp2",
            "roles": ["data", "reflectance"],
            "gsd": 20,
            "file:size": 500,
            "href": "s3://eodata/Sentinel-2/MSI/L2A/scene/T43PHN_20250201T051101_B11_20m.jp2",
        },
    }
    monkeypatch.setattr(
        catalog.httpx,
        "post",
        lambda *_args, **_kwargs: stac_response([feature]),
    )

    scene = catalog.search_sentinel2_catalog(sample_request())["results"][0]

    assert [asset["key"] for asset in scene["available_assets"]] == [
        "B02_10m",
        "B03_10m",
        "B04_10m",
        "B08_10m",
        "B11_20m",
    ]
    assert scene["acquisition_supported"] is True
    assert scene["acquisition_bytes"] == 1000
    assert "href" not in scene["available_assets"][0]

    feature["assets"]["B08_10m"]["href"] = "https://attacker.invalid/band.jp2"
    unsupported = catalog._normalize_feature(feature)
    assert unsupported["acquisition_supported"] is False
    assert unsupported["acquisition_bytes"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {
            **sample_aoi(),
            "coordinates": [[
                [181, 12],
                [181, 12.1],
                [77.1, 12.1],
                [77.1, 12],
                [181, 12],
            ]],
        },
        {
            "type": "Polygon",
            "coordinates": [[
                [77, 12],
                [78, 12.1],
                [77, 12.1],
                [78, 12],
                [77, 12],
            ]],
        },
        {
            "type": "Polygon",
            "coordinates": [[
                [True, 12],
                [77, 12.1],
                [77.1, 12.1],
                [77.1, 12],
                [True, 12],
            ]],
        },
        {
            "type": "Polygon",
            "coordinates": [[
                [0, 0],
                [0, 20],
                [20, 20],
                [20, 0],
                [0, 0],
            ]],
        },
    ],
)
def test_catalog_route_rejects_invalid_or_excessive_geometry(payload, monkeypatch):
    monkeypatch.setattr(
        routes,
        "search_sentinel2_catalog",
        lambda _request: pytest.fail("Invalid geometry reached the catalog."),
    )
    response = client.post(
        "/api/satellite/catalog/search",
        json={
            "aoi": payload,
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        },
    )
    assert response.status_code == 422


def test_catalog_route_rejects_reversed_dates_cloud_and_result_limits(monkeypatch):
    monkeypatch.setattr(
        routes,
        "search_sentinel2_catalog",
        lambda _request: pytest.fail("Invalid search reached the catalog."),
    )
    base = {
        "aoi": sample_aoi(),
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
    }
    for overrides in (
        {"start_date": "2025-02-01", "end_date": "2025-01-31"},
        {"max_cloud_cover": -1},
        {"max_cloud_cover": 101},
        {"limit": 0},
        {"limit": 51},
    ):
        response = client.post(
            "/api/satellite/catalog/search",
            json={**base, **overrides},
        )
        assert response.status_code == 422


def test_catalog_request_rejects_too_many_aoi_coordinates():
    ring = [
        [77 + 0.01 * math.cos(index * math.tau / 500), 12 + 0.01 * math.sin(index * math.tau / 500)]
        for index in range(500)
    ]
    ring.append(ring[0])
    with pytest.raises(ValueError, match="500 coordinates"):
        sample_request(aoi=GeoJSONPolygon(type="Polygon", coordinates=[ring]))


def test_catalog_route_rejects_too_many_coordinates_before_geometry_processing(monkeypatch):
    ring = [
        [
            77 + 0.01 * math.cos(index * math.tau / 500),
            12 + 0.01 * math.sin(index * math.tau / 500),
        ]
        for index in range(500)
    ]
    ring.append(ring[0])

    def fail_if_catalog_is_called(request):
        pytest.fail(f"Oversized geometry reached the catalog: {request!r}")

    monkeypatch.setattr(
        routes,
        "search_sentinel2_catalog",
        fail_if_catalog_is_called,
    )
    response = client.post(
        "/api/satellite/catalog/search",
        json={
            "aoi": {"type": "Polygon", "coordinates": [ring]},
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        },
    )
    assert response.status_code == 422


def test_catalog_request_rejects_large_geometry_before_shapely_processing(monkeypatch):
    ring = [
        [
            77 + 0.01 * math.cos(index * math.tau / 500),
            12 + 0.01 * math.sin(index * math.tau / 500),
        ]
        for index in range(500)
    ]
    ring.append(ring[0])

    def fail_if_shapely_is_called(geometry):
        pytest.fail(f"Large geometry reached Shapely: {geometry!r}")

    monkeypatch.setattr(
        models,
        "shape",
        fail_if_shapely_is_called,
    )
    with pytest.raises(ValueError, match="500 coordinates"):
        SentinelCatalogSearchRequest.model_validate(
            {
                "aoi": {"type": "Polygon", "coordinates": [ring]},
                "start_date": date(2025, 1, 1),
                "end_date": date(2025, 1, 31),
            }
        )


def test_catalog_search_handles_maximum_calendar_date(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return stac_response([])

    monkeypatch.setattr(catalog.httpx, "post", post)
    request = sample_request(start_date=date.max, end_date=date.max)
    catalog.search_sentinel2_catalog(request)
    assert captured["url"] == catalog.STAC_SEARCH_URL
    assert captured["json"]["datetime"].endswith(
        "/9999-12-31T23:59:59.999999+00:00"
    )


def test_empty_catalog_result_is_a_successful_empty_page(monkeypatch):
    monkeypatch.setattr(catalog.httpx, "post", lambda *_args, **_kwargs: stac_response([]))
    result = catalog.search_sentinel2_catalog(sample_request())
    assert result["results"] == []
    assert result["result_count"] == 0
    assert result["next_cursor"] is None
    assert result["has_more"] is False


@pytest.mark.parametrize(
    ("response", "exception"),
    [
        (httpx.Response(429), catalog.CatalogRateLimitError),
        (httpx.Response(503), catalog.CatalogError),
        (httpx.Response(400), catalog.CatalogResponseError),
        (httpx.Response(200, text="not-json"), catalog.CatalogResponseError),
        (
            httpx.Response(200, json={"type": "FeatureCollection", "features": {}}),
            catalog.CatalogResponseError,
        ),
    ],
)
def test_catalog_upstream_errors_are_explicit(monkeypatch, response, exception):
    monkeypatch.setattr(catalog.httpx, "post", lambda *_args, **_kwargs: response)
    with pytest.raises(exception):
        catalog.search_sentinel2_catalog(sample_request())


def test_catalog_timeout_is_reported(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise httpx.ReadTimeout("upstream timeout")

    monkeypatch.setattr(catalog.httpx, "post", timeout)
    with pytest.raises(catalog.CatalogTimeoutError):
        catalog.search_sentinel2_catalog(sample_request())


def test_missing_optional_metadata_is_normalized_as_null(monkeypatch):
    feature = {
        "type": "Feature",
        "id": "S2_OPTIONAL",
        "properties": {},
    }
    monkeypatch.setattr(
        catalog.httpx,
        "post",
        lambda *_args, **_kwargs: stac_response([feature]),
    )
    scene = catalog.search_sentinel2_catalog(sample_request())["results"][0]
    assert scene["satellite"] is None
    assert scene["acquisition_datetime"] is None
    assert scene["cloud_cover"] is None
    assert scene["bbox"] is None
    assert scene["footprint"] is None
    assert scene["processing_level"] is None
    assert scene["id"] == "S2_OPTIONAL"


def test_catalog_pagination_passes_opaque_cursor_to_fixed_search_endpoint(monkeypatch):
    calls = []
    first_page = stac_response(
        [sample_feature("S2_FIRST")],
        [{
            "rel": "next",
            "method": "POST",
            "href": catalog.STAC_SEARCH_URL,
            "body": {"token": "next-page-token"},
        }],
    )

    def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return first_page if len(calls) == 1 else stac_response([sample_feature("S2_SECOND")])

    monkeypatch.setattr(catalog.httpx, "post", post)
    first = catalog.search_sentinel2_catalog(sample_request())
    second = catalog.search_sentinel2_catalog(
        sample_request(cursor=first["next_cursor"])
    )
    assert first["has_more"] is True
    assert first["next_cursor"]
    assert second["has_more"] is False
    assert second["results"][0]["id"] == "S2_SECOND"
    assert calls[1][0] == catalog.STAC_SEARCH_URL
    assert calls[1][1]["token"] == "next-page-token"


def test_catalog_rejects_untrusted_pagination_url(monkeypatch):
    response = stac_response(
        [],
        [{
            "rel": "next",
            "method": "POST",
            "href": "https://example.invalid/v1/search",
            "body": {"token": "opaque"},
        }],
    )
    monkeypatch.setattr(
        catalog.httpx,
        "post",
        lambda *_args, **_kwargs: response,
    )
    with pytest.raises(catalog.CatalogResponseError):
        catalog.search_sentinel2_catalog(sample_request())


def test_catalog_endpoint_requires_authentication_when_enabled(monkeypatch):
    protected_settings = replace(
        settings,
        require_auth=True,
        database_url=None,
        supabase_url=None,
        supabase_anon_key=None,
        supabase_service_role_key=None,
    )
    monkeypatch.setattr(main, "settings", protected_settings)
    monkeypatch.setattr(auth, "settings", protected_settings)
    response = client.post(
        "/api/satellite/catalog/search",
        json={
            "aoi": sample_aoi(),
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        },
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "A valid Supabase sign-in is required."


def test_catalog_endpoint_accepts_authenticated_request(monkeypatch):
    protected_settings = replace(
        settings,
        require_auth=True,
        database_url=None,
        supabase_url=None,
        supabase_anon_key=None,
        supabase_service_role_key=None,
    )
    monkeypatch.setattr(main, "settings", protected_settings)
    monkeypatch.setattr(auth, "settings", protected_settings)
    monkeypatch.setattr(
        main,
        "verify_access_token",
        lambda _token: auth.AuthenticatedUser("verified-user"),
    )
    monkeypatch.setattr(
        routes,
        "search_sentinel2_catalog",
        lambda request: {
            "provider": catalog.PROVIDER_NAME,
            "collection": catalog.STAC_COLLECTION,
            "result_count": 0,
            "limit": request.limit,
            "has_more": False,
            "next_cursor": None,
            "results": [],
        },
    )
    response = client.post(
        "/api/satellite/catalog/search",
        headers={"Authorization": "Bearer valid-session-token"},
        json={
            "aoi": sample_aoi(),
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        },
    )
    assert response.status_code == 200
    assert response.json()["provider"] == catalog.PROVIDER_NAME


def test_catalog_endpoint_returns_normalized_search_result(monkeypatch):
    monkeypatch.setattr(
        routes,
        "search_sentinel2_catalog",
        lambda request: {
            "provider": catalog.PROVIDER_NAME,
            "collection": catalog.STAC_COLLECTION,
            "result_count": 1,
            "limit": request.limit,
            "has_more": False,
            "next_cursor": None,
            "results": [{
                "id": "S2A_ROUTE",
                "collection": catalog.STAC_COLLECTION,
                "satellite": None,
                "acquisition_datetime": None,
                "cloud_cover": None,
                "bbox": None,
                "footprint": None,
                "processing_level": None,
                "catalog_url": "https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items/S2A_ROUTE",
                "provider": catalog.PROVIDER_NAME,
            }],
        },
    )
    response = client.post(
        "/api/satellite/catalog/search",
        json={
            "aoi": sample_aoi(),
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        },
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "S2A_ROUTE"
    assert response.json()["results"][0]["cloud_cover"] is None
