from dataclasses import replace
from pathlib import Path
from uuid import UUID

import numpy as np
import httpx
import pytest
import rasterio
from fastapi.testclient import TestClient
from pyproj import Transformer
from rasterio.transform import from_origin

import api.routes as routes
import main as application
from auth import AuthenticatedUser
from config import settings
from main import app
from persistence.artifacts import LocalArtifactStore
from persistence.database import PersistenceRepository
from persistence.manager import PersistenceManager
from satellite import acquisition
from satellite import catalog
from satellite.models import SentinelSceneAcquisitionRequest

client = TestClient(app)
SCENE_ID = "S2C_MSIL2A_20250201T051101_N0511_R019_T43PHN_20250201T095810"
PRODUCT_UUID = "f459defd-6279-4ce1-9edc-cfd73a1e6057"
GRANULE = "L2A_T43PHN_A002134_20250201T052343"


def aoi_for_test_raster() -> dict:
    project = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    ring = [
        project.transform(x, y)
        for x, y in (
            (500_010, 1_499_970),
            (500_060, 1_499_970),
            (500_060, 1_500_000),
            (500_010, 1_500_000),
            (500_010, 1_499_970),
        )
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def acquisition_request(aoi: dict | None = None) -> SentinelSceneAcquisitionRequest:
    return SentinelSceneAcquisitionRequest(
        provider="Copernicus Data Space Ecosystem",
        collection="sentinel-2-l2a",
        aoi=aoi
        or {
            "type": "Polygon",
            "coordinates": [
                [[77.0, 12.0], [77.0, 12.1], [77.1, 12.1], [77.1, 12.0], [77.0, 12.0]]
            ],
        },
        scene_id=SCENE_ID,
    )


def sample_item() -> dict:
    assets = {
        "Product": {
            "href": (
                "https://download.dataspace.copernicus.eu/odata/v1/Products("
                f"{PRODUCT_UUID})/$value"
            )
        }
    }
    for band in acquisition.BANDS:
        assets[f"{band}_10m"] = {
            "href": (
                "s3://eodata/Sentinel-2/MSI/L2A/2025/02/01/"
                f"{SCENE_ID}.SAFE/GRANULE/{GRANULE}/IMG_DATA/R10m/"
                f"T43PHN_20250201T051101_{band}_10m.jp2"
            )
        }
    return {
        "type": "Feature",
        "id": SCENE_ID,
        "collection": "sentinel-2-l2a",
        "properties": {
            "datetime": "2025-02-01T05:11:01Z",
            "platform": "sentinel-2c",
            "product:type": "S2MSI2A",
        },
        "assets": assets,
    }


def make_geotiff(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=6,
        count=4,
        dtype="uint16",
        crs="EPSG:32643",
        transform=from_origin(500_000, 1_500_000, 10, 10),
        nodata=0,
    ) as destination:
        destination.write(
            np.stack(
                [
                    np.arange(48, dtype=np.uint16).reshape(6, 8) + offset
                    for offset in (10, 20, 30, 40)
                ]
            )
        )
        for index, band in enumerate(acquisition.BANDS, start=1):
            destination.set_band_description(index, band)
        destination.update_tags(
            PLATFORM="sentinel-2c",
            ACQUISITION_DATE="2025-02-01",
            STAC_PROVIDER=acquisition.PROVIDER,
            STAC_COLLECTION=acquisition.COLLECTION,
            STAC_ITEM_ID=SCENE_ID,
            STAC_ITEM_URL=acquisition.STAC_ITEM_URL.format(scene_id=SCENE_ID),
            CDSE_PRODUCT_UUID=PRODUCT_UUID,
            PROCESSING_LEVEL="S2MSI2A",
            ASSET_BANDS=",".join(acquisition.BANDS),
        )


def jp2_available() -> bool:
    with rasterio.Env() as environment:
        return "JP2OpenJPEG" in environment.drivers()


@pytest.fixture
def acquisition_backend(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    output_dir = data_dir / "outputs"
    output_dir.mkdir(parents=True)
    local_settings = replace(
        settings,
        data_dir=data_dir,
        output_dir=output_dir,
        database_url=None,
        supabase_url=None,
        supabase_anon_key=None,
        supabase_service_role_key=None,
        require_auth=False,
    )
    repository = PersistenceRepository(f"sqlite:///{tmp_path / 'acquisition.sqlite'}")
    repository.create_schema_for_tests()
    manager = PersistenceManager(
        local_settings,
        repository=repository,
        artifact_store=LocalArtifactStore(output_dir),
    )
    monkeypatch.setattr(routes, "settings", local_settings)
    monkeypatch.setattr(routes, "persistence_manager", manager)
    monkeypatch.setattr(application, "settings", local_settings)
    app.dependency_overrides[routes.current_user] = lambda: AuthenticatedUser("user-1")
    yield manager
    app.dependency_overrides.pop(routes.current_user, None)
    manager.close()


def test_acquisition_request_accepts_stable_scene_id_and_rejects_arbitrary_inputs():
    assert acquisition_request().scene_id == SCENE_ID
    assert acquisition_request().scene_id.startswith("S2C_")
    with pytest.raises(ValueError):
        SentinelSceneAcquisitionRequest(
            provider="Copernicus Data Space Ecosystem",
            collection="sentinel-2-l2a",
            aoi=aoi_for_test_raster(),
            scene_id="https://attacker.example/product",
        )
    with pytest.raises(ValueError):
        SentinelSceneAcquisitionRequest(
            provider="other",
            collection="sentinel-2-l2a",
            aoi=aoi_for_test_raster(),
            scene_id=SCENE_ID,
        )


def test_item_assets_are_limited_to_validated_official_product_and_four_bands():
    product_uuid, assets = acquisition._asset_references(sample_item(), SCENE_ID)

    assert str(UUID(product_uuid)) == PRODUCT_UUID
    assert set(assets) == set(acquisition.BANDS)
    assert all(asset.endswith(f"_{band}_10m.jp2") for band, asset in assets.items())

    invalid_item = sample_item()
    invalid_item["assets"]["B02_10m"]["href"] = "https://127.0.0.1/private"
    with pytest.raises(acquisition.AcquisitionResponseError):
        acquisition._asset_references(invalid_item, SCENE_ID)
    del invalid_item["assets"]["B08_10m"]
    with pytest.raises(acquisition.AcquisitionResponseError):
        acquisition._asset_references(invalid_item, SCENE_ID)


def test_download_retries_rate_limits_and_validates_jp2_content(monkeypatch, tmp_path):
    payload = acquisition.JP2_SIGNATURE + b"valid-test-payload"
    calls = []

    class Response:
        headers = {
            "content-length": str(len(payload)),
            "content-type": "image/jp2",
        }

        def __init__(self, status_code):
            self.status_code = status_code

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self, _chunk_size):
            yield payload

    def stream(_method, _url, **kwargs):
        calls.append(kwargs["headers"])
        return Response(429 if len(calls) == 1 else 200)

    monkeypatch.setattr(acquisition, "_validate_url", lambda _url: None)
    monkeypatch.setattr(acquisition.httpx, "stream", stream)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _duration: None)
    destination = tmp_path / "B02.jp2"

    transferred = acquisition._download_file(
        PRODUCT_UUID,
        "scene/IMG_DATA/R10m/T43PHN_B02_10m.jp2",
        destination,
        token="mock-provider-token",
        max_bytes=100,
    )

    assert transferred == len(payload)
    assert destination.read_bytes() == payload
    assert calls == [
        {"Authorization": "Bearer mock-provider-token"},
        {"Authorization": "Bearer mock-provider-token"},
    ]


def test_download_rejects_excessive_and_invalid_asset_content(monkeypatch, tmp_path):
    class Response:
        def __init__(self, status_code, headers, payload=b""):
            self.status_code = status_code
            self.headers = headers
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self, _chunk_size):
            yield self.payload

    monkeypatch.setattr(acquisition, "_validate_url", lambda _url: None)
    destination = tmp_path / "B02.jp2"
    monkeypatch.setattr(
        acquisition.httpx,
        "stream",
        lambda *_args, **_kwargs: Response(
            200,
            {"content-length": "101", "content-type": "image/jp2"},
        ),
    )
    with pytest.raises(acquisition.AcquisitionSizeError):
        acquisition._download_file(
            PRODUCT_UUID,
            "scene/IMG_DATA/R10m/T43PHN_B02_10m.jp2",
            destination,
            token="mock-provider-token",
            max_bytes=100,
        )

    monkeypatch.setattr(
        acquisition.httpx,
        "stream",
        lambda *_args, **_kwargs: Response(
            200,
            {"content-length": "12", "content-type": "image/jpeg"},
            b"not a jp2 file",
        ),
    )
    with pytest.raises(acquisition.AcquisitionResponseError):
        acquisition._download_file(
            PRODUCT_UUID,
            "scene/IMG_DATA/R10m/T43PHN_B02_10m.jp2",
            destination,
            token="mock-provider-token",
            max_bytes=100,
        )


def test_download_times_out_after_bounded_retries(monkeypatch, tmp_path):
    calls = []

    def timeout(*_args, **_kwargs):
        calls.append(True)
        raise httpx.ReadTimeout("upstream timeout")

    monkeypatch.setattr(acquisition, "_validate_url", lambda _url: None)
    monkeypatch.setattr(acquisition.httpx, "stream", timeout)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _duration: None)

    with pytest.raises(acquisition.AcquisitionTimeoutError):
        acquisition._download_file(
            PRODUCT_UUID,
            "scene/IMG_DATA/R10m/T43PHN_B02_10m.jp2",
            tmp_path / "B02.jp2",
            token="mock-provider-token",
            max_bytes=100,
        )
    assert len(calls) == acquisition.MAX_RETRIES + 1


def test_acquisition_requires_server_credentials_and_uses_documented_token_grant(monkeypatch):
    monkeypatch.setattr(
        acquisition,
        "settings",
        replace(settings, cdse_username=None, cdse_password=None),
    )
    with pytest.raises(acquisition.AcquisitionConfigurationError):
        acquisition._access_token()

    monkeypatch.setattr(
        acquisition,
        "settings",
        replace(settings, cdse_username="account", cdse_password="backend-secret"),
    )
    monkeypatch.setattr(acquisition, "_assert_official_public_host", lambda host: None)
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured["data"] = kwargs["data"]
        return acquisition.httpx.Response(200, json={"access_token": "short-lived-token"})

    monkeypatch.setattr(acquisition.httpx, "post", post)
    assert acquisition._access_token() == "short-lived-token"
    assert captured["url"] == acquisition.TOKEN_URL
    assert captured["data"] == {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": "account",
        "password": "backend-secret",
    }


def test_acquisition_response_errors_are_explicit():
    assert isinstance(acquisition._response_error(429), acquisition.AcquisitionRateLimitError)
    assert isinstance(acquisition._response_error(403), acquisition.AcquisitionAuthorizationError)
    assert isinstance(acquisition._response_error(503), acquisition.AcquisitionUnavailableError)
    with pytest.raises(acquisition.AcquisitionSizeError):
        acquisition._check_content_length(str(20), 10)
    with pytest.raises(acquisition.AcquisitionResponseError):
        acquisition._check_content_length("not-a-number", 10)


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (429, acquisition.AcquisitionRateLimitError),
        (401, acquisition.AcquisitionAuthorizationError),
        (503, acquisition.AcquisitionUnavailableError),
    ],
)
def test_catalog_item_read_maps_upstream_status_without_exposing_response(
    monkeypatch,
    status_code,
    error_type,
):
    class Response:
        headers = {}

        def __init__(self, status):
            self.status_code = status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(acquisition, "_assert_official_public_host", lambda host: None)
    monkeypatch.setattr(
        acquisition.httpx,
        "stream",
        lambda *args, **kwargs: Response(status_code),
    )

    with pytest.raises(error_type):
        acquisition._read_json(acquisition.STAC_ITEM_URL.format(scene_id=SCENE_ID))


def test_catalog_item_read_handles_timeout_malformed_json_and_untrusted_hosts(monkeypatch):
    monkeypatch.setattr(acquisition, "_assert_official_public_host", lambda host: None)

    def timeout(*args, **kwargs):
        raise acquisition.httpx.ReadTimeout(
            "timed out",
            request=acquisition.httpx.Request("GET", acquisition.STAC_ITEM_URL),
        )

    monkeypatch.setattr(acquisition.httpx, "stream", timeout)
    with pytest.raises(acquisition.AcquisitionTimeoutError):
        acquisition._read_json(acquisition.STAC_ITEM_URL.format(scene_id=SCENE_ID))

    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_bytes(self):
            yield b"<invalid-json>"

    monkeypatch.setattr(
        acquisition.httpx,
        "stream",
        lambda *args, **kwargs: Response(),
    )
    with pytest.raises(acquisition.AcquisitionResponseError):
        acquisition._read_json(acquisition.STAC_ITEM_URL.format(scene_id=SCENE_ID))

    with pytest.raises(acquisition.AcquisitionResponseError):
        acquisition._validate_url("https://127.0.0.1/private")


def test_acquisition_route_requires_an_authenticated_user(acquisition_backend, monkeypatch):
    app.dependency_overrides[routes.current_user] = lambda: None
    monkeypatch.setattr(
        routes.satellite_acquisition,
        "acquire_scene",
        lambda *args, **kwargs: pytest.fail("Unauthenticated acquisition reached the provider."),
    )

    response = client.post(
        "/api/satellite/catalog/acquire",
        json={
            "provider": "Copernicus Data Space Ecosystem",
            "collection": "sentinel-2-l2a",
            "aoi": aoi_for_test_raster(),
            "scene_id": SCENE_ID,
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Sign in to acquire and save a Copernicus scene."


def test_failed_acquisition_cleans_staging_directory(acquisition_backend, monkeypatch):
    temporary_directories = []
    original_temporary_directory = routes.tempfile.TemporaryDirectory

    def track_temporary_directory(*args, **kwargs):
        temporary_directory = original_temporary_directory(*args, **kwargs)
        temporary_directories.append(temporary_directory)
        return temporary_directory

    def fail_acquisition(_request, destination, **_kwargs):
        destination.write_bytes(b"partial acquisition")
        raise acquisition.AcquisitionUnavailableError()

    monkeypatch.setattr(routes.tempfile, "TemporaryDirectory", track_temporary_directory)
    monkeypatch.setattr(routes.satellite_acquisition, "acquire_scene", fail_acquisition)
    response = client.post(
        "/api/satellite/catalog/acquire",
        json={
            "provider": acquisition.PROVIDER,
            "collection": acquisition.COLLECTION,
            "aoi": aoi_for_test_raster(),
            "scene_id": SCENE_ID,
        },
    )

    assert response.status_code == 503
    assert temporary_directories
    assert all(not Path(directory.name).exists() for directory in temporary_directories)


def test_acquisition_route_rejects_area_above_configured_limit(
    acquisition_backend,
    monkeypatch,
):
    local_settings = replace(routes.settings, max_satellite_acquisition_area_km2=1)
    monkeypatch.setattr(routes, "settings", local_settings)
    monkeypatch.setattr(
        routes.satellite_acquisition,
        "acquire_scene",
        lambda *args, **kwargs: pytest.fail("Oversized AOI reached the provider."),
    )
    response = client.post(
        "/api/satellite/catalog/acquire",
        json={
            "provider": acquisition.PROVIDER,
            "collection": acquisition.COLLECTION,
            "aoi": acquisition_request().aoi.model_dump(),
            "scene_id": SCENE_ID,
        },
    )
    assert response.status_code == 422
    assert "AOI exceeds the configured 1 km² limit" in response.json()["detail"]


def test_acquired_scene_uses_existing_owner_scoped_ingestion_and_duplicate_protection(
    acquisition_backend,
    monkeypatch,
    tmp_path,
):
    calls = []

    def acquire(request, destination, **kwargs):
        calls.append(request.scene_id)
        make_geotiff(destination)
        return {"scene_id": request.scene_id, "bands": list(acquisition.BANDS)}

    monkeypatch.setattr(routes.satellite_acquisition, "acquire_scene", acquire)
    payload = {
        "provider": acquisition.PROVIDER,
        "collection": acquisition.COLLECTION,
        "aoi": aoi_for_test_raster(),
        "scene_id": SCENE_ID,
    }

    first = client.post("/api/satellite/catalog/acquire", json=payload)
    second = client.post("/api/satellite/catalog/acquire", json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["analysis"] == "imagery_ingestion"
    assert first.json()["metadata"]["catalog_source"]["item_id"] == SCENE_ID
    assert first.json()["catalog_acquisition"]["asset_bands"] == list(acquisition.BANDS)
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["duplicate"] is True
    assert calls == [SCENE_ID]
    record = acquisition_backend.get_analysis(first.json()["id"], owner_id="user-1")
    assert record is not None
    assert record["summary"]["metadata"]["catalog_source"]["catalog_url"].endswith(SCENE_ID)


def test_mocked_catalog_acquisition_ingests_and_analyzes_aoi_cropped_scene(
    acquisition_backend,
    monkeypatch,
    tmp_path,
):
    if not jp2_available():
        pytest.skip("The selected Rasterio environment has no JP2OpenJPEG decoder.")

    source_assets = {}
    for index, band in enumerate(acquisition.BANDS, start=1):
        source = tmp_path / f"source-{band}.jp2"
        with rasterio.open(
            source,
            "w",
            driver="JP2OpenJPEG",
            width=8,
            height=6,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=from_origin(500_000, 1_500_000, 10, 10),
            REVERSIBLE="YES",
        ) as destination:
            values = np.full((6, 8), index * 100, dtype=np.uint16)
            if band == "B02":
                values[1, 1] = 0
            destination.write(values, 1)
        source_assets[band] = source

    item = sample_item()
    catalog_feature = {
        **item,
        "geometry": aoi_for_test_raster(),
        "bbox": [75.0, 13.5, 75.001, 13.501],
        "properties": {
            **item["properties"],
            "eo:cloud_cover": 10.0,
        },
    }
    for band in acquisition.BANDS:
        catalog_feature["assets"][f"{band}_10m"].update(
            {
                "title": f"{band} reflectance at 10 m",
                "type": "image/jp2",
                "roles": ["data", "reflectance"],
                "gsd": 10,
                "file:size": 1_000,
            }
        )
    monkeypatch.setattr(
        catalog.httpx,
        "post",
        lambda *_args, **_kwargs: httpx.Response(
            200,
            json={
                "type": "FeatureCollection",
                "features": [catalog_feature],
                "links": [],
            },
        ),
    )
    search_response = client.post(
        "/api/satellite/catalog/search",
        json={
            "aoi": aoi_for_test_raster(),
            "start_date": "2025-02-01",
            "end_date": "2025-02-01",
            "max_cloud_cover": 100,
            "limit": 1,
        },
    )
    assert search_response.status_code == 200, search_response.text
    catalog_scene = search_response.json()["results"][0]
    assert catalog_scene["id"] == SCENE_ID
    assert catalog_scene["acquisition_supported"] is True
    assert catalog_scene["acquisition_bytes"] == 4_000

    safe_root = f"{SCENE_ID}.SAFE"
    product_nodes = {
        (): [{"Name": safe_root}],
        (safe_root,): [{"Name": "GRANULE"}],
        (safe_root, "GRANULE"): [{"Name": GRANULE}],
        (safe_root, "GRANULE", GRANULE): [{"Name": "IMG_DATA"}],
        (safe_root, "GRANULE", GRANULE, "IMG_DATA"): [{"Name": "R10m"}],
        (
            safe_root,
            "GRANULE",
            GRANULE,
            "IMG_DATA",
            "R10m",
        ): [
            {
                "Name": asset_path.split("/")[-1],
                "ContentLength": source_assets[band].stat().st_size,
            }
            for band, asset_path in acquisition._asset_references(item, SCENE_ID)[1].items()
        ],
    }
    monkeypatch.setattr(acquisition, "_access_token", lambda: "mock-provider-token")
    requested_headers = []

    def read_json(url, *, headers=None):
        requested_headers.append(headers)
        if url == acquisition.STAC_ITEM_URL.format(scene_id=SCENE_ID):
            return item
        for parents, nodes in product_nodes.items():
            if url == acquisition._node_url(PRODUCT_UUID, parents):
                return {"result": nodes}
        pytest.fail(f"Unexpected mocked Copernicus metadata URL: {url}")

    downloaded_bands = []

    def download_file(
        product_uuid,
        node_path,
        destination,
        *,
        token,
        max_bytes,
    ):
        band = destination.stem
        content = source_assets[band].read_bytes()
        assert product_uuid == PRODUCT_UUID
        assert token == "mock-provider-token"
        assert len(content) <= max_bytes
        destination.write_bytes(content)
        downloaded_bands.append(band)
        return len(content)

    monkeypatch.setattr(acquisition, "_read_json", read_json)
    monkeypatch.setattr(acquisition, "_download_file", download_file)
    temporary_directories = []
    original_temporary_directory = acquisition.TemporaryDirectory

    def track_temporary_directory(*args, **kwargs):
        temporary_directory = original_temporary_directory(*args, **kwargs)
        temporary_directories.append(temporary_directory)
        return temporary_directory

    monkeypatch.setattr(acquisition, "TemporaryDirectory", track_temporary_directory)

    response = client.post(
        "/api/satellite/catalog/acquire",
        json={
            "provider": catalog_scene["provider"],
            "collection": catalog_scene["collection"],
            "aoi": aoi_for_test_raster(),
            "scene_id": catalog_scene["id"],
        },
    )

    assert response.status_code == 201, response.text
    ingested_scene = response.json()
    assert ingested_scene["analysis"] == "imagery_ingestion"
    assert ingested_scene["metadata"]["catalog_source"]["item_id"] == SCENE_ID
    assert ingested_scene["metadata"]["width"] == 6
    assert ingested_scene["metadata"]["height"] == 3
    assert downloaded_bands == list(acquisition.BANDS)
    assert any(
        headers == {"Authorization": "Bearer mock-provider-token", "Accept": "application/json"}
        for headers in requested_headers
    )
    assert temporary_directories
    assert all(not Path(directory.name).exists() for directory in temporary_directories)

    analysis_response = client.post(
        f"/api/imagery/scenes/{ingested_scene['id']}/analyses",
        json={"analysis_type": "ndvi"},
    )
    assert analysis_response.status_code == 200, analysis_response.text
    analysis = analysis_response.json()
    assert analysis["status"] == "completed"
    assert analysis["result"]["statistics"]["valid_pixels"] == 14


def test_four_band_geotiff_conversion_preserves_grid_and_catalog_reference(tmp_path):
    if not jp2_available():
        pytest.skip("The selected Rasterio environment has no JP2OpenJPEG decoder.")
    assets = {}
    for index, band in enumerate(acquisition.BANDS, start=1):
        path = tmp_path / f"{band}.jp2"
        with rasterio.open(
            path,
            "w",
            driver="JP2OpenJPEG",
            width=8,
            height=6,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=from_origin(500_000, 1_500_000, 10, 10),
            REVERSIBLE="YES",
        ) as destination:
            data = np.full((6, 8), index * 100, dtype=np.uint16)
            if band == "B02":
                data[1, 1] = 0
            destination.write(data, 1)
        assets[band] = path
    output = tmp_path / "acquired.tif"

    acquisition._compose_geotiff(
        assets,
        output,
        item=sample_item(),
        aoi=acquisition_request(aoi_for_test_raster()).aoi,
        acquisition_datetime="2025-02-01T05:11:01Z",
        scene_id=SCENE_ID,
        product_uuid=PRODUCT_UUID,
        max_pixels=100,
    )

    with rasterio.open(output) as result:
        assert result.driver == "GTiff"
        assert result.count == 4
        assert (result.width, result.height) == (6, 3)
        assert result.descriptions == acquisition.BANDS
        assert result.read(1, masked=True).mean() == pytest.approx(100, abs=20)
        assert result.read(4, masked=True).mean() == pytest.approx(400, abs=20)
        mask = result.dataset_mask()
        assert np.all(result.read(1)[mask > 0] < result.read(2)[mask > 0])
        assert np.all(result.read(3)[mask > 0] < result.read(4)[mask > 0])
        assert result.tags()["STAC_ITEM_ID"] == SCENE_ID
        assert np.count_nonzero(mask) == 14
        assert mask[0, -1] == 0
        assert mask[1, 0] == 0


def test_four_band_conversion_rejects_misaligned_assets(tmp_path):
    if not jp2_available():
        pytest.skip("The selected Rasterio environment has no JP2OpenJPEG decoder.")
    assets = {}
    for index, band in enumerate(acquisition.BANDS, start=1):
        path = tmp_path / f"{band}.jp2"
        x_origin = 500_000 if index < 4 else 500_010
        with rasterio.open(
            path,
            "w",
            driver="JP2OpenJPEG",
            width=8,
            height=6,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=from_origin(x_origin, 1_500_000, 10, 10),
            REVERSIBLE="YES",
        ) as destination:
            destination.write(np.ones((6, 8), dtype=np.uint16), 1)
        assets[band] = path

    with pytest.raises(acquisition.AcquisitionRasterError):
        acquisition._compose_geotiff(
            assets,
            tmp_path / "misaligned.tif",
            item=sample_item(),
            aoi=acquisition_request(aoi_for_test_raster()).aoi,
            acquisition_datetime="2025-02-01T05:11:01Z",
            scene_id=SCENE_ID,
            product_uuid=PRODUCT_UUID,
            max_pixels=100,
        )
