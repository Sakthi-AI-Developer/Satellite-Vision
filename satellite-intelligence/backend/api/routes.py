import json
import hashlib
import heapq
import logging
import mimetypes
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore
from time import monotonic
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import rasterio
from rasterio._err import CPLE_BaseError
from rasterio.crs import CRS
from rasterio.errors import RasterioError
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.datastructures import Headers
from starlette.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError

from config import settings, validate_runtime_settings
from auth import AuthenticatedUser, authenticated_user_id, current_user
from persistence.artifacts import ArtifactStorageError
from persistence.manager import PersistenceManager
from models.landcover_model import LandCoverModel
from processing.change_detection import calculate_change
from processing.data_provider import LocalDataProvider
from processing.landcover import LandCoverClassifier
from processing.ndbi import calculate_ndbi
from processing.ndvi import calculate_ndvi
from processing.ndwi import calculate_ndwi
from processing.preprocessing import (
    DatasetError,
    RasterBand,
    SatelliteDataset,
    build_profile,
    identify_band,
    load_dataset,
    pixel_area_square_metres,
    raster_metadata,
)
from processing.imagery import ImageryValidationError, inspect_geotiff
from satellite import acquisition as satellite_acquisition
from processing.scene_change import run_ndvi_change_detection
from processing.satellite_analysis import (
    analysis_catalog,
    run_satellite_analysis,
    validate_band_mapping,
)
from api.schemas import SceneAIInferenceRequest, SceneAnalysisRequest, SceneChangeDetectionRequest
from processing.provenance import analysis_provenance, dataset_provenance
from processing.visualization import save_class_png, save_index_png, save_rgb_preview
from geoai import inference as geoai_inference
from geoai import (
    analyze_land_cover_transitions,
    detect_risk_indicators,
    evaluate_forecast,
    forecast_vegetation_trends,
    summarize_observation_history,
    summarize_spatial_patterns,
)
from geoai.schemas import SpatialAnalysisRequest, TransitionRequest, VegetationForecastRequest
from satellite.base_provider import (
    ProviderCapabilityUnavailableError,
    ProviderNotConfiguredError,
)
from satellite.cache import CacheManager
from satellite.catalog import CatalogError, search_sentinel2_catalog
from satellite.monitoring import MonitoringPipeline, MonitoringScheduler
from satellite.local_provider import LocalSatelliteProvider
from satellite.live_provider import LiveSatelliteProvider
from satellite.mock_provider import MockSatelliteProvider
from satellite.models import (
    SatelliteDownloadRequest,
    SatelliteSearchRequest,
    SentinelCatalogSearchRequest,
    SentinelSceneAcquisitionRequest,
)
from geoai.trend_models import _parse_date

router = APIRouter()
logger = logging.getLogger("satellite-intelligence")
persistence_manager = PersistenceManager(settings)
analysis_slots = BoundedSemaphore(settings.max_concurrent_analyses)
acquisition_slots = BoundedSemaphore(1)
monitoring_pipeline = (
    MonitoringPipeline(settings, persistence_manager.repository.engine)
    if persistence_manager.repository
    else None
)
monitoring_scheduler = (
    MonitoringScheduler(monitoring_pipeline) if monitoring_pipeline else None
)
ANALYSES = ("ndvi", "ndwi", "ndbi", "landcover", "change_detection")
ANALYSIS_ALIASES = {"change": "change_detection"}
DOWNLOAD_FILES = {
    "ndvi": "ndvi.tif",
    "ndwi": "ndwi.tif",
    "ndbi": "ndbi.tif",
    "landcover": "landcover.tif",
    "change_detection": "change_detection.tif",
    "report": "summary.json",
}
data_provider = LocalDataProvider(settings.data_dir)
EMPTY_DATA_MESSAGE = (
    "No satellite dataset found. Please add Sentinel-2 GeoTIFF bands to backend/data/current/."
)
REQUIRED_BANDS = {
    "ndvi": ("B04", "B08"),
    "ndwi": ("B03", "B08"),
    "ndbi": ("B08", "B11"),
    "landcover": ("B03", "B04", "B08", "B11"),
}
SUPPORTED_BANDS = ("B02", "B03", "B04", "B08", "B11")
DATASET_PERIODS = {"current", "historical"}
DEFAULT_RESULTS_PAGE_SIZE = 50
MAX_RESULTS_PAGE_SIZE = 100
cache_manager = CacheManager(settings.satellite_cache_dir)
_analysis_deadline: ContextVar[float | None] = ContextVar(
    "analysis_deadline",
    default=None,
)


def _check_analysis_deadline() -> None:
    deadline = _analysis_deadline.get()
    if deadline is not None and monotonic() >= deadline:
        raise HTTPException(
            status_code=504,
            detail="Analysis exceeded its configured processing time limit.",
        )


@contextmanager
def _analysis_slot():
    if not analysis_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Analysis capacity is busy. Retry the request shortly.",
            headers={"Retry-After": "1"},
        )
    deadline_token = _analysis_deadline.set(
        monotonic() + settings.max_analysis_seconds
    )
    try:
        yield
    finally:
        _analysis_deadline.reset(deadline_token)
        analysis_slots.release()


def _request_dataset_root() -> Path:
    if not settings.authentication_required:
        return settings.data_dir.resolve()
    user_id = authenticated_user_id.get()
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="A valid Supabase sign-in is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    namespace = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    root = (settings.data_dir / "users" / namespace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for period in DATASET_PERIODS:
        (root / period).mkdir(exist_ok=True)
    return root


def _request_cache_manager() -> CacheManager:
    if not settings.authentication_required:
        return cache_manager
    data_root = _request_dataset_root()
    return CacheManager(settings.satellite_cache_dir / "users" / data_root.name)


def _provider_for_mode(cache: CacheManager | None = None):
    provider_name = (settings.satellite_provider or "local").lower()
    if provider_name == "mock":
        return MockSatelliteProvider(cache or _request_cache_manager())
    if provider_name == "live":
        return LiveSatelliteProvider()
    return LocalSatelliteProvider()


def _dataset(
    directory: Path,
    required: tuple[str, ...] = (),
    *,
    include_other_bands: bool = True,
) -> SatelliteDataset:
    try:
        relative_path = directory.resolve().relative_to(settings.data_dir.resolve())
    except ValueError as exc:
        raise DatasetError("Requested dataset directory is outside the configured data root.") from exc
    if settings.authentication_required:
        data_root = _request_dataset_root()
        scoped_directory = (data_root / relative_path).resolve()
        if data_root not in scoped_directory.parents:
            raise DatasetError("Requested dataset directory is outside the authenticated user's data root.")
        return load_dataset(
            scoped_directory,
            required,
            include_other_bands=include_other_bands,
            max_pixels=settings.max_raster_pixels,
            max_file_bytes=settings.max_raster_bytes,
            max_input_array_bytes=settings.max_input_array_bytes,
            deadline_check=_check_analysis_deadline,
        )
    return data_provider.load(
        relative_path.as_posix(),
        required,
        include_other_bands=include_other_bands,
        max_pixels=settings.max_raster_pixels,
        max_file_bytes=settings.max_raster_bytes,
        max_input_array_bytes=settings.max_input_array_bytes,
        deadline_check=_check_analysis_deadline,
    )


def _public_dataset(dataset: SatelliteDataset, dataset_id: str) -> dict:
    reference = next(iter(dataset.bands.values()), None)
    band_codes = tuple(code for code in SUPPORTED_BANDS if code in dataset.bands)
    return {
        "available": bool(dataset.bands),
        "bands": [
            {
                "code": code,
                "name": band_name,
                "width": dataset.bands[code].width,
                "height": dataset.bands[code].height,
                "crs": dataset.bands[code].crs,
                "resolution": dataset.bands[code].resolution,
            }
            for code, band_name in (
                ("B02", "Blue"),
                ("B03", "Green"),
                ("B04", "Red"),
                ("B08", "Near infrared"),
                ("B11", "Short-wave infrared"),
            )
            if code in dataset.bands
        ],
        "missing_bands": [band for band in SUPPORTED_BANDS if band not in dataset.bands],
        "errors": dataset.errors,
        "bounds": _bounds(reference) if reference else None,
        "metadata": raster_metadata(reference),
        "provenance": dataset_provenance(dataset_id, dataset, band_codes),
    }


def _data_quality_summary(dataset: SatelliteDataset) -> dict:
    if not dataset.bands:
        return {
            "status": "MISSING",
            "message": EMPTY_DATA_MESSAGE,
            "missing_bands": list(SUPPORTED_BANDS),
            "nodata_percentage": 100.0,
            "warnings": ["No current dataset was found. Add Sentinel-2 GeoTIFFs under backend/data/current/."],
        }
    reference = next(iter(dataset.bands.values()))
    valid_pixels = 0
    total_band_pixels = 0
    invalid_band_pixels = 0
    for band in dataset.bands.values():
        valid_pixels += int(np.count_nonzero(band.valid))
        total_band_pixels += int(band.valid.size)
        invalid_band_pixels += int(np.count_nonzero(~band.valid))
    nodata_percentage = (
        invalid_band_pixels / total_band_pixels * 100
        if total_band_pixels
        else 100.0
    )
    warnings = []
    if dataset.missing_bands:
        warnings.append(f"Missing required bands: {', '.join(dataset.missing_bands)}")
    if nodata_percentage > 20:
        warnings.append("A large share of pixels is nodata or invalid.")
    if reference.crs is None:
        warnings.append("Raster CRS is missing; geospatial interpretation is limited.")
    try:
        from processing.preprocessing import validate_alignment

        validate_alignment(dataset.bands.values())
    except DatasetError as exc:
        warnings.append(f"Available bands are not on a verified common grid: {exc}")
    try:
        historical_available = bool(_dataset(settings.data_dir / "historical").bands)
    except DatasetError:
        historical_available = False
    return {
        "status": "GOOD" if not warnings else "WARNING",
        "missing_bands": dataset.missing_bands,
        "nodata_percentage": float(nodata_percentage),
        "crs": reference.crs,
        "width": reference.width,
        "height": reference.height,
        "resolution": list(reference.resolution),
        "valid_band_pixels": valid_pixels,
        "total_band_pixels": total_band_pixels,
        "nodata_percentage_method": "invalid band-pixels divided by all available band-pixels",
        "historical_available": historical_available,
        "warnings": warnings,
    }


def _bounds(reference) -> list[float] | None:
    if not reference or not reference.crs:
        return None
    try:
        bounds = array_bounds(reference.height, reference.width, reference.transform)
        if reference.crs != "EPSG:4326":
            crs = CRS.from_string(reference.crs)
            if not (crs.is_geographic or crs.is_projected):
                return None
            bounds = transform_bounds(
                reference.crs, "EPSG:4326", *bounds, densify_pts=21
            )
    except (CPLE_BaseError, RasterioError, ValueError):
        return None
    if not np.all(np.isfinite(bounds)):
        return None
    return [float(value) for value in bounds]


def _dated_ndvi_observations(
    dataset_ids: tuple[str, ...] = ("current", "historical"),
) -> tuple[list[dict[str, Any]], list[str]]:
    observations: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_dates: set[str] = set()
    for dataset_id in dataset_ids:
        try:
            dataset = _check_dataset(settings.data_dir / dataset_id, "ndvi")
        except DatasetError as exc:
            warnings.append(f"{dataset_id}: {exc}")
            continue
        provenance = dataset_provenance(
            dataset_id, dataset, REQUIRED_BANDS["ndvi"]
        )
        if provenance["data_classification"] == "synthetic":
            warnings.append(
                f"{dataset_id}: synthetic fixture; it is not a satellite observation."
            )
        else:
            warnings.append(
                f"{dataset_id}: acquisition metadata is user-provided and has not been independently verified."
            )
        parsed_date = _parse_date(provenance.get("acquisition_date"))
        if parsed_date is None:
            warnings.append(
                f"{dataset_id}: no valid acquisition date in raster metadata; excluded from temporal analysis."
            )
            continue
        observation_date = parsed_date.date().isoformat()
        if observation_date in seen_dates:
            warnings.append(
                f"{dataset_id}: duplicate acquisition date {observation_date}; excluded from temporal analysis."
            )
            continue
        ndvi = calculate_ndvi(dataset.bands["B04"], dataset.bands["B08"])
        valid_values = ndvi["raster"][ndvi["valid_mask"]]
        if valid_values.size == 0:
            warnings.append(f"{dataset_id}: no valid NDVI pixels; excluded from temporal analysis.")
            continue
        seen_dates.add(observation_date)
        observations.append(
            {
                "date": observation_date,
                "mean_ndvi": float(np.mean(valid_values)),
                "dataset_id": dataset_id,
                "provenance": provenance,
            }
        )
    observations.sort(key=lambda item: item["date"])
    return observations, warnings


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if key not in {"raster", "valid_mask", "reference"}}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save_raster(path: Path, data: np.ndarray, valid: np.ndarray, reference, classes: bool = False) -> None:
    profile = build_profile(reference, dtype="uint8" if classes else "float32", nodata=0 if classes else -9999)
    output = np.where(valid, data, 0 if classes else -9999)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output.astype(profile["dtype"]), 1)
        dst.update_tags(AREA_OR_POINT="Area")


def _save_analysis(
    result_dir: Path,
    analysis_key: str,
    data: dict,
    title: str,
    visual_kind: str | None = None,
    classes: bool = False,
) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    raster_name = DOWNLOAD_FILES[analysis_key]
    _save_raster(result_dir / raster_name, data["raster"], data["valid_mask"], data["reference"], classes)
    image_name = f"{analysis_key}.png"
    if classes:
        save_class_png(data["raster"], data["valid_mask"], result_dir / image_name)
    else:
        save_index_png(data["raster"], data["valid_mask"], result_dir / image_name, visual_kind or analysis_key)
    summary = _json_safe(data)
    summary.update(
        {
            "analysis": title,
            "bounds": _bounds(data["reference"]),
            "visualization_url": f"/api/results/{result_dir.name}/image/{analysis_key}",
            "download_url": f"/api/results/{result_dir.name}/download/{analysis_key}",
        }
    )
    return summary


def _single_result(result_dir: Path, key: str, dataset: SatelliteDataset) -> dict:
    bands = dataset.bands
    if key == "ndvi":
        data = calculate_ndvi(bands["B04"], bands["B08"])
        pixel_area = pixel_area_square_metres(bands["B04"])
        data["vegetation_area_square_metres"] = (
            data["statistics"]["valid_pixels"] * data["vegetation_percentage"] / 100 * pixel_area
            if pixel_area is not None and data["vegetation_percentage"] is not None
            else None
        )
        result = _save_analysis(result_dir, key, data, "NDVI", "ndvi")
    elif key == "ndwi":
        result = _save_analysis(
            result_dir,
            key,
            calculate_ndwi(bands["B03"], bands["B08"]),
            "NDWI",
            "ndwi",
        )
    elif key == "ndbi":
        result = _save_analysis(
            result_dir,
            key,
            calculate_ndbi(bands["B08"], bands["B11"]),
            "NDBI",
            "ndbi",
        )
    elif key == "landcover":
        result = _save_analysis(
            result_dir,
            key,
            LandCoverClassifier().classify(bands),
            "Land-use / land-cover",
            classes=True,
        )
    else:
        raise DatasetError(f"Unsupported analysis: {key}.")
    result["provenance"] = analysis_provenance(
        key, {"current": (dataset, REQUIRED_BANDS[key])}
    )
    return result


def _validate_dataset(dataset: SatelliteDataset, key: str) -> SatelliteDataset:
    dataset.require(REQUIRED_BANDS[key])
    from processing.preprocessing import validate_alignment
    validate_alignment([dataset.bands[code] for code in REQUIRED_BANDS[key]])
    return dataset


def _validate_upload_alignment(upload_path: Path, dataset_dir: Path) -> None:
    try:
        with rasterio.open(upload_path) as uploaded:
            if uploaded.count != 1:
                raise DatasetError("Each uploaded GeoTIFF must contain exactly one band.")
            if uploaded.crs is None:
                raise DatasetError("The uploaded GeoTIFF has no CRS; add georeferenced imagery.")
            if uploaded.width < 1 or uploaded.height < 1:
                raise DatasetError("The uploaded GeoTIFF has no raster pixels.")
            for existing_path in sorted(dataset_dir.iterdir()):
                if not existing_path.is_file() or existing_path.suffix.lower() not in {".tif", ".tiff"}:
                    continue
                if identify_band(existing_path.name) is None:
                    continue
                if existing_path.stat().st_size > settings.max_raster_bytes:
                    raise DatasetError(
                        f"{existing_path.name} exceeds the configured raster file limit."
                    )
                try:
                    with rasterio.open(existing_path) as existing:
                        if existing.width * existing.height > settings.max_raster_pixels:
                            raise DatasetError(
                                f"{existing_path.name} exceeds the configured raster pixel limit."
                            )
                        if (uploaded.width, uploaded.height) != (existing.width, existing.height):
                            raise DatasetError(
                                f"The uploaded raster dimensions differ from {existing_path.name}; "
                                "align the bands before uploading."
                            )
                        if uploaded.crs != existing.crs:
                            raise DatasetError(
                                f"The uploaded raster CRS differs from {existing_path.name}; "
                                "reproject the bands before uploading."
                            )
                        if not np.allclose(
                            tuple(uploaded.transform),
                            tuple(existing.transform),
                            rtol=0,
                            atol=1e-9,
                        ):
                            raise DatasetError(
                                f"The uploaded raster geotransform differs from {existing_path.name}; "
                                "align the bands before uploading."
                            )
                except DatasetError:
                    raise
                except RasterioError:
                    continue
    except DatasetError:
        raise
    except (RasterioError, OSError) as exc:
        raise DatasetError("The uploaded file is not a readable GeoTIFF.") from exc


def _safe_imagery_filename(filename: str | None) -> str:
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip(" .")
    return name[:255]


def _local_imagery_scenes(limit: int) -> list[dict[str, Any]]:
    scenes = []
    if not settings.output_dir.is_dir():
        return scenes
    for directory in settings.output_dir.iterdir():
        if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
            continue
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("analysis") == "imagery_ingestion" and isinstance(
            summary.get("metadata"), dict
        ):
            scenes.append(summary)
    return sorted(
        scenes,
        key=lambda scene: scene.get("created_at") or "",
        reverse=True,
    )[:limit]


@router.post("/imagery/ingest", status_code=201)
async def ingest_imagery(
    file: UploadFile = File(...),
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    original_filename = _safe_imagery_filename(file.filename)
    if Path(original_filename).suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a GeoTIFF file with a .tif or .tiff extension.",
        )
    media_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if media_type and media_type not in {
        "image/tiff",
        "image/geotiff",
        "application/octet-stream",
    }:
        raise HTTPException(status_code=415, detail="The upload must use a GeoTIFF media type.")

    byte_limit = min(
        settings.max_upload_bytes,
        settings.max_raster_bytes,
        settings.max_artifact_bytes,
    )
    total_bytes = 0
    digest = hashlib.sha256()
    try:
        with tempfile.TemporaryDirectory(prefix="satellite-imagery-") as staging:
            source_path = Path(staging) / "source.tif"
            preview_path = Path(staging) / "preview.png"
            with source_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > byte_limit:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds the configured {byte_limit // (1024 * 1024)} MB imagery limit.",
                        )
                    digest.update(chunk)
                    destination.write(chunk)
            if total_bytes == 0:
                raise HTTPException(status_code=422, detail="The uploaded GeoTIFF is empty.")

            try:
                metadata = await run_in_threadpool(
                    inspect_geotiff,
                    source_path,
                    preview_path,
                    original_filename=original_filename,
                    sha256=digest.hexdigest(),
                    max_pixels=settings.max_imagery_pixels,
                )
            except ImageryValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            input_parameters = {"format": "GeoTIFF", "sha256": digest.hexdigest()}
            try:
                job_start = persistence_manager.start_job(
                    uuid4().hex,
                    "imagery_ingestion",
                    input_parameters,
                    owner_id=user.id if user else None,
                    idempotency_key=digest.hexdigest(),
                )
            except Exception as exc:
                logger.error(
                    "Imagery ingestion could not register its processing job.",
                    extra={
                        "event": "imagery.job_registration_failed",
                        "error_type": type(exc).__name__,
                    },
                )
                raise HTTPException(
                    status_code=503,
                    detail="Imagery metadata storage is temporarily unavailable.",
                ) from exc

            retrying = False
            if not job_start.created:
                existing_record = persistence_manager.get_analysis(
                    job_start.analysis_id,
                    user.id if user else None,
                )
                existing_job = persistence_manager.get_job(
                    job_start.job_id,
                    user.id if user else None,
                )
                if (
                    existing_record is not None
                    and existing_job is not None
                    and existing_record["status"] == "failed"
                    and existing_job["status"] == "failed"
                ):
                    try:
                        retrying = persistence_manager.retry_failed_imagery_job(
                            job_start.analysis_id,
                            job_start.job_id,
                            input_parameters,
                            user.id if user else None,
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed imagery ingestion could not be resumed.",
                            extra={
                                "event": "imagery.retry_registration_failed",
                                "job_id": job_start.job_id,
                                "error_type": type(exc).__name__,
                            },
                        )
                        raise HTTPException(
                            status_code=503,
                            detail="Imagery metadata storage is temporarily unavailable.",
                        ) from exc
                if not retrying:
                    existing = _existing_analysis_response(
                        job_start.analysis_id,
                        job_start.job_id,
                        "imagery_ingestion",
                        input_parameters,
                        user.id if user else None,
                    )
                    return {**existing, "duplicate": True}

            result_id = job_start.analysis_id
            result_dir = settings.output_dir / result_id
            try:
                settings.output_dir.mkdir(parents=True, exist_ok=True)
                if result_dir.is_symlink():
                    raise ValueError("Imagery output directory is not a safe directory.")
                if result_dir.exists():
                    if not result_dir.is_dir() or result_dir.resolve().parent != settings.output_dir.resolve():
                        raise ValueError("Imagery output directory is not a safe directory.")
                    allowed_files = {"source.tif", "preview.png", "summary.json"}
                    for existing_path in result_dir.iterdir():
                        if existing_path.is_symlink() or existing_path.name not in allowed_files:
                            raise ValueError("Imagery output directory contains unexpected files.")
                else:
                    result_dir.mkdir(exist_ok=False)
                shutil.copyfile(source_path, result_dir / "source.tif")
                shutil.copyfile(preview_path, result_dir / "preview.png")
                previous_summary = None
                if retrying:
                    try:
                        candidate = json.loads(
                            (result_dir / "summary.json").read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        candidate = None
                    if (
                        isinstance(candidate, dict)
                        and candidate.get("analysis") == "imagery_ingestion"
                        and candidate.get("id") == result_id
                        and isinstance(candidate.get("metadata"), dict)
                        and candidate["metadata"].get("sha256")
                        == digest.hexdigest()
                    ):
                        previous_summary = candidate
                        metadata = candidate["metadata"]
                persistence_mode = (
                    "cloud-private"
                    if persistence_manager.cloud_persistence_active
                    else "local-only"
                )
                result = {
                    "success": True,
                    "id": result_id,
                    "job_id": job_start.job_id,
                    "analysis": "imagery_ingestion",
                    "status": "completed",
                    "created_at": (
                        previous_summary.get("created_at")
                        if previous_summary
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    "metadata": metadata,
                    "preview_url": (
                        f"/api/results/{result_id}/artifacts/preview.png/download"
                    ),
                    "source_download_url": (
                        f"/api/results/{result_id}/artifacts/source.tif/download"
                    ),
                    "persistence": {
                        "mode": persistence_mode,
                        "bucket": (
                            settings.supabase_storage_bucket
                            if persistence_mode == "cloud-private"
                            else None
                        ),
                    },
                }
                _store_result(result_id, result_dir, result, job_start.job_id)
                return result
            except Exception as exc:
                try:
                    persistence_manager.fail_job(
                        job_start.job_id,
                        "Imagery artifacts could not be persisted.",
                    )
                except Exception as state_error:
                    logger.error(
                        "Imagery ingestion failure state could not be saved.",
                        extra={
                            "event": "imagery.job_state_update_failed",
                            "job_id": job_start.job_id,
                            "error_type": type(state_error).__name__,
                        },
                    )
                logger.error(
                    "Imagery ingestion did not complete.",
                    extra={
                        "event": "imagery.ingestion_failed",
                        "job_id": job_start.job_id,
                        "error_type": type(exc).__name__,
                    },
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Imagery artifacts could not be fully persisted. "
                        "Recoverable local processing files were retained."
                    ),
                    headers={"X-Processing-Job-ID": job_start.job_id},
                ) from exc
    except HTTPException:
        raise
    except OSError as exc:
        logger.error(
            "Imagery upload staging failed.",
            extra={
                "event": "imagery.upload_staging_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=503,
            detail="The imagery upload could not be staged; try again later.",
        ) from exc


def _existing_acquired_scene(scene_id: str, owner_id: str) -> dict[str, Any] | None:
    scenes = persistence_manager.list_analyses_by_type(
        "imagery_ingestion",
        owner_id=owner_id,
        limit=100,
    )
    for scene in scenes or []:
        summary = scene.get("summary")
        metadata = summary.get("metadata") if isinstance(summary, dict) else None
        catalog_source = metadata.get("catalog_source") if isinstance(metadata, dict) else None
        if isinstance(catalog_source, dict) and catalog_source.get("item_id") == scene_id:
            return summary
    return None


@router.post("/satellite/catalog/acquire", status_code=201)
async def acquire_catalog_scene(
    request: SentinelSceneAcquisitionRequest,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Sign in to acquire and save a Copernicus scene.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    acquisition_area_km2 = request.aoi.estimate_area_km2()
    if acquisition_area_km2 > settings.max_satellite_acquisition_area_km2:
        raise HTTPException(
            status_code=422,
            detail=(
                "Acquisition AOI exceeds the configured "
                f"{settings.max_satellite_acquisition_area_km2:g} km² limit."
            ),
        )
    existing_scene = _existing_acquired_scene(request.scene_id, user.id)
    if existing_scene is not None:
        return {
            **existing_scene,
            "catalog_acquisition": existing_scene.get("metadata", {}).get("catalog_source"),
            "duplicate": True,
        }
    if not acquisition_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Satellite acquisition capacity is busy. Retry shortly.",
            headers={"Retry-After": "1"},
        )
    byte_limit = min(
        settings.max_upload_bytes,
        settings.max_raster_bytes,
        settings.max_artifact_bytes,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="satellite-acquisition-") as staging:
            geotiff_path = Path(staging) / f"{request.scene_id}.tif"
            try:
                await run_in_threadpool(
                    satellite_acquisition.acquire_scene,
                    request,
                    geotiff_path,
                    max_bytes=byte_limit,
                    max_pixels=settings.max_imagery_pixels,
                    aoi_max_area_km2=settings.max_satellite_acquisition_area_km2,
                )
            except satellite_acquisition.AcquisitionError as exc:
                logger.warning(
                    "Copernicus scene acquisition did not complete.",
                    extra={
                        "event": "satellite.acquisition_failed",
                        "scene_id": request.scene_id,
                        "status_code": exc.status_code,
                        "error_type": type(exc).__name__,
                    },
                )
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

            with geotiff_path.open("rb") as source_file:
                upload = UploadFile(
                    file=source_file,
                    filename=geotiff_path.name,
                    headers=Headers({"content-type": "image/tiff"}),
                )
                try:
                    scene = await ingest_imagery(upload, user)
                finally:
                    await upload.close()
            return {
                **scene,
                "catalog_acquisition": scene.get("metadata", {}).get("catalog_source"),
            }
    finally:
        acquisition_slots.release()


@router.get("/imagery/scenes")
def list_imagery_scenes(
    limit: int = Query(20, ge=1, le=50),
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    scenes = persistence_manager.list_analyses_by_type(
        "imagery_ingestion",
        owner_id=user.id if user else None,
        limit=limit,
    )
    if scenes is None:
        if user is not None or settings.authentication_required:
            summaries = []
        else:
            summaries = _local_imagery_scenes(limit)
    else:
        summaries = [
            scene["summary"]
            for scene in scenes
            if isinstance(scene.get("summary"), dict)
        ]
    return {"success": True, "scenes": summaries}


def _owned_imagery_scene(
    scene_id: str,
    owner_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    scene_id = _validated_id(scene_id)
    record = persistence_manager.get_analysis(scene_id, owner_id=owner_id)
    if record is not None:
        if record["analysis"] != "imagery_ingestion" or not isinstance(
            record["summary"], dict
        ):
            raise HTTPException(status_code=404, detail="Imagery scene was not found.")
        return record["summary"], record
    if owner_id is not None or settings.authentication_required:
        raise HTTPException(status_code=404, detail="Imagery scene was not found.")
    try:
        summary_path = _result_directory(scene_id) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (HTTPException, OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Imagery scene was not found.")
    if summary.get("analysis") != "imagery_ingestion":
        raise HTTPException(status_code=404, detail="Imagery scene was not found.")
    return summary, None


@router.get("/imagery/scenes/{scene_id}")
def get_imagery_scene(
    scene_id: str,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    summary, _ = _owned_imagery_scene(scene_id, user.id if user else None)
    return summary


def _copy_owned_scene_source(
    scene_id: str,
    owner_id: str | None,
    scene_summary: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    try:
        stored = persistence_manager.get_artifact(
            scene_id,
            "source.tif",
            owner_id=owner_id,
        )
    except (ArtifactStorageError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The imagery scene is temporarily unavailable from private storage.",
        ) from exc
    if stored is None:
        raise HTTPException(status_code=404, detail="The imagery source artifact was not found.")
    source_stream, artifact = stored
    byte_limit = min(settings.max_raster_bytes, settings.max_artifact_bytes)
    if int(artifact.get("size_bytes", 0)) > byte_limit:
        source_stream.close()
        raise HTTPException(
            status_code=413,
            detail="The imagery source exceeds the configured processing file-size limit.",
        )

    expected_digest = scene_summary.get("metadata", {}).get("sha256")
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with destination.open("wb") as output:
            while chunk := source_stream.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > byte_limit:
                    raise HTTPException(
                        status_code=413,
                        detail="The imagery source exceeds the configured processing file-size limit.",
                    )
                digest.update(chunk)
                output.write(chunk)
    except HTTPException:
        raise
    except (ArtifactStorageError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The imagery scene could not be read from private storage.",
        ) from exc
    finally:
        source_stream.close()

    if total_bytes <= 0 or total_bytes != int(artifact.get("size_bytes", total_bytes)):
        raise HTTPException(
            status_code=503,
            detail="The imagery source artifact is incomplete in private storage.",
        )
    if expected_digest and digest.hexdigest() != expected_digest:
        raise HTTPException(
            status_code=503,
            detail="The imagery source failed its stored integrity check.",
        )
    return artifact


def _scene_analysis_artifacts(result_id: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    result_metadata = result["result"]
    for artifact_name in (
        result_metadata["preview"]["artifact_name"],
        result_metadata["raster"]["artifact_name"],
        "summary.json",
    ):
        output.append(
            {
                "artifact_name": artifact_name,
                "artifact_type": (
                    "image" if artifact_name == "preview.png"
                    else "report" if artifact_name == "summary.json"
                    else "raster"
                ),
                "download_url": (
                    f"/api/results/{result_id}/artifacts/"
                    f"{quote(artifact_name, safe='')}/download"
                ),
            }
        )
    return output


@router.get("/imagery/ai-models")
def list_ai_models() -> dict[str, Any]:
    return geoai_inference.list_models(settings)


def _fail_ai_inference_job(job_id: str, detail: str) -> None:
    try:
        persistence_manager.fail_job(job_id, detail)
    except Exception as state_error:
        logger.error(
            "AI inference failure state could not be persisted.",
            extra={
                "event": "ai_inference.job_state_update_failed",
                "job_id": job_id,
                "error_type": type(state_error).__name__,
            },
        )


@router.post("/imagery/scenes/{scene_id}/ai-inference", status_code=201)
def run_scene_ai_inference(
    scene_id: str,
    request: SceneAIInferenceRequest,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict[str, Any]:
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Sign in to run and save AI inference results.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    registry = geoai_inference.list_models(settings)
    model = next(
        (item for item in registry["models"] if item["id"] == request.model_id),
        None,
    )
    if model is None:
        raise HTTPException(status_code=404, detail="The requested AI model was not found.")
    if model["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=model["readiness_detail"] or geoai_inference.ModelNotReadyError.detail,
        )

    scene_summary, _ = _owned_imagery_scene(scene_id, user.id)
    metadata = scene_summary.get("metadata")
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=422, detail="The imagery scene metadata is incomplete.")
    input_parameters = {
        "scene_id": scene_summary["id"],
        "scene_sha256": metadata.get("sha256"),
        "model_id": request.model_id,
        "model_version": model["version"],
        "model_weights_sha256": geoai_inference.model_weights_sha256(settings),
    }
    idempotency_key = hashlib.sha256(
        json.dumps(input_parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with _analysis_slot():
        try:
            with tempfile.TemporaryDirectory(prefix="satellite-ai-inference-") as staging:
                staging_path = Path(staging)
                source_path = staging_path / "source.tif"
                source_artifact = _copy_owned_scene_source(
                    scene_summary["id"],
                    user.id,
                    scene_summary,
                    source_path,
                )
                try:
                    job_start = persistence_manager.start_job(
                        uuid4().hex,
                        "ai_landcover",
                        input_parameters,
                        owner_id=user.id,
                        idempotency_key=idempotency_key,
                    )
                except Exception as exc:
                    logger.error(
                        "AI inference could not register its processing job.",
                        extra={
                            "event": "ai_inference.job_registration_failed",
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="AI inference metadata storage is temporarily unavailable.",
                    ) from exc

                result_id, job_id = job_start.analysis_id, job_start.job_id
                if not job_start.created:
                    existing_record = persistence_manager.get_analysis(result_id, user.id)
                    if (
                        existing_record is None
                        or existing_record["analysis"] != "ai_landcover"
                        or existing_record["input_parameters"] != input_parameters
                    ):
                        raise HTTPException(status_code=404, detail="AI inference result was not found.")
                    if isinstance(existing_record.get("summary"), dict):
                        return {**existing_record["summary"], "duplicate": True}
                    return {
                        "success": existing_record["status"] != "failed",
                        "id": result_id,
                        "job_id": job_id,
                        "analysis": "ai_landcover",
                        "status": existing_record["status"],
                        "error": existing_record.get("error"),
                        "artifacts": [],
                        "duplicate": True,
                    }

                try:
                    classification_path = staging_path / "classification.tif"
                    confidence_path = staging_path / "confidence.tif"
                    inference_result = geoai_inference.run_landcover_inference(
                        source_path,
                        classification_path,
                        confidence_path,
                        configuration=settings,
                        deadline_check=_check_analysis_deadline,
                    )
                    _check_analysis_deadline()
                    output_directory = settings.output_dir.resolve()
                    output_directory.mkdir(parents=True, exist_ok=True)
                    result_directory = settings.output_dir / result_id
                    if result_directory.exists() or result_directory.is_symlink():
                        raise ValueError("AI inference output directory is not available.")
                    result_directory.mkdir(exist_ok=False)
                    shutil.copyfile(classification_path, result_directory / "classification.tif")
                    shutil.copyfile(confidence_path, result_directory / "confidence.tif")
                    result = {
                        "success": True,
                        "id": result_id,
                        "job_id": job_id,
                        "analysis": "ai_landcover",
                        "status": "completed",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "scene": {
                            "id": scene_summary["id"],
                            "filename": metadata.get("filename"),
                            "acquisition_date": metadata.get("acquisition_date"),
                        },
                        "source_artifact": {
                            "size_bytes": source_artifact["size_bytes"],
                            "sha256": metadata.get("sha256"),
                        },
                        "result": inference_result,
                        "artifacts": [
                            {
                                "artifact_name": name,
                                "artifact_type": "raster",
                                "download_url": (
                                    f"/api/results/{result_id}/artifacts/"
                                    f"{quote(name, safe='')}/download"
                                ),
                            }
                            for name in ("classification.tif", "confidence.tif")
                        ],
                    }
                    _store_result(result_id, result_directory, result, job_id)
                    return result
                except geoai_inference.InferenceError as exc:
                    _fail_ai_inference_job(job_id, exc.detail)
                    raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
                except Exception as exc:
                    _fail_ai_inference_job(
                        job_id,
                        "AI inference or result persistence did not complete.",
                    )
                    if isinstance(exc, HTTPException):
                        raise
                    logger.error(
                        "AI inference job failed.",
                        extra={
                            "event": "ai_inference.job_failed",
                            "job_id": job_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="AI inference or result persistence did not complete.",
                        headers={"X-Processing-Job-ID": job_id},
                    ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "AI inference input could not be staged.",
                extra={
                    "event": "ai_inference.staging_failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="The imagery scene could not be staged for AI inference.",
            ) from exc


@router.get("/imagery/ai-inference/{analysis_id}")
def get_ai_inference_result(
    analysis_id: str,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict[str, Any]:
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Sign in to retrieve AI inference results.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    analysis_id = _validated_id(analysis_id)
    record = persistence_manager.get_analysis(analysis_id, owner_id=user.id)
    if record is None or record.get("analysis") != "ai_landcover":
        raise HTTPException(status_code=404, detail="AI inference result was not found.")
    if isinstance(record.get("summary"), dict):
        return record["summary"]
    return {
        "success": record["status"] != "failed",
        "id": record["id"],
        "job_id": record.get("job_id"),
        "analysis": "ai_landcover",
        "status": record["status"],
        "error": record.get("error"),
        "artifacts": [],
    }


def _fail_scene_analysis_job(job_id: str, detail: str) -> None:
    try:
        persistence_manager.fail_job(job_id, detail)
    except Exception as state_error:
        logger.error(
            "Failed to persist satellite scene analysis failure state.",
            extra={
                "event": "scene_analysis.job_state_update_failed",
                "job_id": job_id,
                "error_type": type(state_error).__name__,
            },
        )


@router.get("/imagery/analysis-types")
def list_imagery_analysis_types(
    scene_id: str | None = Query(default=None, min_length=32, max_length=32),
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    scene_summary = None
    scene_bands = []
    if scene_id is not None:
        scene_summary, _ = _owned_imagery_scene(
            scene_id,
            user.id if user else None,
        )
        metadata = scene_summary.get("metadata")
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=422, detail="The imagery scene metadata is incomplete.")
        scene_bands = [
            band
            for band in metadata.get("bands", [])
            if isinstance(band, dict)
        ]
    catalog = analysis_catalog(
        scene_summary.get("metadata") if scene_summary else None
    )
    if scene_summary is not None:
        for item in catalog:
            if item["id"] in {"ndvi", "ndwi"} and item["available"]:
                item["availability_note"] = (
                    "Band labels were read from the GeoTIFF and have not been independently verified."
                )
            elif item["id"] in {"ndvi", "ndwi"}:
                item["availability_note"] = (
                    "Choose the band numbers matching the documented roles before submitting."
                )
            elif item["id"] == "kmeans" and item["available"]:
                item["availability_note"] = "Select 2–16 feature bands and 2–10 clusters."
        return {
            "success": True,
            "scene_id": scene_summary["id"],
            "band_count": scene_summary["metadata"].get("band_count"),
            "bands": scene_bands,
            "analysis_types": catalog,
        }
    return {"success": True, "analysis_types": catalog}


def _scene_date(metadata: dict[str, Any], role: str) -> date:
    value = metadata.get("acquisition_date")
    try:
        if not isinstance(value, str):
            raise ValueError
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"The {role} scene has no valid acquisition date in its source metadata.",
        ) from exc
    return parsed


def _sensor_identity(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    if key == "sensor":
        if "msi" in normalized or "sentinel2" in normalized:
            return "sentinel2msi"
        if "oli" in normalized or "tirs" in normalized:
            return "landsatoli"
    if "sentinel2" in normalized or re.search(r"s2[ab]", normalized):
        return "sentinel2"
    if "landsat" in normalized or re.search(r"lc0[89]", normalized):
        return "landsat"
    return normalized


def _validate_scene_sensor_compatibility(
    baseline_metadata: dict[str, Any],
    comparison_metadata: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    for identity_key in ("sensor", "platform"):
        baseline_identity = _sensor_identity(baseline_metadata, identity_key)
        comparison_identity = _sensor_identity(comparison_metadata, identity_key)
        if baseline_identity and comparison_identity:
            if baseline_identity != comparison_identity:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "The scenes report incompatible "
                        f"{identity_key} identities; compare scenes from compatible sensors."
                    ),
                )
        else:
            warnings.append(
                f"{identity_key.title()} identity is missing or unverified for at least one scene."
            )
    return warnings


def _validate_scene_band_identity(
    metadata: dict[str, Any],
    mapping: dict[str, int],
    role: str,
) -> None:
    expected_codes = {"red": "B04", "nir": "B08"}
    code_by_index = {
        band.get("index"): band.get("code")
        for band in metadata.get("bands", [])
        if isinstance(band, dict) and isinstance(band.get("index"), int)
    }
    for spectral_role, index in mapping.items():
        observed_code = code_by_index.get(index)
        if observed_code in {"B02", "B03", "B04", "B08", "B11"} and observed_code != expected_codes[spectral_role]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The selected {role} band {index} is labelled {observed_code}, "
                    f"not the required {spectral_role} band {expected_codes[spectral_role]}."
                ),
            )


def _scene_change_artifacts(result_id: str) -> list[dict[str, str]]:
    return [
        {
            "artifact_name": artifact_name,
            "artifact_type": artifact_type,
            "download_url": (
                f"/api/results/{result_id}/artifacts/"
                f"{quote(artifact_name, safe='')}/download"
            ),
        }
        for artifact_name, artifact_type in (
            ("preview.png", "image"),
            ("ndvi-difference.tif", "raster"),
            ("change-mask.tif", "raster"),
            ("summary.json", "report"),
        )
    ]


@router.post("/imagery/change-detection")
def compare_imagery_scenes(
    request: SceneChangeDetectionRequest,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    owner_id = user.id if user else None
    baseline_summary, _ = _owned_imagery_scene(
        request.baseline_scene_id,
        owner_id,
    )
    comparison_summary, _ = _owned_imagery_scene(
        request.comparison_scene_id,
        owner_id,
    )
    baseline_metadata = baseline_summary.get("metadata")
    comparison_metadata = comparison_summary.get("metadata")
    if not isinstance(baseline_metadata, dict) or not isinstance(comparison_metadata, dict):
        raise HTTPException(status_code=422, detail="A scene has incomplete imagery metadata.")

    baseline_date = _scene_date(baseline_metadata, "baseline")
    comparison_date = _scene_date(comparison_metadata, "comparison")
    if comparison_date <= baseline_date:
        raise HTTPException(
            status_code=422,
            detail="The comparison scene acquisition date must be later than the baseline date.",
        )
    compatibility_warnings = _validate_scene_sensor_compatibility(
        baseline_metadata,
        comparison_metadata,
    )
    input_parameters = {
        "method": "ndvi_difference",
        "baseline_scene_id": baseline_summary["id"],
        "baseline_sha256": baseline_metadata.get("sha256"),
        "baseline_acquisition_date": baseline_date.isoformat(),
        "baseline_band_mapping": request.baseline_band_mapping,
        "comparison_scene_id": comparison_summary["id"],
        "comparison_sha256": comparison_metadata.get("sha256"),
        "comparison_acquisition_date": comparison_date.isoformat(),
        "comparison_band_mapping": request.comparison_band_mapping,
        "threshold": request.threshold,
        "threshold_units": "absolute NDVI difference",
    }
    idempotency_key = hashlib.sha256(
        json.dumps(input_parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with _analysis_slot():
        try:
            with tempfile.TemporaryDirectory(prefix="satellite-change-") as staging:
                staging_directory = Path(staging)
                baseline_path = staging_directory / "baseline.tif"
                comparison_path = staging_directory / "comparison.tif"
                baseline_artifact = _copy_owned_scene_source(
                    baseline_summary["id"],
                    owner_id,
                    baseline_summary,
                    baseline_path,
                )
                comparison_artifact = _copy_owned_scene_source(
                    comparison_summary["id"],
                    owner_id,
                    comparison_summary,
                    comparison_path,
                )
                try:
                    with rasterio.open(baseline_path) as baseline, rasterio.open(
                        comparison_path
                    ) as comparison:
                        baseline_mapping = validate_band_mapping(
                            baseline,
                            "ndvi",
                            baseline_metadata,
                            request.baseline_band_mapping,
                        )
                        comparison_mapping = validate_band_mapping(
                            comparison,
                            "ndvi",
                            comparison_metadata,
                            request.comparison_band_mapping,
                        )
                except DatasetError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                except (RasterioError, OSError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail="A stored scene source is not a readable GeoTIFF.",
                    ) from exc
                _validate_scene_band_identity(
                    baseline_metadata,
                    baseline_mapping,
                    "baseline",
                )
                _validate_scene_band_identity(
                    comparison_metadata,
                    comparison_mapping,
                    "comparison",
                )
                for role in ("red", "nir"):
                    baseline_code = next(
                        (
                            band.get("code")
                            for band in baseline_metadata.get("bands", [])
                            if isinstance(band, dict)
                            and band.get("index") == baseline_mapping[role]
                        ),
                        None,
                    )
                    comparison_code = next(
                        (
                            band.get("code")
                            for band in comparison_metadata.get("bands", [])
                            if isinstance(band, dict)
                            and band.get("index") == comparison_mapping[role]
                        ),
                        None,
                    )
                    if (
                        baseline_code in {"B02", "B03", "B04", "B08", "B11"}
                        and comparison_code in {"B02", "B03", "B04", "B08", "B11"}
                        and baseline_code != comparison_code
                    ):
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"The selected {role} bands have conflicting spectral labels "
                                "between scenes; choose corresponding spectral bands."
                            ),
                        )

                try:
                    job_start = persistence_manager.start_job(
                        uuid4().hex,
                        "scene_change_detection",
                        input_parameters,
                        owner_id=owner_id,
                        idempotency_key=idempotency_key,
                    )
                except Exception as exc:
                    logger.error(
                        "Scene change detection could not register its processing job.",
                        extra={
                            "event": "scene_change.job_registration_failed",
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Change-detection metadata storage is temporarily unavailable.",
                    ) from exc

                result_id, job_id = job_start.analysis_id, job_start.job_id
                retrying = False
                if not job_start.created:
                    existing_record = persistence_manager.get_analysis(result_id, owner_id)
                    if (
                        existing_record is None
                        or existing_record["analysis"] != "scene_change_detection"
                        or existing_record["input_parameters"] != input_parameters
                    ):
                        raise HTTPException(status_code=404, detail="Change-detection analysis was not found.")
                    if existing_record["status"] == "failed":
                        try:
                            retrying = persistence_manager.retry_failed_job(
                                result_id,
                                job_id,
                                input_parameters,
                                owner_id,
                            )
                        except Exception as exc:
                            logger.error(
                                "Failed scene change-detection job could not be resumed.",
                                extra={
                                    "event": "scene_change.retry_registration_failed",
                                    "job_id": job_id,
                                    "error_type": type(exc).__name__,
                                },
                            )
                            raise HTTPException(
                                status_code=503,
                                detail="Change-detection metadata storage is temporarily unavailable.",
                            ) from exc
                    if not retrying:
                        if isinstance(existing_record.get("summary"), dict):
                            return {**existing_record["summary"], "duplicate": True}
                        status = existing_record["status"]
                        return {
                            "success": status != "failed",
                            "id": result_id,
                            "job_id": job_id,
                            "analysis": "scene_change_detection",
                            "status": "processing" if status in {"queued", "running"} else status,
                            "error": existing_record.get("error"),
                            "duplicate": True,
                            "artifacts": [],
                        }

                result_directory = settings.output_dir / result_id
                try:
                    output_directory = settings.output_dir.resolve()
                    output_directory.mkdir(parents=True, exist_ok=True)
                    allowed_outputs = {
                        "ndvi-difference.tif",
                        "change-mask.tif",
                        "preview.png",
                        "summary.json",
                    }
                    if result_directory.is_symlink():
                        raise ValueError("Output path is not safe.")
                    if result_directory.exists():
                        if (
                            not result_directory.is_dir()
                            or result_directory.resolve().parent != output_directory
                        ):
                            raise ValueError("Output path is not safe.")
                        if any(
                            item.is_symlink() or item.name not in allowed_outputs
                            for item in result_directory.iterdir()
                        ):
                            raise ValueError("Output directory contains unexpected files.")
                    else:
                        result_directory.mkdir(exist_ok=False)
                except Exception as exc:
                    _fail_scene_analysis_job(
                        job_id,
                        "Scene change-detection outputs could not be staged safely.",
                    )
                    logger.error(
                        "Scene change-detection output directory could not be prepared.",
                        extra={
                            "event": "scene_change.output_directory_failed",
                            "job_id": job_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Change-detection outputs could not be prepared safely.",
                    ) from exc

                try:
                    change_result = run_ndvi_change_detection(
                        baseline_path,
                        comparison_path,
                        result_directory,
                        baseline_metadata=baseline_metadata,
                        comparison_metadata=comparison_metadata,
                        baseline_band_mapping=baseline_mapping,
                        comparison_band_mapping=comparison_mapping,
                        threshold=request.threshold,
                        max_pixels=settings.max_imagery_pixels,
                        max_output_bytes=settings.max_artifact_bytes,
                        deadline_check=_check_analysis_deadline,
                    )
                    _check_analysis_deadline()
                    result = {
                        "success": True,
                        "id": result_id,
                        "job_id": job_id,
                        "analysis": "scene_change_detection",
                        "analysis_type": "ndvi_difference",
                        "status": "completed",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "baseline": {
                            "id": baseline_summary["id"],
                            "filename": baseline_metadata.get("original_filename"),
                            "acquisition_date": baseline_date.isoformat(),
                            "platform": baseline_metadata.get("platform"),
                            "sensor": baseline_metadata.get("sensor"),
                            "band_mapping": baseline_mapping,
                            "source_size_bytes": baseline_artifact["size_bytes"],
                        },
                        "comparison": {
                            "id": comparison_summary["id"],
                            "filename": comparison_metadata.get("original_filename"),
                            "acquisition_date": comparison_date.isoformat(),
                            "platform": comparison_metadata.get("platform"),
                            "sensor": comparison_metadata.get("sensor"),
                            "band_mapping": comparison_mapping,
                            "source_size_bytes": comparison_artifact["size_bytes"],
                        },
                        "result": {
                            **change_result,
                            "compatibility_warnings": compatibility_warnings,
                        },
                        "artifacts": _scene_change_artifacts(result_id),
                    }
                    _store_result(result_id, result_directory, result, job_id)
                    logger.info(
                        "Two-scene NDVI change detection completed.",
                        extra={
                            "event": "scene_change.job_completed",
                            "job_id": job_id,
                            "valid_pixels": change_result["valid_comparison_pixels"],
                        },
                    )
                    return result
                except DatasetError as exc:
                    _fail_scene_analysis_job(job_id, str(exc))
                    raise HTTPException(
                        status_code=422,
                        detail=str(exc),
                        headers={"X-Processing-Job-ID": job_id},
                    ) from exc
                except Exception as exc:
                    _fail_scene_analysis_job(
                        job_id,
                        "Scene change detection or artifact persistence failed.",
                    )
                    if isinstance(exc, HTTPException):
                        raise
                    logger.error(
                        "Scene change detection failed.",
                        extra={
                            "event": "scene_change.job_failed",
                            "job_id": job_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Scene change detection or artifact persistence failed. "
                            "Recoverable local outputs, if any, were retained."
                        ),
                        headers={"X-Processing-Job-ID": job_id},
                    ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Scene change detection could not stage its input scenes.",
                extra={
                    "event": "scene_change.staging_failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="The selected scenes could not be staged for comparison.",
            ) from exc


@router.post("/imagery/scenes/{scene_id}/analyses")
def analyze_imagery_scene(
    scene_id: str,
    request: SceneAnalysisRequest,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    owner_id = user.id if user else None
    scene_summary, _ = _owned_imagery_scene(scene_id, owner_id)
    metadata = scene_summary.get("metadata")
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=422, detail="The imagery scene metadata is incomplete.")
    if request.analysis_type == "kmeans" and request.band_mapping is not None:
        raise HTTPException(
            status_code=422,
            detail="Band-role mappings apply only to NDVI or McFeeters NDWI.",
        )
    if request.analysis_type != "kmeans" and request.feature_bands is not None:
        raise HTTPException(
            status_code=422,
            detail="Feature-band selections apply only to K-means classification.",
        )

    input_parameters = {
        "scene_id": scene_summary["id"],
        "scene_sha256": metadata.get("sha256"),
        "analysis_type": request.analysis_type,
        "band_mapping": request.band_mapping,
        "feature_bands": request.feature_bands,
        "cluster_count": request.cluster_count if request.analysis_type == "kmeans" else None,
        "random_seed": request.random_seed if request.analysis_type == "kmeans" else None,
    }
    idempotency_key = hashlib.sha256(
        json.dumps(input_parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with _analysis_slot():
        try:
            with tempfile.TemporaryDirectory(prefix="satellite-analysis-") as staging:
                source_path = Path(staging) / "source.tif"
                source_artifact = _copy_owned_scene_source(
                    scene_summary["id"],
                    owner_id,
                    scene_summary,
                    source_path,
                )
                try:
                    with rasterio.open(source_path) as source:
                        mapping = validate_band_mapping(
                            source,
                            request.analysis_type,
                            metadata,
                            request.band_mapping,
                        ) if request.analysis_type in {"ndvi", "ndwi"} else None
                        if request.analysis_type == "kmeans":
                            bands = (
                                request.feature_bands
                                if request.feature_bands is not None
                                else list(range(1, source.count + 1))
                            )
                            if any(not 1 <= index <= source.count for index in bands):
                                raise DatasetError(
                                    f"Feature band numbers must be between 1 and {source.count}."
                                )
                except DatasetError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                except (RasterioError, OSError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422,
                        detail="The stored imagery source is not a readable GeoTIFF.",
                    ) from exc

                try:
                    job_start = persistence_manager.start_job(
                        uuid4().hex,
                        f"scene_{request.analysis_type}",
                        input_parameters,
                        owner_id=owner_id,
                        idempotency_key=idempotency_key,
                    )
                except Exception as exc:
                    logger.error(
                        "Satellite scene analysis could not register its processing job.",
                        extra={
                            "event": "scene_analysis.job_registration_failed",
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Analysis metadata storage is temporarily unavailable.",
                    ) from exc

                result_id, job_id = job_start.analysis_id, job_start.job_id
                retrying = False
                existing_record = None
                if not job_start.created:
                    existing_record = persistence_manager.get_analysis(result_id, owner_id)
                    if (
                        existing_record is None
                        or existing_record["analysis"] != f"scene_{request.analysis_type}"
                        or existing_record["input_parameters"] != input_parameters
                    ):
                        raise HTTPException(status_code=404, detail="Analysis was not found.")
                    if existing_record["status"] == "failed":
                        try:
                            retrying = persistence_manager.retry_failed_job(
                                result_id,
                                job_id,
                                input_parameters,
                                owner_id,
                            )
                        except Exception as exc:
                            logger.error(
                                "Failed satellite scene analysis could not be resumed.",
                                extra={
                                    "event": "scene_analysis.retry_registration_failed",
                                    "job_id": job_id,
                                    "error_type": type(exc).__name__,
                                },
                            )
                            raise HTTPException(
                                status_code=503,
                                detail="Analysis metadata storage is temporarily unavailable.",
                            ) from exc
                    if not retrying:
                        if isinstance(existing_record.get("summary"), dict):
                            return {**existing_record["summary"], "duplicate": True}
                        status = existing_record["status"]
                        return {
                            "success": True,
                            "id": result_id,
                            "job_id": job_id,
                            "analysis": existing_record["analysis"],
                            "analysis_type": request.analysis_type,
                            "status": "processing" if status in {"queued", "running"} else status,
                            "error": existing_record.get("error"),
                            "duplicate": True,
                            "artifacts": [],
                        }

                try:
                    output_directory = settings.output_dir.resolve()
                    output_directory.mkdir(parents=True, exist_ok=True)
                    result_directory = settings.output_dir / result_id
                    allowed_outputs = {
                        "ndvi.tif",
                        "ndwi.tif",
                        "classification.tif",
                        "preview.png",
                        "summary.json",
                    }
                    if result_directory.is_symlink():
                        raise ValueError("Output path is not safe.")
                    if result_directory.exists():
                        if (
                            not result_directory.is_dir()
                            or result_directory.resolve().parent != output_directory
                        ):
                            raise ValueError("Output path is not safe.")
                        for existing_path in result_directory.iterdir():
                            if existing_path.is_symlink() or existing_path.name not in allowed_outputs:
                                raise ValueError("Output directory contains unexpected files.")
                except Exception as exc:
                    _fail_scene_analysis_job(
                        job_id,
                        "Satellite scene processing could not prepare a safe output directory.",
                    )
                    logger.error(
                        "Satellite scene analysis output directory could not be prepared.",
                        extra={
                            "event": "scene_analysis.output_directory_failed",
                            "job_id": job_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Analysis outputs could not be prepared safely.",
                    ) from exc

                try:
                    analysis_result = run_satellite_analysis(
                        source_path,
                        result_directory,
                        analysis=request.analysis_type,
                        scene_metadata=metadata,
                        band_mapping=mapping,
                        feature_bands=request.feature_bands,
                        cluster_count=request.cluster_count,
                        random_seed=request.random_seed,
                        max_pixels=settings.max_imagery_pixels,
                        max_output_bytes=settings.max_artifact_bytes,
                        deadline_check=_check_analysis_deadline,
                    )
                    _check_analysis_deadline()
                    result = {
                        "success": True,
                        "id": result_id,
                        "job_id": job_id,
                        "analysis": f"scene_{request.analysis_type}",
                        "analysis_type": request.analysis_type,
                        "status": "completed",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "scene": {
                            "id": scene_summary["id"],
                            "filename": metadata.get("filename"),
                            "acquisition_date": metadata.get("acquisition_date"),
                            "bands": metadata.get("bands", []),
                        },
                        "source_artifact": {
                            "size_bytes": source_artifact["size_bytes"],
                            "sha256": metadata.get("sha256"),
                        },
                        "result": analysis_result,
                        "artifacts": [],
                    }
                    result["artifacts"] = _scene_analysis_artifacts(result_id, result)
                    _store_result(result_id, result_directory, result, job_id)
                    logger.info(
                        "Satellite scene analysis completed.",
                        extra={
                            "event": "scene_analysis.job_completed",
                            "job_id": job_id,
                            "analysis_type": request.analysis_type,
                        },
                    )
                    return result
                except DatasetError as exc:
                    _fail_scene_analysis_job(job_id, str(exc))
                    raise HTTPException(
                        status_code=422,
                        detail=str(exc),
                        headers={"X-Processing-Job-ID": job_id},
                    ) from exc
                except Exception as exc:
                    _fail_scene_analysis_job(
                        job_id,
                        "Satellite scene processing or artifact persistence failed.",
                    )
                    if isinstance(exc, HTTPException):
                        raise
                    logger.error(
                        "Satellite scene analysis failed.",
                        extra={
                            "event": "scene_analysis.job_failed",
                            "job_id": job_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Satellite scene processing or artifact persistence failed. "
                        "Recoverable local outputs, if any, were retained.",
                        headers={"X-Processing-Job-ID": job_id},
                    ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Satellite scene analysis could not be staged.",
                extra={
                    "event": "scene_analysis.staging_failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="The satellite scene could not be staged for analysis.",
            ) from exc


@router.get("/imagery/analyses/{analysis_id}")
def get_imagery_analysis(
    analysis_id: str,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    analysis_id = _validated_id(analysis_id)
    record = persistence_manager.get_analysis(
        analysis_id,
        owner_id=user.id if user else None,
    )
    if record is not None:
        if not isinstance(record["analysis"], str) or not record["analysis"].startswith("scene_"):
            raise HTTPException(status_code=404, detail="Satellite scene analysis was not found.")
        if isinstance(record.get("summary"), dict):
            return record["summary"]
        status = record["status"]
        return {
            "success": status != "failed",
            "id": record["id"],
            "analysis": record["analysis"],
            "analysis_type": record["input_parameters"].get("analysis_type"),
            "status": "processing" if status in {"queued", "running"} else status,
            "error": record.get("error"),
            "artifacts": [],
        }
    if user is not None or settings.authentication_required:
        raise HTTPException(status_code=404, detail="Satellite scene analysis was not found.")
    try:
        summary_path = _result_directory(analysis_id) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (HTTPException, OSError, json.JSONDecodeError):
        raise HTTPException(status_code=404, detail="Satellite scene analysis was not found.")
    if not isinstance(summary, dict) or not str(summary.get("analysis", "")).startswith("scene_"):
        raise HTTPException(status_code=404, detail="Satellite scene analysis was not found.")
    return summary


@router.post("/dataset/{period}/upload", status_code=201)
async def upload_dataset_band(
    period: str,
    file: UploadFile = File(...),
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    if period not in DATASET_PERIODS:
        raise HTTPException(status_code=404, detail="Dataset period must be current or historical.")
    if Path(file.filename or "").suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(status_code=415, detail="Upload a GeoTIFF file with a .tif or .tiff extension.")
    band_code = identify_band(Path(file.filename or "").name)
    if band_code not in SUPPORTED_BANDS:
        raise HTTPException(
            status_code=422,
            detail="Filename must identify one Sentinel-2 band: B02, B03, B04, B08, or B11.",
        )

    dataset_root = _request_dataset_root()
    dataset_dir = (dataset_root / period).resolve()
    if dataset_dir.parent != dataset_root.resolve():
        raise HTTPException(status_code=400, detail="Invalid dataset destination.")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = dataset_dir / f"{band_code}.tif"
    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail=f"{band_code} already exists in the {period} dataset; existing files are not overwritten.",
        )

    temp_path: Path | None = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".satellite-upload-",
            suffix=".tif",
            dir=dataset_dir,
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit.",
                    )
                if total_bytes > settings.max_raster_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the configured raster file-size limit.",
                    )
                temporary.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=422, detail="The uploaded GeoTIFF is empty.")

        try:
            with rasterio.open(temp_path) as source:
                if source.count != 1:
                    raise DatasetError("Each uploaded GeoTIFF must contain exactly one band.")
                if source.crs is None:
                    raise DatasetError("The uploaded GeoTIFF has no CRS; add georeferenced imagery.")
                pixel_count = source.width * source.height
                if pixel_count > settings.max_raster_pixels:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"GeoTIFF has {pixel_count} pixels; the configured limit is "
                            f"{settings.max_raster_pixels} pixels per raster."
                        ),
                    )
                has_valid_pixel = False
                for _, window in source.block_windows(1):
                    block = source.read(1, window=window, masked=True)
                    if np.any(~np.ma.getmaskarray(block) & np.isfinite(block.filled(np.nan))):
                        has_valid_pixel = True
                        break
                if not has_valid_pixel:
                    raise DatasetError("The uploaded GeoTIFF contains no valid pixels.")
        except DatasetError:
            raise
        except (RasterioError, OSError, ValueError) as exc:
            raise DatasetError("The uploaded file is corrupt or is not a readable GeoTIFF.") from exc

        _validate_upload_alignment(temp_path, dataset_dir)
        try:
            os.link(temp_path, destination)
        except FileExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"{band_code} already exists in the {period} dataset; existing files are not overwritten.",
            ) from exc
        return {
            "success": True,
            "period": period,
            "band": band_code,
            "filename": destination.name,
            "size_bytes": total_bytes,
            "message": f"{band_code} uploaded to the {period} dataset.",
        }
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _check_dataset(directory: Path, key: str) -> SatelliteDataset:
    return _validate_dataset(
        _dataset(
            directory,
            REQUIRED_BANDS[key],
            include_other_bands=False,
        ),
        key,
    )


def _historical_analysis(
    result_dir: Path,
    current: SatelliteDataset | None = None,
    historical: SatelliteDataset | None = None,
) -> dict:
    historical_dir = settings.data_dir / "historical"
    if not historical_dir.exists():
        raise DatasetError("Historical imagery is unavailable. Current-period analysis is still available.")
    historical = historical or _dataset(historical_dir, REQUIRED_BANDS["ndvi"])
    if not historical.bands:
        raise DatasetError("Historical imagery is unavailable. Current-period analysis is still available.")
    historical = _validate_dataset(historical, "ndvi")
    current = _validate_dataset(current, "ndvi") if current else _check_dataset(settings.data_dir / "current", "ndvi")
    data = calculate_change(
        current.bands["B04"],
        current.bands["B08"],
        historical.bands["B04"],
        historical.bands["B08"],
    )
    current_band = current.bands["B04"]
    pixel_area = pixel_area_square_metres(current_band)
    changed_pct = data["changed_pixel_percentage"]
    data["changed_area_square_metres"] = (
        data["statistics"]["valid_pixels"] * changed_pct / 100 * pixel_area
        if pixel_area is not None and changed_pct is not None
        else None
    )
    result = _save_analysis(
        result_dir, "change_detection", data, "Historical change detection", "change"
    )
    result["provenance"] = analysis_provenance(
        "change_detection",
        {
            "current": (current, REQUIRED_BANDS["ndvi"]),
            "historical": (historical, REQUIRED_BANDS["ndvi"]),
        },
    )
    return result


def _store_result(result_id: str, directory: Path, response: dict, job_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    serializable = _json_safe(response)
    (directory / "summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    try:
        persistence_manager.persist_result(result_id, job_id, serializable, directory)
    except ArtifactStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Analysis outputs could not be persisted: {exc} "
                "Local outputs have been retained for recovery."
            ),
            headers={"X-Processing-Job-ID": job_id},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Analysis outputs could not be persisted because metadata storage failed. "
                "Local outputs have been retained for recovery."
            ),
            headers={"X-Processing-Job-ID": job_id},
        ) from exc


def _validate_idempotency_key(value: str | None) -> str | None:
    if value is not None and not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", value):
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key must contain 16-128 safe ASCII characters.",
        )
    return value


def _existing_analysis_response(
    analysis_id: str,
    job_id: str,
    analysis: str,
    input_parameters: dict[str, Any],
    owner_id: str | None,
) -> dict:
    record = persistence_manager.get_analysis(analysis_id, owner_id)
    job = persistence_manager.get_job(job_id, owner_id)
    if record is None or job is None:
        raise HTTPException(
            status_code=409,
            detail="The matching analysis request is already registered.",
            headers={"X-Processing-Job-ID": job_id},
        )
    if (
        record["analysis"] != analysis
        or record["input_parameters"] != input_parameters
        or job["input_parameters"] != input_parameters
    ):
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was already used for a different analysis request.",
            headers={"X-Processing-Job-ID": job_id},
        )
    if job["status"] == "completed" and isinstance(record["summary"], dict):
        return record["summary"]
    if job["status"] in {"queued", "running"}:
        detail = "The matching analysis request is still processing."
    else:
        detail = "The matching analysis request failed; use a new Idempotency-Key to retry."
    raise HTTPException(
        status_code=409,
        detail=detail,
        headers={"X-Processing-Job-ID": job_id},
    )


def _run_analysis(
    key: str,
    owner_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict:
    key = ANALYSIS_ALIASES.get(key, key)
    if key not in ANALYSES:
        raise HTTPException(status_code=404, detail="The requested analysis is not available.")
    result_id = uuid4().hex
    input_parameters = {
        "analysis_key": key,
        "dataset_ids": ["current", "historical"] if key == "change_detection" else ["current"],
    }
    job_start = persistence_manager.start_job(
        result_id,
        key,
        input_parameters,
        owner_id=owner_id,
        idempotency_key=idempotency_key,
    )
    result_id, job_id = job_start.analysis_id, job_start.job_id
    if not job_start.created:
        return _existing_analysis_response(
            result_id, job_id, key, input_parameters, owner_id
        )
    result_dir = settings.output_dir / result_id
    try:
        _check_analysis_deadline()
        if key == "change_detection":
            analysis = _historical_analysis(result_dir)
        else:
            dataset = _check_dataset(settings.data_dir / "current", key)
            analysis = _single_result(result_dir, key, dataset)
        _check_analysis_deadline()
        summary = {
            "success": True,
            "id": result_id,
            "job_id": job_id,
            "analysis": analysis["analysis"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": analysis,
        }
        _store_result(result_id, result_dir, summary, job_id)
        logger.info(
            "Analysis job completed.",
            extra={
                "event": "analysis.job_completed",
                "job_id": job_id,
                "analysis": key,
            },
        )
        return summary
    except DatasetError as exc:
        try:
            persistence_manager.fail_job(job_id, str(exc))
        except Exception as state_error:
            logger.error(
                "Failed to persist analysis job failure state.",
                extra={
                    "event": "analysis.job_state_update_failed",
                    "job_id": job_id,
                    "analysis": key,
                    "error_type": type(state_error).__name__,
                },
            )
        logger.warning(
            "Analysis job failed validation.",
            extra={
                "event": "analysis.job_failed",
                "job_id": job_id,
                "analysis": key,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=422,
            detail=str(exc),
            headers={"X-Processing-Job-ID": job_id},
        ) from exc
    except Exception as exc:
        safe_detail = (
            exc.detail
            if isinstance(exc, HTTPException) and isinstance(exc.detail, str)
            else "Analysis processing failed."
        )
        try:
            persistence_manager.fail_job(job_id, safe_detail)
        except Exception as state_error:
            logger.error(
                "Failed to persist analysis job failure state.",
                extra={
                    "event": "analysis.job_state_update_failed",
                    "job_id": job_id,
                    "analysis": key,
                    "error_type": type(state_error).__name__,
                },
            )
        logger.error(
            "Analysis job failed.",
            extra={
                "event": "analysis.job_failed",
                "job_id": job_id,
                "analysis": key,
                "error_type": type(exc).__name__,
            },
        )
        if isinstance(exc, HTTPException) and exc.status_code == 504:
            headers = dict(exc.headers or {})
            headers.setdefault("X-Processing-Job-ID", job_id)
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.detail,
                headers=headers,
            ) from exc
        raise


def _result_directory(result_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", result_id):
        raise HTTPException(status_code=404, detail="Result was not found.")
    directory = (settings.output_dir / result_id).resolve()
    if directory.parent != settings.output_dir.resolve() or not directory.is_dir():
        raise HTTPException(status_code=404, detail="Result was not found.")
    return directory


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "satellite-intelligence-api",
    }


@router.get("/persistence/status")
def persistence_status() -> dict:
    return persistence_manager.status()


@router.get("/auth/status")
def authentication_status() -> dict:
    return {
        "authentication_required": settings.authentication_required,
        "supabase_auth_configured": bool(settings.supabase_url and settings.supabase_anon_key),
    }


def _validated_id(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise HTTPException(status_code=404, detail="Record was not found.")
    return value


@router.get("/analyses/{analysis_id}")
def get_analysis_record(
    analysis_id: str,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    analysis_id = _validated_id(analysis_id)
    record = persistence_manager.get_analysis(analysis_id, user.id if user else None)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis metadata is unavailable.")
    return record


@router.get("/jobs/{job_id}")
def get_processing_job(
    job_id: str,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    job_id = _validated_id(job_id)
    record = persistence_manager.get_job(job_id, user.id if user else None)
    if record is None:
        raise HTTPException(status_code=404, detail="Processing job is unavailable.")
    return record


@router.get("/ready")
def ready() -> JSONResponse:
    report = persistence_manager.readiness(validate_runtime_settings())
    return JSONResponse(
        status_code=200 if report["status"] == "ok" else 503,
        content={
            **report,
        "service": "satellite-intelligence-api",
        "stage": "A",
        },
    )


@router.get("/dataset")
def dataset_status() -> dict:
    try:
        current = _public_dataset(_dataset(settings.data_dir / "current"), "current")
        historical = _public_dataset(
            _dataset(settings.data_dir / "historical"), "historical"
        )
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    provider = _provider_for_mode()
    return {
        "success": True,
        "current": current,
        "historical": historical,
        "provider": provider.name,
        "provider_configured": provider.configured,
        "ready_for_analysis": bool(current["available"]),
        "message": None if current["available"] else EMPTY_DATA_MESSAGE,
    }


@router.get("/satellite/status")
def satellite_status() -> dict:
    provider = _provider_for_mode()
    return {"success": True, **provider.status()}


@router.post(
    "/satellite/catalog/search",
    dependencies=[Depends(current_user)],
)
def satellite_catalog_search(
    request: SentinelCatalogSearchRequest,
) -> dict:
    try:
        return {"success": True, **search_sentinel2_catalog(request)}
    except CatalogError as exc:
        headers = {"Retry-After": "30"} if exc.status_code == 429 else None
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
            headers=headers,
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/satellite/search")
def satellite_search(request: SatelliteSearchRequest) -> dict:
    provider = _provider_for_mode()
    try:
        results = provider.search(request)
    except ProviderNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderCapabilityUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "provider": provider.name, "results": results}


@router.post("/satellite/download")
def satellite_download(
    request: SatelliteDownloadRequest,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    provider = _provider_for_mode()
    try:
        result = provider.download(request.product_id, aoi=request.aoi)
    except ProviderNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderCapabilityUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, KeyError) else 400, detail=str(exc)) from exc
    return {"success": True, "provider": provider.name, "download": result}


@router.get("/satellite/products")
def satellite_products() -> dict:
    provider = _provider_for_mode()
    return {"success": True, "provider": provider.name, "results": provider.list_products()}


@router.get("/satellite/products/{product_id}")
def satellite_product(product_id: str) -> dict:
    provider = _provider_for_mode()
    try:
        return {"success": True, "provider": provider.name, "result": provider.get_product(product_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/satellite/cache")
def satellite_cache() -> dict:
    return {
        "success": True,
        "provider": _provider_for_mode().name,
        "results": _request_cache_manager().list_products(),
    }


@router.delete("/satellite/cache/{product_id}")
def clear_satellite_cache(product_id: str) -> dict:
    try:
        deleted = _request_cache_manager().delete_product(product_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "deleted": deleted, "product_id": product_id}


@router.get("/metadata")
def metadata() -> dict:
    try:
        current = _dataset(settings.data_dir / "current")
        historical = _dataset(settings.data_dir / "historical")
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "success": True,
        "current": {
            "available": bool(current.bands),
            "metadata": raster_metadata(next(iter(current.bands.values()), None)),
            "bands": list(current.bands.keys()),
            "provenance": dataset_provenance(
                "current", current, tuple(current.bands.keys())
            ),
        },
        "historical": {
            "available": bool(historical.bands),
            "metadata": raster_metadata(next(iter(historical.bands.values()), None)),
            "bands": list(historical.bands.keys()),
            "provenance": dataset_provenance(
                "historical", historical, tuple(historical.bands.keys())
            ),
        },
    }


@router.get("/data-quality")
def data_quality() -> dict:
    try:
        current = _dataset(settings.data_dir / "current", SUPPORTED_BANDS)
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "success": True,
        "current": _data_quality_summary(current),
    }


@router.get("/model/status")
def model_status() -> dict:
    model = LandCoverModel()
    return {
        "success": True,
        "model": model.status(),
    }


@router.get("/geoai/status")
def geoai_status() -> dict:
    return {
        "success": True,
        "status": "ok",
        "available": True,
        "components": [
            "spatial_analysis",
            "trend_models",
            "vegetation_forecast",
            "transition_analysis",
            "risk_indicators",
            "model_evaluation",
            "uncertainty",
        ],
        "endpoints": [
            "/api/geoai/status",
            "/api/geoai/demo",
            "/api/geoai/spatial-analysis",
            "/api/geoai/vegetation-forecast",
            "/api/geoai/land-cover-transitions",
            "/api/geoai/model-evaluation",
            "/api/geoai/risk-indicators",
            "/api/geoai/history",
        ],
    }


@router.get("/geoai/demo")
def geoai_demo_scenario(
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    del user
    synthetic_wkt = (
        'LOCAL_CS["Synthetic demo grid",'
        'LOCAL_DATUM["Synthetic local datum",0],UNIT["metre",1],'
        "AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH]]"
    )
    reflectance = {
        "B02": np.full((4, 4), 0.1, dtype=np.float32),
        "B03": np.array(
            [
                [0.4, 0.2, 0.1, 0.3],
                [0.4, 0.2, 0.1, 0.3],
                [0.3, 0.4, 0.2, 0.1],
                [0.3, 0.4, 0.2, 0.1],
            ],
            dtype=np.float32,
        ),
        "B04": np.array(
            [
                [0.1, 0.3, 0.2, 0.4],
                [0.1, 0.3, 0.2, 0.4],
                [0.2, 0.1, 0.3, 0.4],
                [0.2, 0.1, 0.3, 0.4],
            ],
            dtype=np.float32,
        ),
        "B08": np.array(
            [
                [0.6, 0.4, 0.3, 0.4],
                [0.6, 0.4, 0.3, 0.4],
                [0.3, 0.6, 0.4, 0.5],
                [0.3, 0.6, 0.4, 0.5],
            ],
            dtype=np.float32,
        ),
        "B11": np.array(
            [
                [0.2, 0.6, 0.7, 0.2],
                [0.2, 0.6, 0.7, 0.2],
                [0.7, 0.2, 0.6, 0.5],
                [0.7, 0.2, 0.6, 0.5],
            ],
            dtype=np.float32,
        ),
    }
    valid = np.ones((4, 4), dtype=bool)
    bands = {
        code: RasterBand(
            code=code,
            data=values,
            valid=valid.copy(),
            profile={"nodata": None},
            crs=synthetic_wkt,
            transform=Affine.identity(),
            width=4,
            height=4,
            resolution=(1.0, 1.0),
            tags={
                "SATELLITE_VISION_DATA_KIND": "synthetic",
                "SOURCE": "deterministic in-memory hackathon demo fixture",
            },
        )
        for code, values in reflectance.items()
    }
    dataset = SatelliteDataset(
        directory=Path("synthetic-demo-fixture"),
        bands=bands,
        missing_bands=[],
    )
    raw_results = {
        "ndvi": calculate_ndvi(bands["B04"], bands["B08"]),
        "ndwi": calculate_ndwi(bands["B03"], bands["B08"]),
        "ndbi": calculate_ndbi(bands["B08"], bands["B11"]),
        "landcover": LandCoverClassifier().classify(bands),
    }
    return {
        "success": True,
        "scenario_id": "synthetic-index-baseline-v1",
        "data_classification": "synthetic",
        "source": "deterministic in-memory synthetic fixture",
        "acquisition_date": None,
        "geographic_coverage": None,
        "sensor": None,
        "message": (
            "This reproducible 4x4 example is synthetic, not satellite imagery. "
            "It has no acquisition date, real-world coordinates, or validated reflectance."
        ),
        "results": {
            key: _json_safe(result) for key, result in raw_results.items()
        },
        "provenance": {
            key: analysis_provenance(
                key, {"synthetic_demo": (dataset, REQUIRED_BANDS[key])}
            )
            for key in raw_results
        },
    }


@router.post("/geoai/spatial-analysis")
def geoai_spatial_analysis(request: SpatialAnalysisRequest) -> dict:
    if request.aoi is not None:
        raise HTTPException(
            status_code=422,
            detail="AOI clipping is not implemented; this endpoint summarizes the full input raster extent.",
        )
    dataset_dir = settings.data_dir / request.dataset_id
    try:
        dataset = _dataset(dataset_dir)
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    historical = None
    if request.include_history and request.dataset_id == "current":
        try:
            historical = _check_dataset(
                settings.data_dir / "historical", "ndvi"
            )
        except DatasetError as exc:
            history_warning = str(exc)
        else:
            history_warning = None
    else:
        history_warning = None
    result = summarize_spatial_patterns(dataset, historical, block_size=request.block_size)
    provenance_datasets = {
        request.dataset_id: (dataset, ("B04", "B08"))
    }
    if historical is not None:
        provenance_datasets["historical"] = (historical, ("B04", "B08"))
    result["provenance"] = analysis_provenance("ndvi", provenance_datasets)
    result["observation_dates"] = [
        date
        for date in (
            dataset_provenance(
                dataset_id, source_dataset, bands
            ).get("acquisition_date")
            for dataset_id, (source_dataset, bands) in provenance_datasets.items()
        )
        if date
    ]
    if history_warning:
        result["warnings"] = [history_warning]
    return {"success": result.get("status") == "ok", **result}


@router.post("/geoai/vegetation-forecast")
def geoai_vegetation_forecast(request: VegetationForecastRequest) -> dict:
    caller_supplied = request.observations is not None
    warnings: list[str] = []
    if caller_supplied:
        observations = request.observations or []
        warnings.append(
            "Caller-supplied observations are not independently verified as satellite acquisitions."
        )
    else:
        dataset_ids = (
            ("current", "historical")
            if request.dataset_id == "all"
            else (request.dataset_id,)
        )
        observations, warnings = _dated_ndvi_observations(dataset_ids)
    if request.lookback_limit is not None and len(observations) > request.lookback_limit:
        observations = observations[-request.lookback_limit:]
    result = forecast_vegetation_trends(observations, horizon_days=request.horizon_days)
    result["data_source"] = (
        "caller_supplied_unverified"
        if caller_supplied
        else "unverified_raster_acquisition_tags"
    )
    result["data_classification"] = (
        "caller_supplied_unverified"
        if caller_supplied
        else (
            "synthetic"
            if any(
                observation["provenance"]["data_classification"] == "synthetic"
                for observation in observations
            )
            else "unverified_local_input"
        )
    )
    result["observation_warnings"] = warnings
    result["observation_count"] = len(observations)
    if caller_supplied and result.get("status") == "ok":
        result["evaluation"] = {
            "status": "unavailable",
            "message": "Evaluation metrics are suppressed for caller-supplied observations that have not been independently verified.",
            "metrics": {},
        }
    if result.get("status") == "insufficient-data":
        return {"success": False, **result}
    return {"success": True, **result}


@router.post("/geoai/land-cover-transitions")
def geoai_land_cover_transitions(request: TransitionRequest) -> dict:
    try:
        current = _dataset(settings.data_dir / request.current_dataset_id)
        historical = _dataset(settings.data_dir / request.historical_dataset_id)
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = analyze_land_cover_transitions(current, historical)
    result["provenance"] = analysis_provenance(
        "landcover",
        {
            "current": (current, REQUIRED_BANDS["landcover"]),
            "historical": (historical, REQUIRED_BANDS["landcover"]),
        },
    )
    if result.get("status") == "insufficient-data":
        return {"success": False, **result}
    return {"success": True, **result}


@router.get("/geoai/model-evaluation")
def geoai_model_evaluation() -> dict:
    observations, warnings = _dated_ndvi_observations()
    if len(observations) < 5:
        return {
            "success": False,
            "status": "insufficient-data",
            "model": "linear-trend forecast baseline",
            "data_classification": (
                "synthetic"
                if any(
                    item["provenance"]["data_classification"] == "synthetic"
                    for item in observations
                )
                else "unverified_local_input"
            ),
            "message": "Evaluation requires at least five distinct, metadata-dated NDVI observations; the available inputs do not establish supervised land-cover model accuracy.",
            "observation_count": len(observations),
            "observation_warnings": warnings,
            "metrics": {},
        }
    result = evaluate_forecast(
        [item["mean_ndvi"] for item in observations]
    )
    result["model"] = "linear-trend forecast baseline"
    result["observation_dates"] = [item["date"] for item in observations]
    result["observation_warnings"] = warnings
    result["data_classification"] = (
        "synthetic"
        if any(
            item["provenance"]["data_classification"] == "synthetic"
            for item in observations
        )
        else "unverified_local_input"
    )
    result["limitations"] = (
        "This is a time-ordered evaluation of the simple NDVI trend baseline, "
        "not a supervised land-cover classifier evaluation. It uses only "
        "metadata-dated local rasters and is not a substitute for independent ground truth."
    )
    return {"success": result.get("status") == "ok", **result}


@router.get("/geoai/risk-indicators")
def geoai_risk_indicators() -> dict:
    try:
        current = _check_dataset(settings.data_dir / "current", "ndvi")
    except DatasetError as exc:
        return {
            "success": False,
            "status": "insufficient-data",
            "message": str(exc),
            "indicators": [],
            "count": 0,
        }
    spatial = summarize_spatial_patterns(current)
    if spatial.get("status") != "ok":
        return {
            "success": False,
            "status": "insufficient-data",
            "message": spatial.get("message", "Current raster data is insufficient."),
            "indicators": [],
            "count": 0,
        }
    provenance_datasets = {"current": (current, REQUIRED_BANDS["ndvi"])}
    current_date = dataset_provenance(
        "current", current, REQUIRED_BANDS["ndvi"]
    ).get("acquisition_date")
    observation_dates = [current_date] if current_date else []
    transition_summary = None
    try:
        historical = _check_dataset(settings.data_dir / "historical", "ndvi")
    except DatasetError:
        historical = None
    if historical is not None:
        provenance_datasets["historical"] = (
            historical,
            REQUIRED_BANDS["ndvi"],
        )
        historical_date = dataset_provenance(
            "historical", historical, REQUIRED_BANDS["ndvi"]
        ).get("acquisition_date")
        if historical_date:
            observation_dates.insert(0, historical_date)
        spatial = summarize_spatial_patterns(current, historical)
        if all(code in current.bands and code in historical.bands for code in REQUIRED_BANDS["landcover"]):
            transition_summary = analyze_land_cover_transitions(
                current, historical
            )
            if transition_summary.get("status") != "ok":
                transition_summary = None
    spatial["observation_dates"] = observation_dates
    spatial["data_quality"] = {
        "status": "available",
        "source": "available local raster inputs",
        "warnings": [
            warning
            for dataset_id, (source_dataset, band_codes) in provenance_datasets.items()
            for warning in dataset_provenance(
                dataset_id, source_dataset, band_codes
            )["quality_warnings"]
        ],
    }
    result = detect_risk_indicators(
        spatial, transition_summary=transition_summary
    )
    result["evidence_sources"] = {
        dataset_id: dataset_provenance(dataset_id, source_dataset, band_codes)
        for dataset_id, (source_dataset, band_codes) in provenance_datasets.items()
    }
    result["data_classification"] = (
        "synthetic"
        if any(
            source["data_classification"] == "synthetic"
            for source in result["evidence_sources"].values()
        )
        else "unverified_local_input"
    )
    return {"success": True, **result}


@router.get("/geoai/history")
def geoai_history() -> dict:
    observations, warnings = _dated_ndvi_observations()
    if len(observations) < 2:
        return {
            "success": False,
            "status": "insufficient-data",
            "message": "At least two distinct raster acquisition dates are required; untagged or undated inputs are not treated as a time series.",
            "observation_count": len(observations),
            "observation_warnings": warnings,
            "series": observations,
        }
    summary = summarize_observation_history(observations)
    summary["observation_warnings"] = warnings
    summary["data_classification"] = (
        "synthetic"
        if any(
            observation["provenance"]["data_classification"] == "synthetic"
            for observation in observations
        )
        else "unverified_local_input"
    )
    return {"success": summary.get("status") == "ok", **summary}


def _visualization_path(kind: str) -> Path:
    if settings.authentication_required:
        data_root = _request_dataset_root()
        output_dir = settings.output_dir / "users" / data_root.name / "visualizations"
    else:
        output_dir = settings.output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{kind}.png"


def _generate_visualization(kind: str) -> Path:
    if kind == "rgb":
        dataset = _dataset(settings.data_dir / "current")
        dataset.require(("B02", "B03", "B04"))
        output = _visualization_path(kind)
        save_rgb_preview(dataset.bands["B04"], dataset.bands["B03"], dataset.bands["B02"], output)
        return output
    if kind == "ndvi":
        dataset = _check_dataset(settings.data_dir / "current", "ndvi")
        result = calculate_ndvi(dataset.bands["B04"], dataset.bands["B08"])
        output = _visualization_path(kind)
        save_index_png(result["raster"], result["valid_mask"], output, "ndvi")
        return output
    if kind == "ndwi":
        dataset = _check_dataset(settings.data_dir / "current", "ndwi")
        result = calculate_ndwi(dataset.bands["B03"], dataset.bands["B08"])
        output = _visualization_path(kind)
        save_index_png(result["raster"], result["valid_mask"], output, "ndwi")
        return output
    if kind == "ndbi":
        dataset = _check_dataset(settings.data_dir / "current", "ndbi")
        result = calculate_ndbi(dataset.bands["B08"], dataset.bands["B11"])
        output = _visualization_path(kind)
        save_index_png(result["raster"], result["valid_mask"], output, "ndbi")
        return output
    if kind == "landcover":
        dataset = _check_dataset(settings.data_dir / "current", "landcover")
        result = LandCoverClassifier().classify(dataset.bands)
        output = _visualization_path(kind)
        save_class_png(result["raster"], result["valid_mask"], output)
        return output
    if kind == "change":
        current = _check_dataset(settings.data_dir / "current", "ndvi")
        historical = _dataset(settings.data_dir / "historical")
        if not historical.bands:
            raise DatasetError("Historical imagery is unavailable for change visualization.")
        historical = _validate_dataset(historical, "ndvi")
        result = calculate_change(current.bands["B04"], current.bands["B08"], historical.bands["B04"], historical.bands["B08"])
        output = _visualization_path(kind)
        save_index_png(result["raster"], result["valid_mask"], output, "change")
        return output
    raise HTTPException(status_code=404, detail="Visualization layer was not found.")


@router.get("/visualization/{kind}")
def visualization(
    kind: str,
    user: AuthenticatedUser | None = Depends(current_user),
):
    if kind not in {"rgb", "ndvi", "ndwi", "ndbi", "landcover", "change"}:
        raise HTTPException(status_code=404, detail="Visualization layer was not found.")
    try:
        path = _generate_visualization(kind)
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/png")


@router.post("/analyze/all")
def analyze_all(
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    idempotency_key = _validate_idempotency_key(idempotency_key)
    with _analysis_slot():
        return _run_all_analysis(idempotency_key, user)


def _run_all_analysis(
    idempotency_key: str | None,
    user: AuthenticatedUser | None,
) -> dict:
    result_id = uuid4().hex
    input_parameters = {
        "analysis_keys": list(ANALYSES),
        "dataset_ids": ["current", "historical"],
    }
    job_start = persistence_manager.start_job(
        result_id,
        "all",
        input_parameters,
        owner_id=user.id if user else None,
        idempotency_key=idempotency_key,
    )
    result_id, job_id = job_start.analysis_id, job_start.job_id
    if not job_start.created:
        return _existing_analysis_response(
            result_id,
            job_id,
            "all",
            input_parameters,
            user.id if user else None,
        )
    result_dir = settings.output_dir / result_id
    try:
        _check_analysis_deadline()
        current = _dataset(settings.data_dir / "current")
        historical = _dataset(settings.data_dir / "historical")
        results = {}
        for key in ANALYSES:
            _check_analysis_deadline()
            try:
                if key == "change_detection":
                    results[key] = {
                        "success": True,
                        **_historical_analysis(result_dir, current=current, historical=historical),
                    }
                else:
                    validated = _validate_dataset(current, key)
                    results[key] = {
                        "success": True,
                        **_single_result(result_dir, key, validated),
                    }
            except DatasetError as exc:
                results[key] = {"success": False, "message": str(exc)}
        response = {
            "success": True,
            "id": result_id,
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "current": bool(current.bands),
                "historical": bool(historical.bands),
                "message": None if current.bands else EMPTY_DATA_MESSAGE,
            },
            **results,
        }
        _check_analysis_deadline()
        _store_result(result_id, result_dir, response, job_id)
        return response
    except Exception as exc:
        persistence_manager.fail_job(job_id, f"{type(exc).__name__}: {exc}")
        if isinstance(exc, HTTPException) and exc.status_code == 504:
            headers = dict(exc.headers or {})
            headers.setdefault("X-Processing-Job-ID", job_id)
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.detail,
                headers=headers,
            ) from exc
        raise


@router.post("/analyze/{analysis_key}")
def analyze(
    analysis_key: str,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    with _analysis_slot():
        return _run_analysis(
            analysis_key,
            user.id if user else None,
            _validate_idempotency_key(idempotency_key),
        )


@router.get("/results")
def list_results(
    limit: int = Query(DEFAULT_RESULTS_PAGE_SIZE, ge=1, le=MAX_RESULTS_PAGE_SIZE),
    offset: int = Query(0, ge=0, le=10_000),
    include_incomplete: bool = Query(False),
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    results_by_id = {}
    owner_id = user.id if user else None
    has_more = False
    if include_incomplete:
        persisted_results = persistence_manager.list_analyses(
            owner_id,
            limit=limit,
            offset=offset,
            include_incomplete=True,
        )
    else:
        persisted_results = persistence_manager.list_analyses(
            owner_id,
            limit=limit,
            offset=offset,
        )
    if persisted_results is not None:
        has_more = len(persisted_results) > limit
        for result in persisted_results[:limit]:
            item = {
                "id": result["id"],
                "created_at": result["created_at"],
                "analysis": result["analysis"],
            }
            if include_incomplete:
                item["status"] = result.get("status")
                item["completed_at"] = result.get("completed_at")
                item["history"] = _analysis_history_context(
                    result.get("summary"),
                    result.get("input_parameters"),
                )
            results_by_id[result["id"]] = item
        page = list(results_by_id.values())
        return {
            "success": True,
            "results": page,
            "has_more": has_more,
            "next_offset": offset + len(page) if has_more else None,
        }

    output_dir = settings.output_dir
    if user is None and not settings.authentication_required and output_dir.exists():
        def modified_time(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        candidate_limit = offset + limit + 1
        candidates = heapq.nlargest(
            candidate_limit,
            (
                path
                for path in output_dir.iterdir()
                if path.is_dir() and (path / "summary.json").is_file()
            ),
            key=modified_time,
        )
        has_more = len(candidates) > offset + limit
        for path in candidates[offset : offset + limit]:
            summary_path = path / "summary.json"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary_id = summary.get("id")
                if summary_id:
                    item = {
                        "id": summary_id,
                        "created_at": summary.get("created_at"),
                        "analysis": summary.get("analysis", "Analysis bundle"),
                    }
                    if include_incomplete:
                        item["status"] = "completed"
                        item["history"] = _analysis_history_context(summary, {})
                    results_by_id[summary_id] = item
            except (OSError, json.JSONDecodeError):
                continue
    page = list(results_by_id.values())
    return {
        "success": True,
        "results": page,
        "has_more": has_more if user is None and not settings.authentication_required else False,
        "next_offset": (
            offset + len(page)
            if has_more and user is None and not settings.authentication_required
            else None
        ),
    }


def _analysis_history_context(
    summary: Any,
    input_parameters: Any,
) -> dict[str, Any]:
    summary = summary if isinstance(summary, dict) else {}
    input_parameters = input_parameters if isinstance(input_parameters, dict) else {}
    analysis_result = summary.get("result")
    analysis_result = analysis_result if isinstance(analysis_result, dict) else summary

    parameter_keys = (
        "scene_id",
        "analysis_type",
        "method",
        "band_mapping",
        "feature_bands",
        "cluster_count",
        "random_seed",
        "baseline_scene_id",
        "baseline_band_mapping",
        "comparison_scene_id",
        "comparison_band_mapping",
        "threshold",
        "threshold_units",
        "analysis_key",
        "analysis_keys",
        "dataset_ids",
    )
    parameters = {
        key: input_parameters[key]
        for key in parameter_keys
        if key in input_parameters
    }

    source_scenes = []
    for role, source_key, parameter_key in (
        ("Scene", "scene", "scene_id"),
        ("Baseline", "baseline", "baseline_scene_id"),
        ("Comparison", "comparison", "comparison_scene_id"),
    ):
        source = summary.get(source_key)
        source = source if isinstance(source, dict) else {}
        metadata = source.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        scene_id = source.get("id") or input_parameters.get(parameter_key)
        filename = (
            source.get("filename")
            or metadata.get("original_filename")
            or metadata.get("filename")
        )
        if isinstance(filename, str):
            filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        acquisition_date = (
            source.get("acquisition_date")
            or metadata.get("acquisition_date")
            or input_parameters.get(
                f"{source_key}_acquisition_date"
                if source_key != "scene"
                else "acquisition_date"
            )
        )
        if scene_id or filename or acquisition_date:
            source_scenes.append(
                {
                    "role": role,
                    "id": scene_id,
                    "filename": filename,
                    "acquisition_date": acquisition_date,
                    "platform": source.get("platform"),
                    "sensor": source.get("sensor"),
                    "band_mapping": source.get("band_mapping"),
                }
            )

    if not source_scenes and summary.get("analysis") == "imagery_ingestion":
        metadata = summary.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        filename = metadata.get("original_filename") or metadata.get("filename")
        if isinstance(filename, str):
            filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        source_scenes.append(
            {
                "role": "Uploaded scene",
                "id": summary.get("id"),
                "filename": filename,
                "acquisition_date": metadata.get("acquisition_date"),
            }
        )

    statistics = {}

    def safe_history_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: safe_history_value(child)
                for key, child in value.items()
                if not any(
                    sensitive in key.lower()
                    for sensitive in (
                        "owner",
                        "error",
                        "url",
                        "path",
                        "bucket",
                        "storage",
                        "token",
                        "secret",
                        "credential",
                        "download",
                    )
                )
            }
        if isinstance(value, list):
            return [safe_history_value(child) for child in value]
        return value

    def collect_statistics(value: Any, label: str = "", depth: int = 0) -> None:
        if not isinstance(value, dict) or depth > 5:
            return
        statistic_values = value.get("statistics")
        if isinstance(statistic_values, dict):
            statistics[label or "Summary"] = safe_history_value(statistic_values)
        for key, child in value.items():
            if key != "statistics" and isinstance(child, dict):
                collect_statistics(child, f"{label}.{key}".strip("."), depth + 1)

    collect_statistics(analysis_result)
    metric_keys = (
        "total_pixels",
        "total_target_pixels",
        "valid_comparison_pixels",
        "valid_pixels",
        "excluded_pixels",
        "changed_pixels",
        "unchanged_pixels",
        "valid_pixel_percentage",
        "change_percentage_of_valid_pixels",
        "positive_change_pixels",
        "negative_change_pixels",
        "area_square_metres",
        "area_square_kilometres",
        "valid_comparison_area_square_metres",
        "changed_area_square_metres",
        "positive_change_area_square_metres",
        "negative_change_area_square_metres",
    )
    metrics = {
        key: analysis_result[key]
        for key in metric_keys
        if key in analysis_result
    }
    if isinstance(analysis_result.get("classes"), list):
        metrics["spectral_clusters"] = safe_history_value(analysis_result["classes"])
    raster = analysis_result.get("raster")
    raster = raster if isinstance(raster, dict) else {}
    metadata = summary.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    spatial_keys = (
        "crs",
        "bounds_wgs84",
        "width",
        "height",
        "resolution",
        "resolution_units",
        "transform",
        "nodata",
        "nodata_value",
        "dtype",
        "extent_wgs84",
    )
    spatial_metadata = {
        key: analysis_result[key]
        for key in spatial_keys
        if key in analysis_result
    }
    if "bounds" in analysis_result and "bounds_wgs84" not in spatial_metadata:
        spatial_metadata["bounds_wgs84"] = analysis_result["bounds"]
    for key in (*spatial_keys, "bounds"):
        if key in raster and key not in spatial_metadata:
            spatial_metadata[key] = raster[key]
    alignment = analysis_result.get("alignment")
    if isinstance(alignment, dict):
        spatial_metadata["alignment"] = safe_history_value(
            {
                key: alignment[key]
                for key in (
                    "target",
                    "resampling",
                    "baseline_crs",
                    "baseline_transform",
                    "baseline_width",
                    "baseline_height",
                    "comparison_was_reprojected_or_resampled",
                )
                if key in alignment
            }
        )
    if not spatial_metadata and metadata:
        spatial_metadata.update({
            key: metadata[key]
            for key in (*spatial_keys, "bounds")
            if key in metadata
        })

    method = (
        summary.get("analysis_type")
        or analysis_result.get("analysis_type")
        or analysis_result.get("method")
        or input_parameters.get("analysis_type")
        or input_parameters.get("method")
        or input_parameters.get("analysis_key")
    )
    threshold = analysis_result.get("threshold")
    if threshold is None:
        threshold = input_parameters.get("threshold")

    return {
        "source_scenes": source_scenes,
        "parameters": parameters,
        "method": method,
        "statistics": statistics,
        "metrics": metrics,
        "threshold": threshold,
        "threshold_units": (
            analysis_result.get("threshold_units")
            or input_parameters.get("threshold_units")
        ),
        "limitations": (
            analysis_result.get("limitations")
            or analysis_result.get("interpretation_note")
        ),
        "warnings": safe_history_value(
            analysis_result.get("warnings")
            or analysis_result.get("area_warnings")
            or []
        ),
        "spatial_metadata": spatial_metadata,
    }


@router.get("/results/{result_id}")
def get_result(
    result_id: str,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    result_id = _validated_id(result_id)
    record = persistence_manager.get_analysis(result_id, user.id if user else None)
    if record is not None:
        if record["status"] != "completed" or record["summary"] is None:
            raise HTTPException(status_code=404, detail="Result was not found.")
        return record["summary"]
    if user is not None or settings.authentication_required:
        raise HTTPException(status_code=404, detail="Result was not found.")
    try:
        summary_path = _result_directory(result_id) / "summary.json"
    except HTTPException:
        raise HTTPException(status_code=404, detail="Result was not found.")
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail="Result was not found.")
    return json.loads(summary_path.read_text(encoding="utf-8"))


@router.get("/results/{result_id}/artifacts")
def list_result_artifacts(
    result_id: str,
    user: AuthenticatedUser | None = Depends(current_user),
) -> dict:
    result_id = _validated_id(result_id)
    artifacts = []
    for artifact in persistence_manager.get_artifacts(
        result_id, user.id if user else None
    ):
        artifacts.append(
            {
                "artifact_name": artifact["artifact_name"],
                "media_type": artifact["media_type"],
                "artifact_type": artifact.get("artifact_type", "file"),
                "size_bytes": artifact["size_bytes"],
                "created_at": artifact.get("created_at"),
                "download_url": (
                    f"/api/results/{result_id}/artifacts/"
                    f"{quote(artifact['artifact_name'], safe='')}/download"
                ),
            }
        )
    return {
        "success": True,
        "result_id": result_id,
        "artifacts": artifacts,
    }


@router.get("/results/{result_id}/artifacts/{artifact_name}/download")
def download_registered_artifact(
    result_id: str,
    artifact_name: str,
    user: AuthenticatedUser | None = Depends(current_user),
):
    result_id = _validated_id(result_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", artifact_name):
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    try:
        stored = persistence_manager.get_artifact(
            result_id,
            artifact_name,
            owner_id=user.id if user else None,
        )
    except (ArtifactStorageError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Analysis artifact storage is temporarily unavailable.",
        ) from exc
    if stored is None:
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    file_obj, artifact = stored

    def chunks():
        try:
            while chunk := file_obj.read(1024 * 1024):
                yield chunk
        finally:
            file_obj.close()

    return StreamingResponse(
        chunks(),
        media_type=artifact["media_type"] or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{artifact_name}"'},
    )


@router.get("/results/{result_id}/signed-download")
def create_signed_download(
    result_id: str,
    artifact_name: str,
    user: AuthenticatedUser | None = Depends(current_user),
):
    result_id = _validated_id(result_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", artifact_name):
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    try:
        signed_url = persistence_manager.create_signed_download_url(
            result_id,
            artifact_name,
            owner_id=user.id if user else None,
            expires_in=300,
        )
    except (ArtifactStorageError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Analysis artifact storage is temporarily unavailable.",
        ) from exc
    if signed_url is None:
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    return {"signed_url": signed_url, "expires_in": 300}


def _stream_artifact(
    result_id: str,
    filename: str,
    media_type: str,
    download: bool = False,
    owner_id: str | None = None,
):
    result_id = _validated_id(result_id)
    try:
        stored = persistence_manager.get_artifact(result_id, filename, owner_id)
    except (ArtifactStorageError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Analysis artifact storage is temporarily unavailable.",
        ) from exc
    if stored is not None:
        file_obj, artifact = stored
        actual_media_type = artifact["media_type"] or media_type
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if download else {}

        def chunks():
            try:
                while chunk := file_obj.read(1024 * 1024):
                    yield chunk
            finally:
                file_obj.close()

        return StreamingResponse(chunks(), media_type=actual_media_type, headers=headers)
    if owner_id is not None or settings.authentication_required:
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    try:
        path = _result_directory(result_id) / filename
    except HTTPException:
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename if download else None,
    )


@router.get("/results/{result_id}/download/{file_key}")
def download_result(
    result_id: str,
    file_key: str,
    user: AuthenticatedUser | None = Depends(current_user),
):
    if file_key not in DOWNLOAD_FILES:
        raise HTTPException(status_code=404, detail="Download was not found.")
    return _stream_artifact(
        result_id,
        DOWNLOAD_FILES[file_key],
        mimetypes.guess_type(DOWNLOAD_FILES[file_key])[0] or "application/octet-stream",
        download=True,
        owner_id=user.id if user else None,
    )


@router.get("/results/{result_id}/image/{analysis_key}")
def get_result_image(
    result_id: str,
    analysis_key: str,
    user: AuthenticatedUser | None = Depends(current_user),
):
    if analysis_key not in ANALYSES:
        raise HTTPException(status_code=404, detail="Map layer was not found.")
    return _stream_artifact(
        result_id, f"{analysis_key}.png", "image/png", owner_id=user.id if user else None
    )


def _monitoring_for_user(
    user: AuthenticatedUser | None,
) -> MonitoringPipeline:
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    if monitoring_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Persistent satellite monitoring is unavailable.",
        )
    return monitoring_pipeline


@router.get("/monitoring/status")
def get_monitoring_status(
    user: AuthenticatedUser | None = Depends(current_user),
):
    pipeline = _monitoring_for_user(user)
    try:
        return pipeline.status()
    except (SQLAlchemyError, ValueError) as exc:
        logger.error(
            "Persistent satellite monitoring status is unavailable.",
            extra={"event": "monitor.api_status_failed", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Persistent satellite monitoring state is unavailable.",
        ) from exc


@router.get("/monitoring/scenes")
def list_monitored_scenes(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100_000),
    user: AuthenticatedUser | None = Depends(current_user),
):
    pipeline = _monitoring_for_user(user)
    try:
        scenes = pipeline.store.list_scenes(limit, offset)
    except SQLAlchemyError as exc:
        logger.error(
            "Persistent satellite scene history is unavailable.",
            extra={"event": "monitor.api_scenes_failed", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Persistent satellite monitoring history is unavailable.",
        ) from exc
    return {"scenes": scenes, "limit": limit, "offset": offset}


@router.get("/monitoring/scenes/{scene_id}")
def get_monitored_scene(
    scene_id: str,
    user: AuthenticatedUser | None = Depends(current_user),
):
    pipeline = _monitoring_for_user(user)
    try:
        scene = pipeline.store.scene(scene_id)
    except SQLAlchemyError as exc:
        logger.error(
            "Persistent satellite scene details are unavailable.",
            extra={"event": "monitor.api_scene_failed", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Persistent satellite monitoring history is unavailable.",
        ) from exc
    if scene is None:
        raise HTTPException(status_code=404, detail="Monitored scene was not found.")
    return scene


@router.post("/monitoring/scan", status_code=202)
async def request_monitoring_scan(
    user: AuthenticatedUser | None = Depends(current_user),
):
    pipeline = _monitoring_for_user(user)
    if not pipeline.configured or monitoring_scheduler is None:
        raise HTTPException(
            status_code=503,
            detail=pipeline._disabled_reason(),
        )
    try:
        accepted = await monitoring_scheduler.request_manual_scan()
    except SQLAlchemyError as exc:
        logger.error(
            "A manual satellite scan could not be scheduled.",
            extra={"event": "monitor.api_manual_scan_failed", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Persistent satellite monitoring is unavailable.",
        ) from exc
    if not accepted:
        raise HTTPException(
            status_code=409,
            detail="A satellite monitoring scan is already running.",
        )
    return {"status": "accepted", "message": "Satellite scan queued."}


@router.get("/monitoring/scenes/{scene_id}/artifacts/{artifact_name}")
def get_monitoring_artifact(
    scene_id: str,
    artifact_name: str,
    user: AuthenticatedUser | None = Depends(current_user),
):
    pipeline = _monitoring_for_user(user)
    if artifact_name not in {"classification.tif", "confidence.tif"}:
        raise HTTPException(status_code=404, detail="Monitoring artifact was not found.")
    try:
        key = pipeline.store.artifact_key(scene_id, artifact_name)
    except SQLAlchemyError as exc:
        logger.error(
            "Persistent satellite analysis metadata is unavailable.",
            extra={"event": "monitor.api_artifact_failed", "error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="Persistent satellite monitoring history is unavailable.",
        ) from exc
    if not key:
        raise HTTPException(status_code=404, detail="Monitoring artifact was not found.")
    root = pipeline.settings.satellite_monitor_data_dir.resolve()
    path = (root / key).resolve()
    if (
        Path(key).is_absolute()
        or ".." in Path(key).parts
        or not path.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise HTTPException(status_code=404, detail="Monitoring artifact was not found.")
    media_type = "image/tiff"
    return FileResponse(
        path,
        media_type=media_type,
        filename=artifact_name,
        headers={"Cache-Control": "no-store"},
    )
