from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.util import find_spec
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import joblib
import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.windows import Window
import sklearn

from config import Settings, settings
from processing.preprocessing import identify_band

LANDCOVER_MODEL_ID = "sentinel2-landcover-sklearn"
LANDCOVER_BANDS = ("B02", "B03", "B04", "B08")
BLOCK_SIZE = 256
MAX_CLASSES = 65_534
NORMALIZATIONS = {"stored_values", "divide_by_10000"}
PRITHVI_TINY_TL_REPOSITORY = (
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL"
)
PRITHVI_TINY_TL_REVISION = "335eadc2c45ad5abe7bd307223e1c48c5b60c41b"


class InferenceError(Exception):
    status_code = 422
    detail = "AI inference could not be completed for this scene."


class ModelNotReadyError(InferenceError):
    status_code = 503
    detail = "The local Sentinel-2 land-cover model is not ready."


class ModelConfigurationError(InferenceError):
    status_code = 503
    detail = "The local Sentinel-2 land-cover model configuration is invalid."


class InferenceInputError(InferenceError):
    status_code = 422
    detail = "The stored scene does not meet this model's input requirements."


class InferenceExecutionError(InferenceError):
    status_code = 422
    detail = "The configured model could not produce valid class probabilities."


class InferenceTimeoutError(InferenceError):
    status_code = 504
    detail = "AI inference exceeded its configured processing time limit."


@dataclass(frozen=True)
class LandCoverModelSpec:
    name: str
    version: str
    license: str
    source: str
    attribution: str
    artifact_path: Path
    artifact_size_bytes: int
    artifact_sha256: str
    bands: tuple[str, ...]
    normalization: str
    class_mapping: dict[str, str]


def _within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _is_public_path(path: Path) -> bool:
    backend_root = Path(__file__).resolve().parents[1]
    project_root = backend_root.parent
    public_directories = (
        project_root / "frontend" / "public",
        project_root / "frontend" / "dist",
        backend_root / "static",
    )
    return any(_within(path, directory.resolve()) for directory in public_directories)


def _safe_model_root(configuration: Settings) -> Path:
    root = configuration.ai_model_dir.resolve()
    if _is_public_path(root):
        raise ModelConfigurationError("AI model files must not be stored in a public directory.")
    return root


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_model_spec(configuration: Settings) -> tuple[LandCoverModelSpec | None, str | None]:
    try:
        root = _safe_model_root(configuration)
        configured_manifest = configuration.ai_landcover_manifest_path
        if configured_manifest.is_symlink():
            raise ModelConfigurationError("The model manifest may not be a symbolic link.")
        manifest_path = configured_manifest.resolve()
        if not _within(manifest_path, root) or _is_public_path(manifest_path):
            raise ModelConfigurationError()
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return None, "Configure a local model manifest and trusted weights in AI_MODEL_DIR."
        if manifest_path.stat().st_size > 64 * 1024:
            raise ModelConfigurationError()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ModelConfigurationError as exc:
        return None, str(exc) or exc.detail
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "The local model manifest could not be read as valid JSON."

    if not isinstance(manifest, dict):
        return None, "The local model manifest must contain a JSON object."
    deployment_approved = manifest.get("deployment_approved")
    if deployment_approved is not None and not isinstance(deployment_approved, bool):
        return None, "The model deployment approval flag must be a boolean."
    if deployment_approved is not True:
        return None, (
            "This model artifact is not approved for production inference; "
            "deployment_approved must be true."
        )
    text_fields = ("name", "version", "license", "source", "attribution", "artifact")
    if any(
        not isinstance(manifest.get(field), str)
        or not manifest[field].strip()
        or len(manifest[field]) > 512
        for field in text_fields
    ):
        return None, "The local model manifest is missing required model and attribution fields."
    bands = manifest.get("bands")
    if (
        not isinstance(bands, list)
        or len(bands) != len(LANDCOVER_BANDS)
        or any(not isinstance(band, str) for band in bands)
        or set(bands) != set(LANDCOVER_BANDS)
    ):
        return None, "The model manifest must specify each required Sentinel-2 band exactly once."
    normalization = manifest.get("normalization")
    if not isinstance(normalization, str) or normalization not in NORMALIZATIONS:
        return None, "Model normalization must be stored_values or divide_by_10000."
    class_mapping = manifest.get("class_mapping", {})
    if not isinstance(class_mapping, dict) or any(
        not isinstance(code, str)
        or not code.isdigit()
        or code == "0"
        or not isinstance(name, str)
        or not name.strip()
        or len(name) > 128
        for code, name in class_mapping.items()
    ):
        return None, "The model class mapping must map positive class codes to non-empty names."
    if len(set(class_mapping.values())) != len(class_mapping):
        return None, "The model class mapping contains duplicate class names."

    artifact_name = Path(manifest["artifact"])
    if (
        artifact_name.is_absolute()
        or not artifact_name.parts
        or any(part in {".", ".."} for part in artifact_name.parts)
    ):
        return None, "The model artifact must be a relative path inside AI_MODEL_DIR."
    configured_artifact = root / artifact_name
    if configured_artifact.is_symlink():
        return None, "The model artifact may not be a symbolic link."
    artifact_path = configured_artifact.resolve()
    if (
        not _within(artifact_path, root)
        or _is_public_path(artifact_path)
        or artifact_path.is_symlink()
        or not artifact_path.is_file()
    ):
        return None, "The configured model weights are not available in the private model directory."
    try:
        artifact_size = artifact_path.stat().st_size
    except OSError:
        return None, "The configured model weights could not be inspected."
    if artifact_size <= 0 or artifact_size > configuration.max_ai_model_bytes:
        return None, "The configured model weights are empty or exceed MAX_AI_MODEL_BYTES."
    try:
        artifact_sha256 = _hash_file(artifact_path)
    except OSError:
        return None, "The configured model weights could not be read."
    expected_sha256 = manifest.get("artifact_sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        return None, "The model manifest must declare a valid artifact SHA-256 checksum."
    if expected_sha256 != artifact_sha256:
        return None, "The model artifact checksum does not match its manifest."
    return (
        LandCoverModelSpec(
            name=manifest["name"].strip(),
            version=manifest["version"].strip(),
            license=manifest["license"].strip(),
            source=manifest["source"].strip(),
            attribution=manifest["attribution"].strip(),
            artifact_path=artifact_path,
            artifact_size_bytes=artifact_size,
            artifact_sha256=artifact_sha256,
            bands=tuple(bands),
            normalization=normalization,
            class_mapping=class_mapping,
        ),
        None,
    )


def list_models(configuration: Settings = settings) -> dict[str, Any]:
    spec, reason = _read_model_spec(configuration)
    ready = spec is not None
    prithvi_weights_present = (
        configuration.ai_model_dir / "Prithvi_EO_V2_tiny_TL.pt"
    ).is_file()
    prithvi_runtime_available = (
        find_spec("torch") is not None and find_spec("terratorch") is not None
    )
    prithvi_blockers = [
        "No local Prithvi checkpoint is installed."
        if not prithvi_weights_present
        else "No Prithvi land-cover inference adapter is configured.",
        "The pretrained checkpoint reconstructs masked HLS image sequences; it has no land-cover segmentation head and requires task-specific fine-tuning.",
        "The checkpoint expects four temporal frames and six HLS channels (blue, green, red, narrow NIR, SWIR 1, SWIR 2); this service processes a single scene and currently acquires B02, B03, B04, and B08 only.",
        "Sentinel-2 native B05-B07 are red-edge bands, not the HLS narrow-NIR/SWIR channels; a wavelength-aware B8A/B11/B12 mapping, 30 m grid alignment, and reflectance normalization would be required.",
    ]
    if not prithvi_runtime_available:
        prithvi_blockers.append(
            "The local PyTorch and TerraTorch runtime is not installed."
        )
    return {
        "models": [
            {
                "id": LANDCOVER_MODEL_ID,
                "name": spec.name if spec else "Local Sentinel-2 land-cover classifier",
                "version": spec.version if spec else None,
                "framework": "scikit-learn",
                "framework_version": sklearn.__version__,
                "license": spec.license if spec else "Not supplied",
                "source": spec.source if spec else None,
                "attribution": spec.attribution if spec else None,
                "status": "ready" if ready else "model_not_ready",
                "readiness_detail": reason,
                "input_requirements": {
                    "format": "georeferenced GeoTIFF",
                    "bands": list(spec.bands if spec else LANDCOVER_BANDS),
                    "normalization": spec.normalization if spec else None,
                    "max_pixels": configuration.max_ai_inference_pixels,
                },
                "output_type": "class-index GeoTIFF and per-pixel maximum-probability GeoTIFF",
                "artifact_size_bytes": spec.artifact_size_bytes if spec else None,
            }
        ],
        "candidates": [
            {
                "id": "prithvi-eo-2.0-tiny-tl",
                "name": "Prithvi-EO-2.0 tiny-TL (pretrained backbone)",
                "version": PRITHVI_TINY_TL_REVISION,
                "framework": "PyTorch / TerraTorch",
                "license": "Apache-2.0 (Hugging Face checkpoint model card)",
                "source": PRITHVI_TINY_TL_REPOSITORY,
                "status": "model_not_ready",
                "readiness_detail": " ".join(prithvi_blockers),
                "checkpoint_present": prithvi_weights_present,
                "runtime_available": prithvi_runtime_available,
                "input_requirements": {
                    "format": "four-frame, 224x224 HLS reflectance sequence",
                    "bands": ["B02", "B03", "B04", "B05", "B06", "B07"],
                    "band_semantics": [
                        "blue",
                        "green",
                        "red",
                        "narrow NIR",
                        "SWIR 1",
                        "SWIR 2",
                    ],
                    "pretraining_resolution_m": 30,
                },
                "output_type": "Masked image reconstruction; not a land-cover prediction",
            }
        ],
    }


def model_weights_sha256(configuration: Settings = settings) -> str:
    spec, reason = _read_model_spec(configuration)
    if spec is None:
        raise ModelNotReadyError(reason or ModelNotReadyError.detail)
    return spec.artifact_sha256


def _load_estimator(
    spec: LandCoverModelSpec,
) -> tuple[Any, list[str | int | float]]:
    try:
        estimator = joblib.load(spec.artifact_path)
    except Exception as exc:
        raise ModelConfigurationError(
            "The trusted local model artifact could not be loaded with the installed scikit-learn environment."
        ) from exc
    if not callable(getattr(estimator, "predict_proba", None)):
        raise ModelConfigurationError(
            "The configured classifier must implement predict_proba for confidence output."
        )
    feature_count = getattr(estimator, "n_features_in_", None)
    if feature_count is not None and feature_count != len(spec.bands):
        raise ModelConfigurationError(
            "The configured classifier feature count does not match the manifest band list."
        )
    raw_classes = getattr(estimator, "classes_", None)
    if raw_classes is None:
        raise ModelConfigurationError("The configured classifier has no fitted classes.")
    classes = np.asarray(raw_classes)
    if classes.ndim != 1 or not 2 <= len(classes) <= MAX_CLASSES:
        raise ModelConfigurationError("The configured classifier has an unsupported class count.")
    normalized_classes: list[str | int | float] = []
    for class_value in classes:
        if isinstance(class_value, np.generic):
            class_value = class_value.item()
        if (
            isinstance(class_value, bool)
            or not isinstance(class_value, (str, int, float))
            or isinstance(class_value, float) and not math.isfinite(class_value)
        ):
            raise ModelConfigurationError("The configured classifier contains invalid class labels.")
        normalized_classes.append(class_value)
    if len(set(normalized_classes)) != len(normalized_classes):
        raise ModelConfigurationError("The configured classifier contains duplicate class labels.")
    if spec.class_mapping:
        if any(str(class_value) not in spec.class_mapping for class_value in normalized_classes):
            raise ModelConfigurationError(
                "The model class mapping does not cover every fitted classifier class."
            )
        normalized_classes = [
            spec.class_mapping[str(class_value)] for class_value in normalized_classes
        ]
    return estimator, normalized_classes


def _band_indexes(dataset: rasterio.DatasetReader, bands: tuple[str, ...]) -> dict[str, int]:
    found: dict[str, list[int]] = {band: [] for band in bands}
    for index, description in enumerate(dataset.descriptions, start=1):
        tags = dataset.tags(index)
        code = identify_band(
            " ".join(
                value
                for value in (description or "", tags.get("NAME", ""), tags.get("BAND", ""))
                if value
            )
        )
        if code in found:
            found[code].append(index)
    if any(len(found[band]) != 1 for band in bands):
        raise InferenceInputError(
            "The model requires exactly one source band labelled B02, B03, B04, and B08."
        )
    return {band: found[band][0] for band in bands}


def _raster_metadata(path: Path, *, artifact_name: str, nodata: float | int) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        return {
            "artifact_name": artifact_name,
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "transform": list(dataset.transform)[:6],
            "bounds": list(dataset.bounds),
            "nodata": nodata,
        }


def run_landcover_inference(
    source_path: Path,
    classification_path: Path,
    confidence_path: Path,
    *,
    configuration: Settings = settings,
    deadline_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    spec, reason = _read_model_spec(configuration)
    if spec is None:
        raise ModelNotReadyError(reason or ModelNotReadyError.detail)
    output_paths = (classification_path, confidence_path)
    if (
        classification_path == confidence_path
        or source_path.resolve() in {path.resolve() for path in output_paths}
        or any(path.exists() or path.is_symlink() for path in output_paths)
    ):
        raise InferenceInputError("Inference output paths must be distinct new files.")
    estimator, class_labels = _load_estimator(spec)
    started = time.perf_counter()
    output_files_created = False
    try:
        source_size = source_path.stat().st_size
        if source_size <= 0 or source_size > min(
            configuration.max_raster_bytes,
            configuration.max_artifact_bytes,
        ):
            raise InferenceInputError("The source GeoTIFF is empty or exceeds the configured byte limit.")
        with rasterio.open(source_path) as source:
            if (
                source.driver != "GTiff"
                or source.width < 1
                or source.height < 1
                or source.width * source.height > configuration.max_ai_inference_pixels
                or source.crs is None
                or source.count < len(LANDCOVER_BANDS)
                or source.count > 64
                or any(
                    not np.issubdtype(np.dtype(dtype), np.number)
                    or np.issubdtype(np.dtype(dtype), np.complexfloating)
                    for dtype in source.dtypes
                )
                or not all(math.isfinite(value) for value in source.transform)
                or not math.isfinite(source.transform.determinant)
                or source.transform.determinant == 0
            ):
                raise InferenceInputError(
                    "The source must be a bounded, georeferenced numeric GeoTIFF with a valid transform."
                )
            band_indexes = _band_indexes(source, spec.bands)
            profile = source.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                compress="deflate",
                BIGTIFF="IF_SAFER",
            )
            class_profile = {**profile, "dtype": "uint16", "nodata": 0, "predictor": 2}
            class_profile.update(tiled=True, blockxsize=256, blockysize=256)
            confidence_profile = {
                **profile,
                "dtype": "float32",
                "nodata": -1.0,
                "predictor": 3,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
            valid_pixels = 0
            confidence_min = 1.0
            confidence_max = 0.0
            confidence_sum = 0.0
            class_counts = [0] * len(class_labels)
            class_lookup = {
                (value.item() if isinstance(value, np.generic) else value): index + 1
                for index, value in enumerate(np.asarray(estimator.classes_))
            }
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                output_files_created = True
                with (
                    rasterio.open(classification_path, "w", **class_profile) as classes_dst,
                    rasterio.open(confidence_path, "w", **confidence_profile) as confidence_dst,
                ):
                    classes_dst.set_band_description(1, "predicted_class_index")
                    confidence_dst.set_band_description(
                        1,
                        "maximum_class_probability",
                    )
                    model_tags = {
                        "MODEL_ID": LANDCOVER_MODEL_ID,
                        "MODEL_NAME": spec.name,
                        "MODEL_VERSION": spec.version,
                        "MODEL_LICENSE": spec.license,
                        "MODEL_SOURCE": spec.source,
                        "MODEL_SHA256": spec.artifact_sha256,
                        "INPUT_BANDS": ",".join(spec.bands),
                        "INPUT_NORMALIZATION": spec.normalization,
                    }
                    classes_dst.update_tags(**model_tags)
                    confidence_dst.update_tags(**model_tags)
                    for row in range(0, source.height, BLOCK_SIZE):
                        for column in range(0, source.width, BLOCK_SIZE):
                            if deadline_check:
                                deadline_check()
                            height = min(BLOCK_SIZE, source.height - row)
                            width = min(BLOCK_SIZE, source.width - column)
                            window = Window.from_slices(
                                (row, row + height),
                                (column, column + width),
                            )
                            arrays = [
                                source.read(band_indexes[band], window=window, masked=True)
                                for band in spec.bands
                            ]
                            valid = np.logical_and.reduce(
                                [
                                    ~np.ma.getmaskarray(array)
                                    & np.isfinite(np.asarray(array.data))
                                    for array in arrays
                                ]
                            )
                            class_block = np.zeros((height, width), dtype=np.uint16)
                            confidence_block = np.full(
                                (height, width),
                                confidence_profile["nodata"],
                                dtype=np.float32,
                            )
                            if np.any(valid):
                                features = np.stack(
                                    [
                                        np.asarray(array.data[valid], dtype=np.float32)
                                        for array in arrays
                                    ],
                                    axis=1,
                                )
                                if spec.normalization == "divide_by_10000":
                                    features /= np.float32(10_000)
                                if not np.isfinite(features).all():
                                    raise InferenceInputError(
                                        "The selected input normalization produced non-finite values."
                                    )
                                try:
                                    probabilities = np.asarray(
                                        estimator.predict_proba(features),
                                        dtype=np.float64,
                                    )
                                except Exception as exc:
                                    raise InferenceExecutionError(
                                        "The configured model failed while predicting a raster block."
                                    ) from exc
                                if (
                                    probabilities.shape != (int(valid.sum()), len(class_labels))
                                    or not np.isfinite(probabilities).all()
                                    or np.any(probabilities < 0)
                                    or np.any(probabilities > 1)
                                    or not np.allclose(
                                        probabilities.sum(axis=1),
                                        1.0,
                                        rtol=1e-3,
                                        atol=1e-3,
                                    )
                                ):
                                    raise InferenceExecutionError()
                                predicted_indexes = probabilities.argmax(axis=1)
                                maximum_probability = probabilities[
                                    np.arange(len(predicted_indexes)),
                                    predicted_indexes,
                                ].astype(np.float32)
                                encoded_classes = np.asarray(
                                    [class_lookup[value] for value in np.asarray(estimator.classes_)[predicted_indexes]],
                                    dtype=np.uint16,
                                )
                                class_block[valid] = encoded_classes
                                confidence_block[valid] = maximum_probability
                                valid_count = len(maximum_probability)
                                valid_pixels += valid_count
                                confidence_min = min(confidence_min, float(maximum_probability.min()))
                                confidence_max = max(confidence_max, float(maximum_probability.max()))
                                confidence_sum += float(maximum_probability.sum(dtype=np.float64))
                                for index in range(len(class_labels)):
                                    class_counts[index] += int(np.count_nonzero(predicted_indexes == index))
                            classes_dst.write(class_block, 1, window=window)
                            confidence_dst.write(confidence_block, 1, window=window)
                            mask = valid.astype(np.uint8) * 255
                            classes_dst.write_mask(mask, window=window)
                            confidence_dst.write_mask(mask, window=window)
                    if valid_pixels == 0:
                        raise InferenceInputError("The source GeoTIFF contains no valid model input pixels.")
    except InferenceError:
        if output_files_created:
            for path in output_paths:
                path.unlink(missing_ok=True)
        raise
    except (RasterioError, OSError, ValueError, TypeError, MemoryError) as exc:
        if output_files_created:
            for path in output_paths:
                path.unlink(missing_ok=True)
        raise InferenceInputError(
            "The source or output raster could not be read or written safely."
        ) from exc
    except Exception:
        if output_files_created:
            for path in output_paths:
                path.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    return {
        "status": "completed",
        "model": {
            "id": LANDCOVER_MODEL_ID,
            "name": spec.name,
            "version": spec.version,
            "framework": "scikit-learn",
            "framework_version": sklearn.__version__,
            "license": spec.license,
            "source": spec.source,
            "attribution": spec.attribution,
            "weights_sha256": spec.artifact_sha256,
        },
        "input_requirements": {
            "bands": list(spec.bands),
            "normalization": spec.normalization,
            "interpretation": (
                "Band samples are read in manifest order; scaling is applied exactly as declared. "
                "No resampling or automatic atmospheric correction is performed."
            ),
        },
        "output_type": "per-pixel land-cover class index and maximum class probability",
        "processing_time_seconds": round(elapsed, 6),
        "valid_pixels": valid_pixels,
        "classes": [
            {
                "class_index": index + 1,
                "label": label,
                "pixel_count": class_counts[index],
                "percentage": class_counts[index] / valid_pixels * 100,
            }
            for index, label in enumerate(class_labels)
        ],
        "confidence": {
            "meaning": "maximum class probability returned by the configured classifier",
            "minimum": confidence_min,
            "maximum": confidence_max,
            "mean": confidence_sum / valid_pixels,
            "valid_pixels": valid_pixels,
        },
        "rasters": {
            "classification": _raster_metadata(
                classification_path,
                artifact_name="classification.tif",
                nodata=0,
            ),
            "confidence": _raster_metadata(
                confidence_path,
                artifact_name="confidence.tif",
                nodata=-1,
            ),
        },
    }
