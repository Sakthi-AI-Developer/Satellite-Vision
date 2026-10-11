import ipaddress
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def _default_monitor_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (base / "satellite-vision" / "monitoring").expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    api_host: str
    api_port: int
    cors_origins: list[str]
    satellite_provider: str
    copernicus_client_id: str | None
    copernicus_client_secret: str | None
    satellite_cache_dir: Path
    satellite_timeout: int
    satellite_max_cloud_cover: int
    database_url: str | None
    supabase_url: str | None
    supabase_service_role_key: str | None
    supabase_anon_key: str | None
    require_auth: bool
    supabase_storage_bucket: str
    cdse_username: str | None = None
    cdse_password: str | None = None
    max_satellite_acquisition_area_km2: float = 1_000.0
    max_upload_bytes: int = 512 * 1024 * 1024
    max_raster_bytes: int = 512 * 1024 * 1024
    max_artifact_bytes: int = 512 * 1024 * 1024
    max_raster_pixels: int = 1_000_000
    max_imagery_pixels: int = 200_000_000
    max_input_array_bytes: int = 128 * 1024 * 1024
    max_analysis_seconds: int = 300
    max_concurrent_analyses: int = 1
    ai_model_dir: Path = Path("data/models")
    ai_landcover_manifest_path: Path = Path(
        "data/models/sentinel2-landcover.json"
    )
    max_ai_model_bytes: int = 256 * 1024 * 1024
    max_ai_inference_pixels: int = 10_000_000
    database_pool_size: int = 3
    database_max_overflow: int = 1
    satellite_monitor_enabled: bool = False
    satellite_monitor_timezone: str = "Asia/Kolkata"
    satellite_monitor_schedule_time: str = "03:00"
    satellite_monitor_aoi_json: str | None = None
    satellite_monitor_lookback_days: int = 14
    satellite_monitor_max_cloud_cover: float = 20.0
    satellite_monitor_max_scenes: int = 3
    satellite_monitor_max_retries: int = 2
    satellite_monitor_max_scene_bytes: int = 512 * 1024 * 1024
    satellite_monitor_max_storage_bytes: int = 5 * 1024 * 1024 * 1024
    satellite_monitor_data_dir: Path = field(default_factory=_default_monitor_data_dir)

    @property
    def authentication_required(self) -> bool:
        return bool(
            self.require_auth
            or self.database_url
            or self.supabase_url
            or self.supabase_anon_key
            or self.supabase_service_role_key
        )


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:4173,http://127.0.0.1:4173,"
    "https://team-s4-ten.vercel.app"
)
configured_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        DEFAULT_CORS_ORIGINS,
    ).split(",")
    if origin.strip()
]


def _boolean_setting(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, or no.")


def is_valid_supabase_url(value: str | None) -> bool:
    if not value or value != value.strip():
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if (
        not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port == 0
    ):
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname.lower() == "localhost"
    return loopback


settings = Settings(
    data_dir=Path(os.getenv("DATA_DIR", BACKEND_DIR / "data")).expanduser().resolve(),
    output_dir=Path(
        os.getenv("OUTPUT_DIR", BACKEND_DIR / "data" / "outputs")
    ).expanduser().resolve(),
    api_host=os.getenv("API_HOST", "127.0.0.1"),
    api_port=int(os.getenv("API_PORT", "8000")),
    cors_origins=configured_cors_origins,
    satellite_provider=os.getenv("SATELLITE_PROVIDER", "local").strip().lower() or "local",
    copernicus_client_id=os.getenv("COPERNICUS_CLIENT_ID") or None,
    copernicus_client_secret=os.getenv("COPERNICUS_CLIENT_SECRET") or None,
    cdse_username=os.getenv("CDSE_USERNAME") or None,
    cdse_password=os.getenv("CDSE_PASSWORD") or None,
    max_satellite_acquisition_area_km2=float(
        os.getenv("MAX_SATELLITE_ACQUISITION_AREA_KM2", "1000")
    ),
    satellite_cache_dir=Path(os.getenv("SATELLITE_CACHE_DIR", BACKEND_DIR / "data" / "live")).expanduser().resolve(),
    satellite_timeout=int(os.getenv("SATELLITE_TIMEOUT", "30")),
    satellite_max_cloud_cover=int(os.getenv("SATELLITE_MAX_CLOUD_COVER", "20")),
    database_url=os.getenv("DATABASE_URL") or None,
    supabase_url=os.getenv("SUPABASE_URL") or None,
    supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or None,
    supabase_anon_key=os.getenv("SUPABASE_ANON_KEY") or None,
    require_auth=_boolean_setting("REQUIRE_AUTH", default=False),
    supabase_storage_bucket=os.getenv(
        "SUPABASE_STORAGE_BUCKET", "satellite-analysis-results"
    ).strip(),
    max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024))),
    max_raster_bytes=int(os.getenv("MAX_RASTER_BYTES", str(512 * 1024 * 1024))),
    max_artifact_bytes=int(os.getenv("MAX_ARTIFACT_BYTES", str(512 * 1024 * 1024))),
    max_raster_pixels=int(os.getenv("MAX_RASTER_PIXELS", "1000000")),
    max_imagery_pixels=int(os.getenv("MAX_IMAGERY_PIXELS", "200000000")),
    max_input_array_bytes=int(os.getenv("MAX_INPUT_ARRAY_BYTES", str(128 * 1024 * 1024))),
    max_analysis_seconds=int(os.getenv("MAX_ANALYSIS_SECONDS", "300")),
    max_concurrent_analyses=int(os.getenv("MAX_CONCURRENT_ANALYSES", "1")),
    ai_model_dir=Path(
        os.getenv("AI_MODEL_DIR", BACKEND_DIR / "data" / "models")
    ).expanduser().resolve(),
    ai_landcover_manifest_path=Path(
        os.getenv(
            "AI_LANDCOVER_MODEL_MANIFEST",
            BACKEND_DIR / "data" / "models" / "sentinel2-landcover.json",
        )
    ).expanduser().resolve(),
    max_ai_model_bytes=int(
        os.getenv("MAX_AI_MODEL_BYTES", str(256 * 1024 * 1024))
    ),
    max_ai_inference_pixels=int(
        os.getenv("MAX_AI_INFERENCE_PIXELS", "10000000")
    ),
    database_pool_size=int(os.getenv("DATABASE_POOL_SIZE", "3")),
    database_max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "1")),
    satellite_monitor_enabled=_boolean_setting(
        "SATELLITE_MONITOR_ENABLED", default=False
    ),
    satellite_monitor_timezone=os.getenv(
        "SATELLITE_MONITOR_TIMEZONE", "Asia/Kolkata"
    ).strip(),
    satellite_monitor_schedule_time=os.getenv(
        "SATELLITE_MONITOR_SCHEDULE_TIME", "03:00"
    ).strip(),
    satellite_monitor_aoi_json=os.getenv("SATELLITE_MONITOR_AOI") or None,
    satellite_monitor_lookback_days=int(
        os.getenv("SATELLITE_MONITOR_LOOKBACK_DAYS", "14")
    ),
    satellite_monitor_max_cloud_cover=float(
        os.getenv("SATELLITE_MONITOR_MAX_CLOUD_COVER", "20")
    ),
    satellite_monitor_max_scenes=int(
        os.getenv("SATELLITE_MONITOR_MAX_SCENES", "3")
    ),
    satellite_monitor_max_retries=int(
        os.getenv("SATELLITE_MONITOR_MAX_RETRIES", "2")
    ),
    satellite_monitor_max_scene_bytes=int(
        os.getenv("SATELLITE_MONITOR_MAX_SCENE_BYTES", str(512 * 1024 * 1024))
    ),
    satellite_monitor_max_storage_bytes=int(
        os.getenv(
            "SATELLITE_MONITOR_MAX_STORAGE_BYTES",
            str(5 * 1024 * 1024 * 1024),
        )
    ),
    satellite_monitor_data_dir=Path(
        os.getenv("SATELLITE_MONITOR_DATA_DIR")
        or _default_monitor_data_dir()
    ).expanduser().resolve(),
)


def validate_runtime_settings() -> list[str]:
    issues: list[str] = []
    if settings.api_port <= 0 or settings.api_port > 65535:
        issues.append("API_PORT must be an integer between 1 and 65535.")
    if settings.satellite_provider not in {"local", "live", "mock"}:
        issues.append("SATELLITE_PROVIDER must be one of: local, live, or mock.")
    if settings.satellite_timeout <= 0:
        issues.append("SATELLITE_TIMEOUT must be positive.")
    if settings.satellite_max_cloud_cover < 0 or settings.satellite_max_cloud_cover > 100:
        issues.append("SATELLITE_MAX_CLOUD_COVER must be between 0 and 100.")
    if settings.satellite_monitor_enabled:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(settings.satellite_monitor_timezone)
        except (ZoneInfoNotFoundError, ValueError):
            issues.append("SATELLITE_MONITOR_TIMEZONE must be a valid IANA timezone.")
        if not re.fullmatch(
            r"(?:[01]\d|2[0-3]):[0-5]\d",
            settings.satellite_monitor_schedule_time,
        ):
            issues.append("SATELLITE_MONITOR_SCHEDULE_TIME must use HH:MM format.")
        if not settings.database_url:
            issues.append("DATABASE_URL is required when satellite monitoring is enabled.")
        if not (settings.cdse_username and settings.cdse_password):
            issues.append(
                "CDSE_USERNAME and CDSE_PASSWORD are required when satellite monitoring is enabled."
            )
        if not settings.satellite_monitor_aoi_json:
            issues.append("SATELLITE_MONITOR_AOI must contain a GeoJSON polygon.")
        if not 1 <= settings.satellite_monitor_lookback_days <= 90:
            issues.append("SATELLITE_MONITOR_LOOKBACK_DAYS must be between 1 and 90.")
        if (
            not math.isfinite(settings.satellite_monitor_max_cloud_cover)
            or not 0 <= settings.satellite_monitor_max_cloud_cover <= 100
        ):
            issues.append(
                "SATELLITE_MONITOR_MAX_CLOUD_COVER must be between 0 and 100."
            )
        if not 1 <= settings.satellite_monitor_max_scenes <= 10:
            issues.append("SATELLITE_MONITOR_MAX_SCENES must be between 1 and 10.")
        if not 0 <= settings.satellite_monitor_max_retries <= 5:
            issues.append("SATELLITE_MONITOR_MAX_RETRIES must be between 0 and 5.")
        if (
            settings.satellite_monitor_max_scene_bytes <= 0
            or settings.satellite_monitor_max_storage_bytes
            < settings.satellite_monitor_max_scene_bytes
        ):
            issues.append(
                "Monitor storage limits must be positive and total storage must cover one maximum-sized scene."
            )
        if settings.satellite_monitor_data_dir.is_relative_to(BACKEND_DIR.parent):
            issues.append(
                "SATELLITE_MONITOR_DATA_DIR must be outside the project source tree."
            )
    if (
        not math.isfinite(settings.max_satellite_acquisition_area_km2)
        or settings.max_satellite_acquisition_area_km2 <= 0
        or settings.max_satellite_acquisition_area_km2 > 10_000
    ):
        issues.append(
            "MAX_SATELLITE_ACQUISITION_AREA_KM2 must be greater than 0 and at most 10000."
        )
    if settings.max_upload_bytes <= 0:
        issues.append("MAX_UPLOAD_BYTES must be positive.")
    if settings.max_raster_bytes <= 0:
        issues.append("MAX_RASTER_BYTES must be positive.")
    if settings.max_artifact_bytes <= 0:
        issues.append("MAX_ARTIFACT_BYTES must be positive.")
    if settings.max_raster_pixels <= 0:
        issues.append("MAX_RASTER_PIXELS must be positive.")
    if settings.max_imagery_pixels <= 0:
        issues.append("MAX_IMAGERY_PIXELS must be positive.")
    if settings.max_input_array_bytes <= 0:
        issues.append("MAX_INPUT_ARRAY_BYTES must be positive.")
    if settings.max_analysis_seconds < 1 or settings.max_analysis_seconds > 3600:
        issues.append("MAX_ANALYSIS_SECONDS must be between 1 and 3600.")
    if settings.max_concurrent_analyses < 1 or settings.max_concurrent_analyses > 8:
        issues.append("MAX_CONCURRENT_ANALYSES must be between 1 and 8.")
    if settings.max_ai_model_bytes <= 0:
        issues.append("MAX_AI_MODEL_BYTES must be positive.")
    if settings.max_ai_inference_pixels <= 0:
        issues.append("MAX_AI_INFERENCE_PIXELS must be positive.")
    if settings.database_pool_size < 1 or settings.database_pool_size > 10:
        issues.append("DATABASE_POOL_SIZE must be between 1 and 10.")
    if settings.database_max_overflow < 0 or settings.database_max_overflow > 10:
        issues.append("DATABASE_MAX_OVERFLOW must be between 0 and 10.")
    if settings.database_pool_size + settings.database_max_overflow > 10:
        issues.append(
            "DATABASE_POOL_SIZE plus DATABASE_MAX_OVERFLOW must not exceed 10 connections per process."
        )
    if not settings.supabase_storage_bucket:
        issues.append("SUPABASE_STORAGE_BUCKET must not be empty.")
    if settings.supabase_url and not is_valid_supabase_url(settings.supabase_url):
        issues.append(
            "SUPABASE_URL must be a valid HTTPS URL; plain HTTP is allowed only for loopback development."
        )
    if settings.authentication_required and (
        not settings.supabase_url or not settings.supabase_anon_key
    ):
        issues.append(
            "Authentication requires both SUPABASE_URL and SUPABASE_ANON_KEY."
        )
    if settings.authentication_required and not settings.database_url:
        issues.append("Authentication requires DATABASE_URL for private analysis persistence.")
    if settings.authentication_required and not settings.supabase_service_role_key:
        issues.append(
            "Authentication requires SUPABASE_SERVICE_ROLE_KEY for private artifact storage."
        )
    for path in (settings.data_dir, settings.output_dir, settings.satellite_cache_dir):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            issues.append(f"Unable to create directory {path}: {exc}")
    return issues


for directory in (settings.data_dir / "current", settings.data_dir / "historical", settings.output_dir, settings.satellite_cache_dir):
    directory.mkdir(parents=True, exist_ok=True)
