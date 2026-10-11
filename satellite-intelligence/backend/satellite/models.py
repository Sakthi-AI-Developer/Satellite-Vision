from __future__ import annotations

from datetime import date
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pyproj import Geod
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity


class GeoJSONPolygon(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any]

    @model_validator(mode="after")
    def validate_geometry(self) -> "GeoJSONPolygon":
        if not isinstance(self.coordinates, list):
            raise ValueError("AOI coordinates must be a GeoJSON coordinate array.")
        if self.type == "Polygon":
            rings = self.coordinates
        elif self.type == "MultiPolygon":
            if any(
                not isinstance(polygon, list)
                or any(not isinstance(ring, list) for ring in polygon)
                for polygon in self.coordinates
            ):
                raise ValueError("AOI MultiPolygon coordinates are malformed.")
            rings = [ring for polygon in self.coordinates for ring in polygon]
        else:
            raise ValueError("AOI geometry type must be Polygon or MultiPolygon.")

        if not rings:
            raise ValueError("AOI geometry must contain at least one ring.")

        for ring in rings:
            if not isinstance(ring, list):
                raise ValueError("AOI polygon rings must be coordinate arrays.")
            if len(ring) < 4:
                raise ValueError("AOI polygon rings must contain at least four coordinates.")
            if ring[0] != ring[-1]:
                raise ValueError("AOI polygon rings must be closed.")
            for point in ring:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("Each AOI coordinate must contain longitude and latitude.")
                lon, lat = point
                if (
                    isinstance(lon, bool)
                    or isinstance(lat, bool)
                    or not isinstance(lon, (int, float))
                    or not isinstance(lat, (int, float))
                    or not math.isfinite(lon)
                    or not math.isfinite(lat)
                    or not (-180 <= lon <= 180 and -90 <= lat <= 90)
                ):
                    raise ValueError("AOI coordinates must fall within valid geographic bounds.")

        try:
            geometry = shape(self.model_dump())
        except (TypeError, ValueError) as exc:
            raise ValueError("AOI coordinates do not form a valid GeoJSON geometry.") from exc
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(
                f"AOI geometry is invalid: {explain_validity(geometry)}."
            )
        bbox = self.bounds
        width = abs(bbox[2] - bbox[0])
        height = abs(bbox[3] - bbox[1])
        if width > 30 or height > 30:
            raise ValueError("AOI extent is too large for a single analysis region.")
        if width <= 0 or height <= 0:
            raise ValueError("AOI bounds are invalid.")
        return self

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        flattened = [point for ring in self._rings for point in ring]
        lons = [point[0] for point in flattened]
        lats = [point[1] for point in flattened]
        return min(lons), min(lats), max(lons), max(lats)

    @property
    def _rings(self) -> list[list[list[float]]]:
        if self.type == "Polygon":
            return self.coordinates if self.coordinates else []
        return [ring for polygon in self.coordinates for ring in polygon]

    def estimate_area_km2(self) -> float:
        geometry = shape(self.model_dump())
        polygons = (
            list(geometry.geoms)
            if isinstance(geometry, MultiPolygon)
            else [geometry]
        )
        geod = Geod(ellps="WGS84")
        area_square_metres = sum(
            abs(geod.geometry_area_perimeter(orient(polygon, sign=1.0))[0])
            for polygon in polygons
            if isinstance(polygon, Polygon)
        )
        return area_square_metres / 1_000_000


class SatelliteSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aoi: GeoJSONPolygon
    start_date: date
    end_date: date
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)
    satellite: str = "Sentinel-2"
    product_type: str = "S2MSI2A"

    @model_validator(mode="after")
    def validate_dates(self) -> "SatelliteSearchRequest":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date.")
        return self


class SatelliteDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(..., min_length=1)
    aoi: GeoJSONPolygon | None = None


class SentinelCatalogSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aoi: GeoJSONPolygon
    start_date: date
    end_date: date
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)
    limit: int = Field(default=20, ge=1, le=50)
    cursor: str | None = Field(default=None, min_length=1, max_length=4096)

    @field_validator("aoi", mode="before")
    @classmethod
    def validate_aoi_coordinate_count(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        coordinates = value.get("coordinates")
        if not isinstance(coordinates, list):
            return value
        if value.get("type") == "Polygon":
            rings = coordinates
        elif value.get("type") == "MultiPolygon":
            rings = [
                ring
                for polygon in coordinates
                if isinstance(polygon, list)
                for ring in polygon
                if isinstance(ring, list)
            ]
        else:
            return value
        if sum(len(ring) for ring in rings if isinstance(ring, list)) > 500:
            raise ValueError("AOI geometry cannot contain more than 500 coordinates.")
        return value

    @model_validator(mode="after")
    def validate_search(self) -> "SentinelCatalogSearchRequest":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date.")
        coordinate_count = sum(
            len(ring)
            for ring in self.aoi._rings
            if isinstance(ring, list)
        )
        if coordinate_count > 500:
            raise ValueError("AOI geometry cannot contain more than 500 coordinates.")
        if self.aoi.estimate_area_km2() > 1_000_000:
            raise ValueError("AOI geometry is too large for catalog discovery.")
        return self


class SentinelSceneAcquisitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["Copernicus Data Space Ecosystem"]
    collection: Literal["sentinel-2-l2a"]
    aoi: GeoJSONPolygon
    scene_id: str = Field(
        ...,
        min_length=60,
        max_length=80,
        pattern=r"^S2[ABC]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T[0-9A-Z]{5}_\d{8}T\d{6}$",
    )


class SatelliteProduct(BaseModel):
    product_id: str
    acquisition_date: str | None = None
    cloud_cover: float | None = None
    platform: str
    processing_level: str | None = None
    spatial_coverage: str | None = None
    product_size_mb: float | None = None
    available_bands: list[str] = Field(default_factory=list)
    provider: str = "live"
    available: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["available"] = bool(self.available)
        return payload
