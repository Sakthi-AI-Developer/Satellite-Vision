from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SceneAnalysisRequest(BaseModel):
    analysis_type: Literal["ndvi", "ndwi", "kmeans"]
    band_mapping: dict[str, int] | None = None
    feature_bands: list[int] | None = None
    cluster_count: int = Field(default=3, ge=2, le=10)
    random_seed: int = Field(default=42, ge=0, le=4_294_967_295)

    @field_validator("feature_bands")
    @classmethod
    def validate_feature_bands(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if not 2 <= len(value) <= 16:
            raise ValueError("Select between 2 and 16 feature bands.")
        if len(set(value)) != len(value):
            raise ValueError("Feature band selections must not contain duplicates.")
        if any(not 1 <= index <= 64 for index in value):
            raise ValueError("Feature band numbers must be between 1 and 64.")
        return value


class SceneChangeDetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_scene_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
    )
    comparison_scene_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
    )
    baseline_band_mapping: dict[str, int] | None = None
    comparison_band_mapping: dict[str, int] | None = None
    threshold: float = Field(default=0.1, gt=0, le=2, allow_inf_nan=False)

    @field_validator("baseline_band_mapping", "comparison_band_mapping")
    @classmethod
    def validate_band_roles(
        cls,
        value: dict[str, int] | None,
    ) -> dict[str, int] | None:
        if value is None:
            return value
        if set(value) != {"red", "nir"}:
            raise ValueError("Each scene band mapping must select exactly red and nir.")
        if any(not 1 <= index <= 64 for index in value.values()):
            raise ValueError("Selected source band numbers must be between 1 and 64.")
        if len(set(value.values())) != len(value):
            raise ValueError("Red and near-infrared must use different source bands.")
        return value

    @model_validator(mode="after")
    def validate_scene_ids(self) -> "SceneChangeDetectionRequest":
        if self.baseline_scene_id == self.comparison_scene_id:
            raise ValueError("Baseline and comparison must be different scenes.")
        return self


class SceneAIInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: Literal["sentinel2-landcover-sklearn"]
