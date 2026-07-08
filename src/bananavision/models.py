from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GeoPoint:
    x: float
    y: float
    crs: str | None = None


@dataclass
class InferenceConfig:
    detector: str = "rgb-canopy"
    model_path: str | None = None
    ensemble_model_paths: list[str] | None = None
    ensemble_weights: list[float] | None = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    gsd_cm: float = 2.0
    expected_crown_diameter_m: float = 2.4
    min_component_area_px: int = 80
    max_component_area_px: int | None = None
    split_mat_clusters: bool = True
    min_center_distance_ratio: float = 0.42
    center_distance_weight: float = 0.35
    canopy_fill_ratio: float = 0.58
    max_split_instances: int = 12
    tile_size: int = 1024
    tile_overlap: int = 128
    class_name: str = "banana_plant"
    rgb_threshold_quantile: float = 0.82
    output_overlay: bool = True
    mark_crown_centers: bool = True
    mission_geo_dedupe_distance_m: float = 1.2
    mission_pixel_dedupe_distance_px: float = 32.0

    @property
    def gsd_m(self) -> float:
        return self.gsd_cm / 100.0

    @property
    def expected_crown_diameter_px(self) -> float:
        if self.gsd_m <= 0:
            raise ValueError("gsd_cm must be positive")
        return max(4.0, self.expected_crown_diameter_m / self.gsd_m)

    def validate(self) -> None:
        if self.detector not in {"rgb-canopy", "yolo-seg", "yolo-ensemble"}:
            raise ValueError("detector must be 'rgb-canopy', 'yolo-seg' or 'yolo-ensemble'")
        if self.detector == "yolo-seg" and not self.model_path:
            raise ValueError("model_path is required when detector='yolo-seg'")
        if self.detector == "yolo-ensemble" and (
            not self.ensemble_model_paths or len(self.ensemble_model_paths) < 2
        ):
            raise ValueError(
                "ensemble_model_paths (at least 2) is required when detector='yolo-ensemble'"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if self.gsd_cm <= 0:
            raise ValueError("gsd_cm must be positive")
        if self.expected_crown_diameter_m <= 0:
            raise ValueError("expected_crown_diameter_m must be positive")
        if self.min_component_area_px < 1:
            raise ValueError("min_component_area_px must be at least 1")
        if self.max_split_instances < 1:
            raise ValueError("max_split_instances must be at least 1")
        if not 0.0 <= self.center_distance_weight <= 1.0:
            raise ValueError("center_distance_weight must be between 0 and 1")
        if self.tile_size < 64:
            raise ValueError("tile_size must be at least 64")
        if self.tile_overlap < 0 or self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be >= 0 and smaller than tile_size")
        if not 0.0 < self.rgb_threshold_quantile < 1.0:
            raise ValueError("rgb_threshold_quantile must be between 0 and 1")
        if self.mission_geo_dedupe_distance_m < 0:
            raise ValueError("mission_geo_dedupe_distance_m must be non-negative")
        if self.mission_pixel_dedupe_distance_px < 0:
            raise ValueError("mission_pixel_dedupe_distance_px must be non-negative")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> InferenceConfig:
        accepted = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        config = cls(**{key: value for key, value in values.items() if key in accepted})
        config.validate()
        return config


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    area_px: float
    source: str
    id: str | None = None
    geo: GeoPoint | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "score": round(float(self.score), 6),
            "bbox": [round(float(v), 3) for v in self.bbox],
            "center": [round(float(v), 3) for v in self.center],
            "area_px": round(float(self.area_px), 3),
            "source": self.source,
            "meta": self.meta,
        }
        if self.geo:
            payload["geo"] = {"x": self.geo.x, "y": self.geo.y, "crs": self.geo.crs}
        return payload


@dataclass
class PredictionResult:
    image_path: Path
    width: int
    height: int
    detections: list[Detection]
    config: InferenceConfig
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "meta": self.meta,
            "detections": [detection.to_dict() for detection in self.detections],
        }
