from __future__ import annotations

from copy import copy
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

from .io import write_csv, write_geojson, write_kml
from .models import Detection, PredictionResult


def flatten_detections(results: list[PredictionResult]) -> list[Detection]:
    flattened: list[Detection] = []
    for result in results:
        for detection in result.detections:
            item = copy(detection)
            item.meta = dict(detection.meta)
            item.meta["image"] = str(result.image_path)
            flattened.append(item)
    return flattened


def deduplicate_detections(
    detections: list[Detection],
    geo_distance_m: float = 1.2,
    pixel_distance_px: float = 32.0,
) -> list[Detection]:
    ordered = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: list[Detection] = []
    for detection in ordered:
        if all(
            _detection_distance(detection, existing) > _distance_limit(detection, existing, geo_distance_m, pixel_distance_px)
            for existing in kept
        ):
            kept.append(detection)
    for index, detection in enumerate(kept, start=1):
        detection.id = f"mission-banana-{index:05d}"
    return kept


def write_mission_outputs(
    results: list[PredictionResult],
    output_dir: str | Path,
    geo_distance_m: float = 1.2,
    pixel_distance_px: float = 32.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    raw = flatten_detections(results)
    deduped = deduplicate_detections(raw, geo_distance_m=geo_distance_m, pixel_distance_px=pixel_distance_px)
    csv_path = write_csv(deduped, output_dir / "mission.detections.csv")
    geojson_path = write_geojson(deduped, output_dir / "mission.detections.geojson")
    kml_path = write_kml(deduped, output_dir / "mission.detections.kml", document_name="BananaVision mission detections")
    return {
        "raw_detection_count": len(raw),
        "deduplicated_count": len(deduped),
        "duplicate_count": len(raw) - len(deduped),
        "geo_distance_m": geo_distance_m,
        "pixel_distance_px": pixel_distance_px,
        "csv": str(csv_path),
        "geojson": str(geojson_path),
        "kml": str(kml_path),
    }


def _distance_limit(
    first: Detection,
    second: Detection,
    geo_distance_m: float,
    pixel_distance_px: float,
) -> float:
    return geo_distance_m if first.geo is not None and second.geo is not None else pixel_distance_px


def _detection_distance(first: Detection, second: Detection) -> float:
    if first.geo is not None and second.geo is not None:
        if _looks_like_lonlat(first.geo.x, first.geo.y) and _looks_like_lonlat(second.geo.x, second.geo.y):
            return _haversine_m(first.geo.x, first.geo.y, second.geo.x, second.geo.y)
        return sqrt((first.geo.x - second.geo.x) ** 2 + (first.geo.y - second.geo.y) ** 2)
    return sqrt((first.center[0] - second.center[0]) ** 2 + (first.center[1] - second.center[1]) ** 2)


def _looks_like_lonlat(x: float, y: float) -> bool:
    return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_000.0
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * asin(sqrt(a))
