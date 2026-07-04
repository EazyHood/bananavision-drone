from __future__ import annotations

import csv
import json
import re
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from .runtime import utc_now_iso


def update_inventory(
    detections_geojson: str | Path,
    inventory_dir: str | Path,
    distance_threshold: float = 1.2,
    id_prefix: str = "banana-plant",
    observed_at: str | None = None,
) -> dict[str, Any]:
    detections = read_detection_features(detections_geojson)
    inventory_dir = Path(inventory_dir)
    inventory_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = inventory_dir / "inventory.json"
    inventory = _load_inventory(inventory_path)
    observed_at = observed_at or utc_now_iso()

    created = 0
    updated = 0
    for detection in detections:
        match = _find_match(inventory["plants"], detection, distance_threshold)
        if match is None:
            created += 1
            inventory["plants"].append(_new_plant(detection, len(inventory["plants"]) + 1, id_prefix, observed_at))
        else:
            updated += 1
            _update_plant(match, detection, observed_at)

    inventory["updated_at"] = utc_now_iso()
    inventory["plant_count"] = len(inventory["plants"])
    _write_json(inventory_path, inventory)
    snapshot_path = write_inventory_snapshot(inventory, inventory_dir, observed_at)
    csv_path = write_inventory_csv(inventory, inventory_dir / "inventory.csv")
    geojson_path = write_inventory_geojson(inventory, inventory_dir / "inventory.geojson")
    kml_path = write_inventory_kml(inventory, inventory_dir / "inventory.kml")
    return {
        "detections": len(detections),
        "created": created,
        "updated": updated,
        "plant_count": len(inventory["plants"]),
        "inventory_json": str(inventory_path),
        "inventory_snapshot": str(snapshot_path),
        "inventory_csv": str(csv_path),
        "inventory_geojson": str(geojson_path),
        "inventory_kml": str(kml_path),
    }


def diff_inventories(
    before_inventory: str | Path,
    after_inventory: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    before_path = Path(before_inventory)
    after_path = Path(after_inventory)
    if not before_path.exists():
        raise FileNotFoundError(f"Before inventory not found: {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"After inventory not found: {after_path}")
    before = _load_inventory(before_path)
    after = _load_inventory(after_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_by_id = {plant["plant_id"]: plant for plant in before.get("plants", [])}
    after_by_id = {plant["plant_id"]: plant for plant in after.get("plants", [])}
    before_ids = set(before_by_id)
    after_ids = set(after_by_id)
    new_ids = sorted(after_ids - before_ids)
    missing_ids = sorted(before_ids - after_ids)
    persistent_ids = sorted(before_ids & after_ids)
    report = {
        "before": str(before_path),
        "after": str(after_path),
        "before_count": len(before_ids),
        "after_count": len(after_ids),
        "new_count": len(new_ids),
        "missing_count": len(missing_ids),
        "persistent_count": len(persistent_ids),
        "new_plants": [after_by_id[plant_id] for plant_id in new_ids],
        "missing_plants": [before_by_id[plant_id] for plant_id in missing_ids],
        "persistent_plants": [_persistent_summary(before_by_id[plant_id], after_by_id[plant_id]) for plant_id in persistent_ids],
    }
    _write_json(output_dir / "inventory_diff.json", report)
    write_inventory_geojson({"plants": report["new_plants"]}, output_dir / "inventory_new.geojson")
    write_inventory_geojson({"plants": report["missing_plants"]}, output_dir / "inventory_missing.geojson")
    write_inventory_kml({"plants": report["new_plants"]}, output_dir / "inventory_new.kml")
    write_inventory_kml({"plants": report["missing_plants"]}, output_dir / "inventory_missing.kml")
    return report


def read_detection_features(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry", {}) or {}
        if geometry.get("type") != "Point":
            continue
        coords = geometry.get("coordinates", [])
        if len(coords) < 2:
            continue
        properties = feature.get("properties", {}) or {}
        features.append(
            {
                "x": float(coords[0]),
                "y": float(coords[1]),
                "crs": str(properties.get("crs", "unknown")),
                "score": float(properties.get("score", 0.0)),
                "source_detection_id": properties.get("id"),
                "label": properties.get("label", "banana_plant"),
                "source": properties.get("source", "unknown"),
            }
        )
    return features


def write_inventory_csv(inventory: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "plant_id",
                "x",
                "y",
                "crs",
                "observations",
                "first_seen",
                "last_seen",
                "score_mean",
                "last_detection_id",
            ],
        )
        writer.writeheader()
        for plant in inventory["plants"]:
            writer.writerow({key: plant.get(key, "") for key in writer.fieldnames or []})
    return path


def write_inventory_snapshot(inventory: dict[str, Any], inventory_dir: str | Path, observed_at: str) -> Path:
    snapshot_dir = Path(inventory_dir) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_snapshot_path(snapshot_dir, observed_at)
    _write_json(path, inventory)
    return path


def write_inventory_geojson(inventory: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    features = []
    for plant in inventory["plants"]:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [plant["x"], plant["y"]]},
                "properties": {key: value for key, value in plant.items() if key not in {"x", "y", "history"}},
            }
        )
    payload = {"type": "FeatureCollection", "features": features}
    _write_json(path, payload)
    return path


def write_inventory_kml(inventory: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    placemarks = []
    skipped = 0
    for plant in inventory["plants"]:
        if not _is_lonlat_crs(str(plant.get("crs", ""))) or not _looks_like_lonlat(float(plant["x"]), float(plant["y"])):
            skipped += 1
            continue
        placemarks.append(_plant_placemark(plant))
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>BananaVision plant inventory</name>
    <description>Skipped non-lon/lat plants: {skipped}</description>
{''.join(placemarks)}
  </Document>
</kml>
"""
    path.write_text(content, encoding="utf-8")
    return path


def _load_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "created_at": utc_now_iso(), "updated_at": None, "plant_count": 0, "plants": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _new_plant(detection: dict[str, Any], index: int, id_prefix: str, observed_at: str) -> dict[str, Any]:
    return {
        "plant_id": f"{id_prefix}-{index:06d}",
        "x": detection["x"],
        "y": detection["y"],
        "crs": detection["crs"],
        "observations": 1,
        "first_seen": observed_at,
        "last_seen": observed_at,
        "score_mean": detection["score"],
        "last_detection_id": detection.get("source_detection_id"),
        "label": detection.get("label", "banana_plant"),
        "history": [_history_item(detection, observed_at)],
    }


def _update_plant(plant: dict[str, Any], detection: dict[str, Any], observed_at: str) -> None:
    observations = int(plant.get("observations", 0)) + 1
    plant["x"] = ((float(plant["x"]) * (observations - 1)) + detection["x"]) / observations
    plant["y"] = ((float(plant["y"]) * (observations - 1)) + detection["y"]) / observations
    plant["observations"] = observations
    plant["last_seen"] = observed_at
    plant["last_detection_id"] = detection.get("source_detection_id")
    previous_score = float(plant.get("score_mean", 0.0))
    plant["score_mean"] = ((previous_score * (observations - 1)) + detection["score"]) / observations
    plant.setdefault("history", []).append(_history_item(detection, observed_at))


def _history_item(detection: dict[str, Any], observed_at: str) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "x": detection["x"],
        "y": detection["y"],
        "score": detection["score"],
        "source_detection_id": detection.get("source_detection_id"),
    }


def _find_match(
    plants: list[dict[str, Any]],
    detection: dict[str, Any],
    distance_threshold: float,
) -> dict[str, Any] | None:
    best = None
    best_distance = float("inf")
    for plant in plants:
        if str(plant.get("crs")) != str(detection.get("crs")):
            continue
        distance = _distance(
            float(plant["x"]),
            float(plant["y"]),
            detection["x"],
            detection["y"],
            str(plant.get("crs", "")),
        )
        if distance < best_distance:
            best = plant
            best_distance = distance
    return best if best is not None and best_distance <= distance_threshold else None


def _distance(x1: float, y1: float, x2: float, y2: float, crs: str) -> float:
    if _is_lonlat_crs(crs) and _looks_like_lonlat(x1, y1) and _looks_like_lonlat(x2, y2):
        return _haversine_m(x1, y1, x2, y2)
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _looks_like_lonlat(x: float, y: float) -> bool:
    return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0


def _is_lonlat_crs(crs: str) -> bool:
    normalized = crs.upper()
    return normalized in {"EPSG:4326", "WGS84", "WGS 84", "LONLAT", "LONGITUDE_LATITUDE"}


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_000.0
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * asin(sqrt(a))


def _plant_placemark(plant: dict[str, Any]) -> str:
    name = escape(str(plant["plant_id"]))
    description = escape(f"observations={plant['observations']}; score_mean={float(plant['score_mean']):.4f}")
    return f"""    <Placemark>
      <name>{name}</name>
      <description>{description}</description>
      <Point><coordinates>{float(plant['x']):.8f},{float(plant['y']):.8f},0</coordinates></Point>
    </Placemark>
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _persistent_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "plant_id": after["plant_id"],
        "before_observations": before.get("observations", 0),
        "after_observations": after.get("observations", 0),
        "before_last_seen": before.get("last_seen"),
        "after_last_seen": after.get("last_seen"),
    }


def _safe_timestamp(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "-", value).strip("-") or "snapshot"


def _unique_snapshot_path(snapshot_dir: Path, observed_at: str) -> Path:
    stem = f"inventory_{_safe_timestamp(observed_at)}"
    path = snapshot_dir / f"{stem}.json"
    index = 2
    while path.exists():
        path = snapshot_dir / f"{stem}_{index}.json"
        index += 1
    return path
