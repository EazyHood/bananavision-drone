from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

from .geo import GeoTransform
from .models import Detection, PredictionResult


def assign_ids(detections: list[Detection], prefix: str = "banana") -> list[Detection]:
    for index, detection in enumerate(detections, start=1):
        detection.id = detection.id or f"{prefix}-{index:05d}"
    return detections


def attach_geo(detections: list[Detection], transform: GeoTransform | None) -> list[Detection]:
    if transform is None:
        return detections
    for detection in detections:
        detection.geo = transform.pixel_to_world(*detection.center)
    return detections


def write_json(result: PredictionResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def write_csv(detections: list[Detection], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "label",
        "score",
        "center_x",
        "center_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "area_px",
        "geo_x",
        "geo_y",
        "crs",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for detection in detections:
            geo = detection.geo
            writer.writerow(
                {
                    "id": detection.id,
                    "label": detection.label,
                    "score": f"{detection.score:.6f}",
                    "center_x": f"{detection.center[0]:.3f}",
                    "center_y": f"{detection.center[1]:.3f}",
                    "bbox_x1": f"{detection.bbox[0]:.3f}",
                    "bbox_y1": f"{detection.bbox[1]:.3f}",
                    "bbox_x2": f"{detection.bbox[2]:.3f}",
                    "bbox_y2": f"{detection.bbox[3]:.3f}",
                    "area_px": f"{detection.area_px:.3f}",
                    "geo_x": "" if geo is None else f"{geo.x:.8f}",
                    "geo_y": "" if geo is None else f"{geo.y:.8f}",
                    "crs": "" if geo is None or geo.crs is None else geo.crs,
                    "source": detection.source,
                }
            )
    return path


def write_geojson(detections: list[Detection], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for detection in detections:
        if detection.geo is None:
            x, y = detection.center
            crs = "pixel"
        else:
            x, y = detection.geo.x, detection.geo.y
            crs = detection.geo.crs or "unknown"
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {
                "id": detection.id,
                "label": detection.label,
                "score": detection.score,
                "bbox": detection.bbox,
                "area_px": detection.area_px,
                "source": detection.source,
                "crs": crs,
            },
        }
        features.append(feature)
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_kml(detections: list[Detection], path: str | Path, document_name: str = "BananaVision detections") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    placemarks = []
    skipped = 0
    for detection in detections:
        if detection.geo is None or not _looks_like_lonlat(detection.geo.x, detection.geo.y):
            skipped += 1
            continue
        placemarks.append(_kml_placemark(detection))
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{escape(document_name)}</name>
    <description>BananaVision detections. Skipped non-lon/lat detections: {skipped}</description>
    <Style id="banana-point">
      <IconStyle>
        <color>ff0ad6ff</color>
        <scale>0.8</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
    </Style>
{''.join(placemarks)}
  </Document>
</kml>
"""
    path.write_text(content, encoding="utf-8")
    return path


def _kml_placemark(detection: Detection) -> str:
    assert detection.geo is not None
    name = escape(str(detection.id or detection.label))
    description = escape(
        f"label={detection.label}; score={detection.score:.4f}; source={detection.source}; area_px={detection.area_px:.2f}"
    )
    return f"""    <Placemark>
      <name>{name}</name>
      <styleUrl>#banana-point</styleUrl>
      <description>{description}</description>
      <ExtendedData>
        <Data name="label"><value>{escape(detection.label)}</value></Data>
        <Data name="score"><value>{detection.score:.6f}</value></Data>
        <Data name="source"><value>{escape(detection.source)}</value></Data>
      </ExtendedData>
      <Point><coordinates>{detection.geo.x:.8f},{detection.geo.y:.8f},0</coordinates></Point>
    </Placemark>
"""


def _looks_like_lonlat(x: float, y: float) -> bool:
    return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0


def draw_overlay(image_path: str | Path, detections: list[Detection], path: str | Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cx, cy = detection.center
        draw.rectangle((x1, y1, x2, y2), outline=(255, 214, 10, 255), width=3)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 56, 56, 240))
        label = f"{detection.id or ''} {detection.score:.2f}".strip()
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = max(0, min(int(x1), image.width - text_width - 2))
        label_y = max(0, min(int(y1 + 2), image.height - text_height - 2))
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 2, label_y + text_height + 2),
            fill=(255, 214, 10, 210),
        )
        draw.text((label_x + 1, label_y + 1), label, fill=(5, 18, 25, 255), font=font)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def write_bundle(result: PredictionResult, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result.image_path.stem
    paths = {
        "json": write_json(result, output_dir / f"{stem}.detections.json"),
        "csv": write_csv(result.detections, output_dir / f"{stem}.detections.csv"),
        "geojson": write_geojson(result.detections, output_dir / f"{stem}.detections.geojson"),
        "kml": write_kml(result.detections, output_dir / f"{stem}.detections.kml", document_name=f"{stem} detections"),
    }
    if result.config.output_overlay:
        paths["overlay"] = draw_overlay(result.image_path, result.detections, output_dir / f"{stem}.overlay.jpg")
    return paths
