from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConversionSummary:
    images: int
    objects: int
    labels_written: int
    skipped_objects: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def convert_coco_to_yolo_seg(
    coco_json: str | Path,
    output_labels: str | Path,
    target_names: set[str] | None = None,
    output_class_id: int = 0,
) -> ConversionSummary:
    payload = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    output_labels = Path(output_labels)
    output_labels.mkdir(parents=True, exist_ok=True)

    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    images = {int(item["id"]): item for item in payload.get("images", [])}
    grouped: dict[int, list[str]] = {image_id: [] for image_id in images}
    objects = 0
    skipped = 0

    for annotation in payload.get("annotations", []):
        image_id = int(annotation["image_id"])
        image = images.get(image_id)
        if image is None:
            skipped += 1
            continue
        category_name = categories.get(int(annotation.get("category_id", -1)), "")
        if target_names and category_name not in target_names:
            continue
        width = int(image["width"])
        height = int(image["height"])
        polygon = _coco_annotation_polygon(annotation, width, height)
        if polygon is None:
            skipped += 1
            continue
        grouped.setdefault(image_id, []).append(_yolo_seg_line(output_class_id, polygon, width, height))
        objects += 1

    written = 0
    for image_id, lines in grouped.items():
        image = images[image_id]
        label_path = output_labels / f"{Path(str(image['file_name'])).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    return ConversionSummary(
        images=len(images),
        objects=objects,
        labels_written=written,
        skipped_objects=skipped,
        output_dir=str(output_labels),
    )


def convert_labelme_to_yolo_seg(
    labelme_dir: str | Path,
    output_labels: str | Path,
    target_label: str = "banana_plant",
    output_class_id: int = 0,
) -> ConversionSummary:
    labelme_dir = Path(labelme_dir)
    output_labels = Path(output_labels)
    output_labels.mkdir(parents=True, exist_ok=True)
    json_paths = sorted(labelme_dir.rglob("*.json"))
    objects = 0
    skipped = 0

    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        width = int(payload.get("imageWidth") or 0)
        height = int(payload.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            skipped += len(payload.get("shapes", []))
            continue
        lines: list[str] = []
        for shape in payload.get("shapes", []):
            if str(shape.get("label")) != target_label:
                continue
            polygon = _labelme_shape_polygon(shape)
            if polygon is None:
                skipped += 1
                continue
            lines.append(_yolo_seg_line(output_class_id, polygon, width, height))
            objects += 1
        image_name = payload.get("imagePath") or json_path.with_suffix(".jpg").name
        label_path = output_labels / f"{Path(str(image_name)).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return ConversionSummary(
        images=len(json_paths),
        objects=objects,
        labels_written=len(json_paths),
        skipped_objects=skipped,
        output_dir=str(output_labels),
    )


def _coco_annotation_polygon(
    annotation: dict[str, Any],
    width: int,
    height: int,
) -> list[tuple[float, float]] | None:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list) and segmentation:
        polygons = [segment for segment in segmentation if isinstance(segment, list) and len(segment) >= 6]
        if polygons:
            segment = max(polygons, key=len)
            return _pairs(segment)
    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        x, y, w, h = [float(value) for value in bbox[:4]]
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return None


def _labelme_shape_polygon(shape: dict[str, Any]) -> list[tuple[float, float]] | None:
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        return None
    shape_type = str(shape.get("shape_type") or "polygon")
    if shape_type == "rectangle" and len(points) >= 2:
        (x1, y1), (x2, y2) = points[:2]
        return [(float(x1), float(y1)), (float(x2), float(y1)), (float(x2), float(y2)), (float(x1), float(y2))]
    if len(points) >= 3:
        return [(float(x), float(y)) for x, y in points]
    return None


def _yolo_seg_line(
    class_id: int,
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> str:
    coords: list[str] = []
    for x, y in polygon:
        coords.append(f"{_clamp01(x / width):.6f}")
        coords.append(f"{_clamp01(y / height):.6f}")
    return " ".join([str(class_id), *coords])


def _pairs(values: list[Any]) -> list[tuple[float, float]]:
    return [(float(values[index]), float(values[index + 1])) for index in range(0, len(values) - 1, 2)]


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
