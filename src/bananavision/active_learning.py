from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ReviewItem:
    image: str
    reason: str
    priority: float
    detection_id: str | None = None
    score: float | None = None
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_review_queue(
    predictions_dir: str | Path,
    output_json: str | Path,
    low_confidence: float = 0.45,
    high_split_count: int = 2,
) -> list[ReviewItem]:
    predictions_dir = Path(predictions_dir)
    items: list[ReviewItem] = []
    for path in sorted(predictions_dir.rglob("*.detections.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        image = str(payload.get("image") or path.name)
        detections = payload.get("detections", [])
        if not detections:
            items.append(ReviewItem(image=image, reason="zero_detections", priority=1.0))
            continue
        for detection in detections:
            score = float(detection.get("score", 0.0))
            meta = detection.get("meta", {}) or {}
            split_count = int(meta.get("component_split_count", 1))
            if score <= low_confidence:
                items.append(
                    ReviewItem(
                        image=image,
                        reason="low_confidence",
                        priority=round(1.0 - score, 6),
                        detection_id=detection.get("id"),
                        score=score,
                        bbox=detection.get("bbox"),
                    )
                )
            if split_count >= high_split_count:
                items.append(
                    ReviewItem(
                        image=image,
                        reason="cluster_split",
                        priority=round(min(1.0, split_count / 8.0), 6),
                        detection_id=detection.get("id"),
                        score=score,
                        bbox=detection.get("bbox"),
                    )
                )
    items = sorted(items, key=lambda item: item.priority, reverse=True)
    write_review_queue(items, output_json)
    write_review_queue_csv(items, Path(output_json).with_suffix(".csv"))
    return items


def write_review_queue(items: list[ReviewItem], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"items": [item.to_dict() for item in items], "count": len(items)}
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_json


def write_review_queue_csv(items: list[ReviewItem], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "reason", "priority", "detection_id", "score", "bbox"],
        )
        writer.writeheader()
        for item in items:
            row = item.to_dict()
            row["bbox"] = "" if item.bbox is None else json.dumps(item.bbox)
            writer.writerow(row)
    return output_csv


def export_review_crops(
    queue_json: str | Path,
    output_dir: str | Path,
    margin_px: int = 32,
    max_items: int | None = None,
) -> dict[str, Any]:
    queue_json = Path(queue_json)
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(queue_json.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if max_items is not None:
        items = items[:max_items]

    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    image_cache: dict[str, Image.Image] = {}
    for index, item in enumerate(items, start=1):
        image_path = _resolve_image_path(item.get("image", ""), queue_json.parent)
        bbox = item.get("bbox")
        if image_path is None or not bbox:
            skipped.append({"item": item, "reason": "missing_image_or_bbox"})
            continue
        try:
            image = image_cache.get(str(image_path))
            if image is None:
                image = Image.open(image_path).convert("RGB")
                image_cache[str(image_path)] = image
            crop_box = _crop_box(bbox, image.size, margin_px)
            crop = image.crop(crop_box)
            crop_name = f"{index:05d}_{_safe_stem(image_path.stem)}_{item.get('reason', 'review')}.jpg"
            crop_path = crops_dir / crop_name
            crop.save(crop_path, quality=92)
            exported.append(
                {
                    "crop": str(crop_path),
                    "source_image": str(image_path),
                    "crop_box": list(crop_box),
                    "reason": item.get("reason"),
                    "priority": item.get("priority"),
                    "detection_id": item.get("detection_id"),
                    "score": item.get("score"),
                }
            )
        except Exception as exc:
            skipped.append({"item": item, "reason": str(exc)})

    for image in image_cache.values():
        image.close()
    manifest = {
        "queue": str(queue_json),
        "output_dir": str(output_dir),
        "margin_px": margin_px,
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "items": exported,
        "skipped": skipped,
    }
    manifest_path = output_dir / "review_crops_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _resolve_image_path(image: str, base_dir: Path) -> Path | None:
    if not image:
        return None
    path = Path(image)
    if path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return None


def _crop_box(
    bbox: list[float],
    size: tuple[int, int],
    margin_px: int,
) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return (
        max(0, int(x1) - margin_px),
        max(0, int(y1) - margin_px),
        min(width, int(x2) + margin_px),
        min(height, int(y2) + margin_px),
    )


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)[:80]
