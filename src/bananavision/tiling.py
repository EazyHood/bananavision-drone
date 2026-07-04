from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .pipeline import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class TilingSummary:
    source_images: int
    tiles_written: int
    labels_written: int
    objects_kept: int
    objects_dropped: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tile_yolo_dataset(
    input_root: str | Path,
    output_root: str | Path,
    split: str = "train",
    tile_size: int = 1024,
    overlap: int = 128,
    min_polygon_area_px: float = 64.0,
    include_empty: bool = False,
) -> TilingSummary:
    input_root = Path(input_root)
    output_root = Path(output_root)
    image_dir = input_root / "images" / split
    label_dir = input_root / "labels" / split
    output_image_dir = output_root / "images" / split
    output_label_dir = output_root / "labels" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)

    tiles_written = 0
    labels_written = 0
    kept = 0
    dropped = 0
    for image_path in image_paths:
        label_path = _label_path_for_image(image_path, image_dir, label_dir)
        labels = _read_yolo_labels(label_path) if label_path.exists() else []
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            abs_labels = [_label_to_abs(label, width, height) for label in labels]
            for tile in _tiles(width, height, tile_size, overlap):
                x1, y1, x2, y2 = tile
                tile_labels: list[str] = []
                for class_id, polygon in abs_labels:
                    clipped = clip_polygon_to_rect(polygon, tile)
                    if len(clipped) < 3:
                        dropped += 1
                        continue
                    area = polygon_area(clipped)
                    if area < min_polygon_area_px:
                        dropped += 1
                        continue
                    tile_labels.append(_polygon_to_yolo_line(class_id, clipped, tile))
                    kept += 1
                if not tile_labels and not include_empty:
                    continue
                stem = f"{image_path.stem}_x{x1}_y{y1}"
                image.crop(tile).save(output_image_dir / f"{stem}.jpg", quality=92)
                (output_label_dir / f"{stem}.txt").write_text(
                    "\n".join(tile_labels) + ("\n" if tile_labels else ""),
                    encoding="utf-8",
                )
                tiles_written += 1
                labels_written += 1

    summary = TilingSummary(
        source_images=len(image_paths),
        tiles_written=tiles_written,
        labels_written=labels_written,
        objects_kept=kept,
        objects_dropped=dropped,
        output_dir=str(output_root),
    )
    (output_root / "tiling_summary.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    _write_data_yaml(input_root, output_root, split)
    return summary


def clip_polygon_to_rect(
    polygon: list[tuple[float, float]],
    rect: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    x_min, y_min, x_max, y_max = rect
    clipped = polygon
    clipped = _clip_edge(clipped, lambda p: p[0] >= x_min, lambda a, b: _intersect_vertical(a, b, x_min))
    clipped = _clip_edge(clipped, lambda p: p[0] <= x_max, lambda a, b: _intersect_vertical(a, b, x_max))
    clipped = _clip_edge(clipped, lambda p: p[1] >= y_min, lambda a, b: _intersect_horizontal(a, b, y_min))
    clipped = _clip_edge(clipped, lambda p: p[1] <= y_max, lambda a, b: _intersect_horizontal(a, b, y_max))
    return clipped


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        total += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(total) / 2.0


def _clip_edge(
    polygon: list[tuple[float, float]],
    inside,
    intersect,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def _intersect_vertical(
    a: tuple[float, float],
    b: tuple[float, float],
    x: float,
) -> tuple[float, float]:
    if abs(b[0] - a[0]) < 1e-9:
        return (x, a[1])
    t = (x - a[0]) / (b[0] - a[0])
    return (x, a[1] + t * (b[1] - a[1]))


def _intersect_horizontal(
    a: tuple[float, float],
    b: tuple[float, float],
    y: float,
) -> tuple[float, float]:
    if abs(b[1] - a[1]) < 1e-9:
        return (a[0], y)
    t = (y - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), y)


def _tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and smaller than tile_size")
    stride = tile_size - overlap
    starts_x = _starts(width, tile_size, stride)
    starts_y = _starts(height, tile_size, stride)
    return [(x, y, min(width, x + tile_size), min(height, y + tile_size)) for y in starts_y for x in starts_x]


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _read_yolo_labels(path: Path) -> list[list[float]]:
    labels: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        labels.append([float(part) for part in parts])
    return labels


def _label_to_abs(label: list[float], width: int, height: int) -> tuple[int, list[tuple[float, float]]]:
    class_id = int(label[0])
    values = label[1:]
    if len(values) == 4:
        cx, cy, w, h = values
        x1 = (cx - w / 2.0) * width
        y1 = (cy - h / 2.0) * height
        x2 = (cx + w / 2.0) * width
        y2 = (cy + h / 2.0) * height
        return class_id, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    coords = values[:-1] if len(values) % 2 else values
    polygon = [(coords[index] * width, coords[index + 1] * height) for index in range(0, len(coords), 2)]
    return class_id, polygon


def _polygon_to_yolo_line(
    class_id: int,
    polygon: list[tuple[float, float]],
    tile: tuple[int, int, int, int],
) -> str:
    x1, y1, x2, y2 = tile
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    values: list[str] = [str(class_id)]
    for x, y in polygon:
        values.append(f"{_clamp01((x - x1) / width):.6f}")
        values.append(f"{_clamp01((y - y1) / height):.6f}")
    return " ".join(values)


def _label_path_for_image(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    relative = image_path.relative_to(image_dir)
    return (label_dir / relative).with_suffix(".txt")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _write_data_yaml(input_root: Path, output_root: Path, split: str) -> None:
    source_yaml = input_root / "data.yaml"
    names: dict[int, str] = {0: "banana_plant"}
    if source_yaml.exists():
        payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        raw_names = payload.get("names")
        if isinstance(raw_names, dict):
            names = {int(key): str(value) for key, value in raw_names.items()}
        elif isinstance(raw_names, list):
            names = {index: str(value) for index, value in enumerate(raw_names)}
    data_yaml = output_root / "data.yaml"
    existing = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) if data_yaml.exists() else {}
    existing = existing or {}
    existing["path"] = str(output_root.resolve())
    existing[split] = f"images/{split}"
    existing["names"] = names
    data_yaml.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")
