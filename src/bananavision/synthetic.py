from __future__ import annotations

import json
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw


def generate_scene(
    output_image: str | Path,
    output_truth: str | Path | None = None,
    width: int = 960,
    height: int = 640,
    plant_count: int = 36,
    seed: int = 7,
    clustered_mats: int = 0,
    min_plants_per_mat: int = 2,
    max_plants_per_mat: int = 3,
    cluster_spread_px: float = 28.0,
) -> tuple[Path, Path | None]:
    if clustered_mats < 0:
        raise ValueError("clustered_mats must be non-negative")
    if min_plants_per_mat < 1 or max_plants_per_mat < min_plants_per_mat:
        raise ValueError("plant-per-mat bounds are invalid")
    rng = Random(seed)
    image = Image.new("RGB", (width, height), (92, 104, 72))
    draw = ImageDraw.Draw(image, "RGBA")
    centers: list[tuple[float, float]] = []
    annotations: list[dict[str, float | str]] = []
    remaining = plant_count
    for mat_index in range(clustered_mats):
        if remaining < 2:
            break
        mat_size = min(remaining, rng.randint(min_plants_per_mat, max_plants_per_mat))
        if mat_size < 2:
            break
        group_id = f"mat-{mat_index + 1:03d}"
        base_x = rng.uniform(55, width - 55)
        base_y = rng.uniform(55, height - 55)
        for plant_index in range(mat_size):
            angle = (360.0 / mat_size) * plant_index + rng.uniform(-22, 22)
            radius = rng.uniform(cluster_spread_px * 0.35, cluster_spread_px)
            cx = _clamp(base_x + radius * _cos_deg(angle), 40, width - 40)
            cy = _clamp(base_y + radius * _sin_deg(angle), 40, height - 40)
            centers.append((cx, cy))
            annotations.append({"x": round(cx, 3), "y": round(cy, 3), "group_id": group_id})
            _draw_banana_crown(draw, cx, cy, rng)
        remaining -= mat_size
    for _ in range(remaining):
        if centers and rng.random() < 0.35:
            base = rng.choice(centers)
            cx = min(width - 40, max(40, base[0] + rng.uniform(-28, 32)))
            cy = min(height - 40, max(40, base[1] + rng.uniform(-28, 32)))
        else:
            cx = rng.uniform(45, width - 45)
            cy = rng.uniform(45, height - 45)
        centers.append((cx, cy))
        annotations.append({"x": round(cx, 3), "y": round(cy, 3)})
        _draw_banana_crown(draw, cx, cy, rng)
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image)
    truth_path = None
    if output_truth:
        truth_path = Path(output_truth)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"centers": annotations}
        truth_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_image, truth_path


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _cos_deg(angle: float) -> float:
    from math import cos, radians

    return cos(radians(angle))


def _sin_deg(angle: float) -> float:
    from math import radians, sin

    return sin(radians(angle))


def _draw_banana_crown(draw: ImageDraw.ImageDraw, cx: float, cy: float, rng: Random) -> None:
    leaves = rng.randint(6, 10)
    for leaf in range(leaves):
        angle = (360 / leaves) * leaf + rng.uniform(-14, 14)
        length = rng.uniform(20, 38)
        width = rng.uniform(8, 14)
        green = rng.randint(135, 205)
        color = (rng.randint(28, 60), green, rng.randint(40, 76), rng.randint(185, 235))
        draw.regular_polygon(
            bounding_circle=(cx, cy, length),
            n_sides=4,
            rotation=angle,
            fill=color,
        )
        draw.ellipse((cx - width, cy - width, cx + width, cy + width), fill=(35, green, 45, 220))
