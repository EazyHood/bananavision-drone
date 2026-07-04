from __future__ import annotations

from collections import deque
from math import hypot, pi, sqrt

import numpy as np

from .models import Detection, InferenceConfig


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def nms(
    detections: list[Detection],
    iou_threshold: float,
    min_center_distance_px: float | None = None,
) -> list[Detection]:
    ordered = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: list[Detection] = []
    for detection in ordered:
        if all(
            not _suppresses_detection(
                detection,
                existing,
                iou_threshold=iou_threshold,
                min_center_distance_px=min_center_distance_px,
            )
            for existing in kept
        ):
            kept.append(detection)
    return kept


def _suppresses_detection(
    candidate: Detection,
    existing: Detection,
    iou_threshold: float,
    min_center_distance_px: float | None,
) -> bool:
    if min_center_distance_px is None:
        return bbox_iou(candidate.bbox, existing.bbox) > iou_threshold
    distance = hypot(candidate.center[0] - existing.center[0], candidate.center[1] - existing.center[1])
    return distance <= min_center_distance_px


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    try:
        import cv2  # type: ignore

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cleaned.astype(bool)
    except Exception:
        return mask


def connected_components(mask: np.ndarray, min_area: int = 1) -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    mask = mask.astype(bool)
    try:
        import cv2  # type: ignore

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        components = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if int(area) >= min_area:
                local = labels[y : y + h, x : x + w] == label
                components.append((local, (int(x), int(y), int(x + w), int(y + h)), int(area)))
        return components
    except Exception:
        return _connected_components_python(mask, min_area)


def _connected_components_python(
    mask: np.ndarray, min_area: int
) -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                px, py = queue.popleft()
                pixels.append((px, py))
                for nx in (px - 1, px, px + 1):
                    for ny in (py - 1, py, py + 1):
                        if nx == px and ny == py:
                            continue
                        if 0 <= nx < width and 0 <= ny < height and not seen[ny, nx] and mask[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((nx, ny))
            if len(pixels) < min_area:
                continue
            xs = [pixel[0] for pixel in pixels]
            ys = [pixel[1] for pixel in pixels]
            x1, x2 = min(xs), max(xs) + 1
            y1, y2 = min(ys), max(ys) + 1
            local = np.zeros((y2 - y1, x2 - x1), dtype=bool)
            for px, py in pixels:
                local[py - y1, px - x1] = True
            components.append((local, (x1, y1, x2, y2), len(pixels)))
    return components


def estimate_instance_count(
    area_px: int,
    config: InferenceConfig,
    bbox: tuple[int, int, int, int] | None = None,
) -> int:
    diameter = config.expected_crown_diameter_px
    expected_area = pi * (diameter / 2.0) ** 2 * config.canopy_fill_ratio
    if expected_area <= 1:
        return 1
    area_count = area_px / expected_area
    extent_count = _extent_instance_count(bbox, diameter) if bbox is not None else 1.0
    return int(np.clip(round(max(area_count, extent_count)), 1, config.max_split_instances))


def _extent_instance_count(bbox: tuple[int, int, int, int], diameter: float) -> float:
    x1, y1, x2, y2 = bbox
    major_extent = float(max(x2 - x1, y2 - y1))
    if diameter <= 1 or major_extent <= diameter:
        return 1.0
    spacing = diameter * 0.75
    return 1.0 + ((major_extent - diameter) / max(1.0, spacing))


def peak_candidates(
    component_mask: np.ndarray,
    component_score: np.ndarray,
    count: int,
    min_distance: float,
    distance_weight: float = 0.35,
) -> list[tuple[float, float, float]]:
    ys, xs = np.where(component_mask)
    if len(xs) == 0:
        return []
    values = component_score[ys, xs]
    priority = _peak_priority_values(component_mask, component_score, ys, xs, distance_weight)
    cutoff = np.quantile(priority, 0.62 if count <= 2 else 0.52)
    order = np.argsort(priority)[::-1]
    selected: list[tuple[float, float, float]] = []
    min_distance_sq = min_distance * min_distance
    for index in order[: min(len(order), 20000)]:
        if priority[index] < cutoff and len(selected) >= count:
            break
        x = float(xs[index])
        y = float(ys[index])
        score = float(values[index])
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_distance_sq for sx, sy, _ in selected):
            selected.append((x, y, score))
            if len(selected) >= count:
                break
    if len(selected) < count:
        selected = _fill_with_farthest_points(component_mask, selected, count, min_distance)
    return selected[:count]


def mask_center_distance_score(component_mask: np.ndarray) -> np.ndarray:
    component_mask = component_mask.astype(bool)
    if not component_mask.any():
        return np.zeros(component_mask.shape, dtype=np.float32)
    padded = np.pad(component_mask, 1, constant_values=False)
    try:
        import cv2  # type: ignore

        distance = cv2.distanceTransform(padded.astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    except Exception:
        distance = _chamfer_distance_to_background(padded)[1:-1, 1:-1]
    max_distance = float(distance.max()) if distance.size else 0.0
    if max_distance <= 0:
        return np.zeros(component_mask.shape, dtype=np.float32)
    return (distance / max_distance).astype(np.float32)


def _peak_priority_values(
    component_mask: np.ndarray,
    component_score: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    distance_weight: float,
) -> np.ndarray:
    score_values = component_score[ys, xs].astype(float)
    score_norm = _normalize_values(score_values)
    distance_values = mask_center_distance_score(component_mask)[ys, xs].astype(float)
    distance_norm = _normalize_values(distance_values)
    weight = float(np.clip(distance_weight, 0.0, 1.0))
    if float(np.ptp(score_values)) <= 1e-9:
        weight = 1.0
    return ((1.0 - weight) * score_norm) + (weight * distance_norm)


def _normalize_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    minimum = float(values.min())
    span = float(values.max() - minimum)
    if span <= 1e-9:
        return np.zeros(values.shape, dtype=float)
    return (values - minimum) / span


def _chamfer_distance_to_background(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    inf = float(height + width + 1)
    distance = np.where(mask, inf, 0.0).astype(float)
    diagonal = sqrt(2.0)
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            best = distance[y, x]
            if x > 0:
                best = min(best, distance[y, x - 1] + 1.0)
            if y > 0:
                best = min(best, distance[y - 1, x] + 1.0)
            if x > 0 and y > 0:
                best = min(best, distance[y - 1, x - 1] + diagonal)
            if x + 1 < width and y > 0:
                best = min(best, distance[y - 1, x + 1] + diagonal)
            distance[y, x] = best
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if not mask[y, x]:
                continue
            best = distance[y, x]
            if x + 1 < width:
                best = min(best, distance[y, x + 1] + 1.0)
            if y + 1 < height:
                best = min(best, distance[y + 1, x] + 1.0)
            if x + 1 < width and y + 1 < height:
                best = min(best, distance[y + 1, x + 1] + diagonal)
            if x > 0 and y + 1 < height:
                best = min(best, distance[y + 1, x - 1] + diagonal)
            distance[y, x] = best
    return distance


def _fill_with_farthest_points(
    component_mask: np.ndarray,
    selected: list[tuple[float, float, float]],
    count: int,
    min_distance: float,
) -> list[tuple[float, float, float]]:
    ys, xs = np.where(component_mask)
    if len(xs) == 0:
        return selected
    if not selected:
        selected.append((float(xs.mean()), float(ys.mean()), 0.5))
    min_distance_sq = (min_distance * 0.7) ** 2
    while len(selected) < count:
        best: tuple[float, float, float] | None = None
        best_dist = -1.0
        sample_xs = xs[:: max(1, len(xs) // 4000)]
        sample_ys = ys[:: max(1, len(ys) // 4000)]
        for x, y in zip(sample_xs, sample_ys, strict=False):
            dist = min((float(x) - sx) ** 2 + (float(y) - sy) ** 2 for sx, sy, _ in selected)
            if dist > best_dist and dist >= min_distance_sq:
                best_dist = dist
                best = (float(x), float(y), 0.45)
        if best is None:
            break
        selected.append(best)
    return selected


def split_instances_from_mask(
    mask: np.ndarray,
    score: np.ndarray,
    config: InferenceConfig,
    source: str,
    base_score: float = 1.0,
) -> list[Detection]:
    mask = clean_mask(mask)
    components = connected_components(mask, config.min_component_area_px)
    detections: list[Detection] = []
    radius = config.expected_crown_diameter_px / 2.0
    min_distance = config.expected_crown_diameter_px * config.min_center_distance_ratio
    height, width = mask.shape
    for local_mask, bbox, area in components:
        if config.max_component_area_px and area > config.max_component_area_px:
            continue
        x1, y1, x2, y2 = bbox
        local_score = score[y1:y2, x1:x2]
        count = estimate_instance_count(area, config, bbox) if config.split_mat_clusters else 1
        extent_count = _extent_instance_count(bbox, config.expected_crown_diameter_px)
        candidates = peak_candidates(
            local_mask,
            local_score,
            count,
            min_distance,
            distance_weight=config.center_distance_weight,
        )
        if not candidates:
            continue
        for local_x, local_y, peak_score in candidates:
            cx = x1 + local_x
            cy = y1 + local_y
            det_bbox = (
                max(0.0, cx - radius),
                max(0.0, cy - radius),
                min(float(width), cx + radius),
                min(float(height), cy + radius),
            )
            confidence = float(np.clip(base_score * (0.55 + 0.45 * peak_score), 0.0, 1.0))
            detections.append(
                Detection(
                    label=config.class_name,
                    score=confidence,
                    bbox=det_bbox,
                    center=(float(cx), float(cy)),
                    area_px=float(area / max(1, count)),
                    source=source,
                    meta={
                        "component_area_px": area,
                        "component_split_count": count,
                        "component_extent_count": round(extent_count, 3),
                        "center_distance_weight": config.center_distance_weight,
                    },
                )
            )
    return nms(detections, config.iou_threshold, min_center_distance_px=min_distance * 0.65)
