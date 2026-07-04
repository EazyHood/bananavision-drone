from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TruthPoint:
    x: float
    y: float
    group_id: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        return self.x, self.y


def read_truth_centers(path: str | Path, image: str | Path | None = None) -> list[tuple[float, float]]:
    return [point.center for point in read_truth_points(path, image=image)]


def read_truth_points(path: str | Path, image: str | Path | None = None) -> list[TruthPoint]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = _select_payload(payload, None if image is None else Path(image).name)
    return truth_points_from_payload(selected)


def truth_points_from_payload(payload: Any) -> list[TruthPoint]:
    selected = payload
    centers = selected.get("centers", selected if isinstance(selected, list) else [])
    return [_truth_point(center) for center in centers]


def truth_cluster_summary(points: list[TruthPoint]) -> dict[str, int]:
    groups: dict[str, int] = {}
    for point in points:
        if point.group_id is None:
            continue
        groups[point.group_id] = groups.get(point.group_id, 0) + 1
    cluster_sizes = [size for size in groups.values() if size >= 2]
    return {
        "cluster_count": len(cluster_sizes),
        "cluster_truth_count": sum(cluster_sizes),
    }


def _select_payload(payload: Any, image_name: str | None) -> Any:
    if image_name is None:
        return payload
    if isinstance(payload, dict) and "images" in payload:
        for item in payload["images"]:
            if Path(str(item.get("image", ""))).name == image_name:
                return item
    if isinstance(payload, dict) and image_name in payload:
        return payload[image_name]
    return payload


def _truth_point(center: Any) -> TruthPoint:
    if isinstance(center, dict):
        return TruthPoint(float(center["x"]), float(center["y"]), _group_id(center))
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        group_id = None if len(center) < 3 or center[2] is None else str(center[2])
        return TruthPoint(float(center[0]), float(center[1]), group_id)
    raise ValueError(f"Invalid center annotation: {center!r}")


def _group_id(center: dict[str, Any]) -> str | None:
    for key in ["group_id", "group", "cluster_id", "cluster", "mat_id", "mat"]:
        value = center.get(key)
        if value is not None:
            return str(value)
    return None
