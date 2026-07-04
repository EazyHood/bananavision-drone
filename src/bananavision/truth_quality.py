from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import iter_images
from .runtime import utc_now_iso
from .truth import TruthPoint, read_truth_points, truth_points_from_payload


@dataclass(frozen=True)
class TruthQualityThresholds:
    min_center_distance_px: float = 2.0
    max_group_size: int = 6
    allow_singleton_groups: bool = False


def audit_truth_quality(
    truth_path: str | Path,
    output_json: str | Path,
    image_path: str | Path | None = None,
    thresholds: TruthQualityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or TruthQualityThresholds()
    _validate_thresholds(thresholds)
    truth_path = Path(truth_path)
    image_path_value = None if image_path is None else Path(image_path)
    image_lookup = _image_lookup(image_path_value)
    entries = _truth_entries(truth_path, image_path_value)
    image_reports = [_image_report(entry, image_lookup, thresholds) for entry in entries]
    issues = [issue for report in image_reports for issue in report["issues"]]
    report = {
        "created_at": utc_now_iso(),
        "truth_path": str(truth_path),
        "image_path": None if image_path_value is None else str(image_path_value),
        "status": "pass" if entries and not issues else "fail",
        "thresholds": asdict(thresholds),
        "image_count": len(image_reports),
        "truth_count": sum(report["truth_count"] for report in image_reports),
        "bounded_image_count": sum(1 for report in image_reports if report["width"] is not None and report["height"] is not None),
        "issue_count": len(issues),
        "checks": [
            _threshold_check("truth_images_present", bool(entries), f"image_count={len(entries)}"),
            _threshold_check("truth_quality_issues", not issues, f"issue_count={len(issues)}"),
        ],
        "issues": issues,
        "images": image_reports,
    }
    write_truth_quality_report(report, output_json)
    return report


def write_truth_quality_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_truth_quality_csv(report, output_json.with_suffix(".csv"))
    return output_json


def _truth_entries(truth_path: Path, image_path: Path | None) -> list[dict[str, Any]]:
    if image_path is not None:
        entries = []
        for image in iter_images(image_path):
            truth_source = _truth_source_for_image(truth_path, image)
            entries.append({"image": str(image), "truth_source": str(truth_source), "points": read_truth_points(truth_source, image=image)})
        return entries
    if truth_path.is_dir():
        return [
            {"image": str(path.name), "truth_source": str(path), "points": read_truth_points(path)}
            for path in sorted(truth_path.rglob("*.json"))
            if path.name.endswith(".truth.json") or not path.name.startswith(".")
        ]
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "images" in payload:
        entries = []
        for index, item in enumerate(payload.get("images", [])):
            image = str(item.get("image", f"manifest-entry-{index}"))
            entries.append(
                {
                    "image": image,
                    "truth_source": str(truth_path),
                    "points": truth_points_from_payload(item),
                    "width": _optional_int(item.get("width")),
                    "height": _optional_int(item.get("height")),
                }
            )
        return entries
    return [{"image": truth_path.name, "truth_source": str(truth_path), "points": truth_points_from_payload(payload)}]


def _image_report(
    entry: dict[str, Any],
    image_lookup: dict[str, Path],
    thresholds: TruthQualityThresholds,
) -> dict[str, Any]:
    image = str(entry["image"])
    points = list(entry["points"])
    width, height = _image_dimensions(entry, image_lookup)
    issues: list[dict[str, Any]] = []
    issues.extend(_duplicate_issues(image, points, thresholds.min_center_distance_px))
    issues.extend(_group_issues(image, points, thresholds))
    if width is not None and height is not None:
        issues.extend(_bounds_issues(image, points, width, height))
    return {
        "image": image,
        "truth_source": str(entry.get("truth_source", "")),
        "width": width,
        "height": height,
        "truth_count": len(points),
        "group_count": len({point.group_id for point in points if point.group_id is not None}),
        "issue_count": len(issues),
        "issues": issues,
    }


def _duplicate_issues(image: str, points: list[TruthPoint], min_distance: float) -> list[dict[str, Any]]:
    issues = []
    for left_index, left in enumerate(points):
        for right_index, right in enumerate(points[left_index + 1 :], start=left_index + 1):
            distance = _distance(left, right)
            if distance < min_distance:
                issues.append(
                    {
                        "image": image,
                        "type": "duplicate_or_too_close_center",
                        "left_index": left_index,
                        "right_index": right_index,
                        "distance_px": round(distance, 6),
                        "minimum_px": min_distance,
                    }
                )
    return issues


def _group_issues(
    image: str,
    points: list[TruthPoint],
    thresholds: TruthQualityThresholds,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        if point.group_id is not None:
            groups[point.group_id].append(index)
    issues = []
    for group_id, indexes in sorted(groups.items()):
        if len(indexes) == 1 and not thresholds.allow_singleton_groups:
            issues.append(
                {
                    "image": image,
                    "type": "singleton_group",
                    "group_id": group_id,
                    "indexes": indexes,
                }
            )
        if len(indexes) > thresholds.max_group_size:
            issues.append(
                {
                    "image": image,
                    "type": "oversized_group",
                    "group_id": group_id,
                    "size": len(indexes),
                    "maximum": thresholds.max_group_size,
                    "indexes": indexes,
                }
            )
    return issues


def _bounds_issues(image: str, points: list[TruthPoint], width: int, height: int) -> list[dict[str, Any]]:
    issues = []
    for index, point in enumerate(points):
        if point.x < 0 or point.y < 0 or point.x >= width or point.y >= height:
            issues.append(
                {
                    "image": image,
                    "type": "center_out_of_bounds",
                    "index": index,
                    "x": point.x,
                    "y": point.y,
                    "width": width,
                    "height": height,
                }
            )
    return issues


def _image_lookup(image_path: Path | None) -> dict[str, Path]:
    if image_path is None:
        return {}
    images = iter_images(image_path)
    lookup = {}
    for image in images:
        lookup[str(image)] = image
        lookup[image.name] = image
    return lookup


def _image_dimensions(entry: dict[str, Any], image_lookup: dict[str, Path]) -> tuple[int | None, int | None]:
    width = _optional_int(entry.get("width"))
    height = _optional_int(entry.get("height"))
    if width is not None and height is not None:
        return width, height
    image = image_lookup.get(str(entry.get("image"))) or image_lookup.get(Path(str(entry.get("image"))).name)
    if image is None:
        return None, None
    with Image.open(image) as handle:
        return handle.size


def _truth_source_for_image(truth_path: Path, image: Path) -> Path:
    if truth_path.is_file():
        return truth_path
    candidates = [
        truth_path / f"{image.stem}.truth.json",
        truth_path / f"{image.stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No truth file found for {image.name} in {truth_path}")


def _write_truth_quality_csv(report: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "type", "group_id", "index", "left_index", "right_index", "detail"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for issue in report.get("issues", []):
            writer.writerow(
                {
                    "image": issue.get("image", ""),
                    "type": issue.get("type", ""),
                    "group_id": issue.get("group_id", ""),
                    "index": issue.get("index", ""),
                    "left_index": issue.get("left_index", ""),
                    "right_index": issue.get("right_index", ""),
                    "detail": json.dumps(issue, sort_keys=True),
                }
            )


def _distance(left: TruthPoint, right: TruthPoint) -> float:
    return sqrt((left.x - right.x) ** 2 + (left.y - right.y) ** 2)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_thresholds(thresholds: TruthQualityThresholds) -> None:
    if thresholds.min_center_distance_px < 0:
        raise ValueError("min_center_distance_px must be non-negative")
    if thresholds.max_group_size < 2:
        raise ValueError("max_group_size must be >= 2")


def _threshold_check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}
