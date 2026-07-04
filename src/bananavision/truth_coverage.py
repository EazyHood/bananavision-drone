from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .pipeline import iter_images
from .runtime import utc_now_iso
from .truth import TruthPoint, read_truth_points, truth_points_from_payload


def audit_truth_coverage(
    truth_path: str | Path,
    image_path: str | Path | None = None,
    min_truth_count: int = 0,
    min_cluster_count: int = 0,
    min_cluster_truth_count: int = 0,
    min_cluster_images: int = 0,
    min_cluster_truth_fraction: float = 0.0,
) -> dict[str, Any]:
    truth_path = Path(truth_path)
    image_path_value = None if image_path is None else Path(image_path)
    image_reports = truth_coverage_image_reports(truth_path, image_path_value)
    truth_count = sum(item["truth_count"] for item in image_reports)
    cluster_count = sum(item["cluster_count"] for item in image_reports)
    cluster_truth_count = sum(item["cluster_truth_count"] for item in image_reports)
    cluster_image_count = sum(1 for item in image_reports if item["cluster_count"] > 0)
    cluster_truth_fraction = cluster_truth_count / max(1, truth_count)
    checks = [
        _threshold_check("truth_count", truth_count >= min_truth_count, f"{truth_count} >= {min_truth_count}"),
        _threshold_check("cluster_count", cluster_count >= min_cluster_count, f"{cluster_count} >= {min_cluster_count}"),
        _threshold_check(
            "cluster_truth_count",
            cluster_truth_count >= min_cluster_truth_count,
            f"{cluster_truth_count} >= {min_cluster_truth_count}",
        ),
        _threshold_check(
            "cluster_image_count",
            cluster_image_count >= min_cluster_images,
            f"{cluster_image_count} >= {min_cluster_images}",
        ),
        _threshold_check(
            "cluster_truth_fraction",
            cluster_truth_fraction >= min_cluster_truth_fraction,
            f"{cluster_truth_fraction:.4f} >= {min_cluster_truth_fraction:.4f}",
        ),
    ]
    cluster_sizes = [size for item in image_reports for size in item["cluster_sizes"]]
    return {
        "created_at": utc_now_iso(),
        "truth_path": str(truth_path),
        "image_path": None if image_path_value is None else str(image_path_value),
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "thresholds": {
            "min_truth_count": min_truth_count,
            "min_cluster_count": min_cluster_count,
            "min_cluster_truth_count": min_cluster_truth_count,
            "min_cluster_images": min_cluster_images,
            "min_cluster_truth_fraction": min_cluster_truth_fraction,
        },
        "image_count": len(image_reports),
        "truth_count": truth_count,
        "cluster_count": cluster_count,
        "cluster_truth_count": cluster_truth_count,
        "cluster_image_count": cluster_image_count,
        "cluster_truth_fraction": cluster_truth_fraction,
        "mean_cluster_size": 0.0 if not cluster_sizes else sum(cluster_sizes) / len(cluster_sizes),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "min_detectable_count_error_rate": 0.0 if truth_count <= 0 else 1.0 / truth_count,
        "min_detectable_cluster_recall_loss": 0.0 if cluster_truth_count <= 0 else 1.0 / cluster_truth_count,
        "min_detectable_cluster_full_detection_loss": 0.0 if cluster_count <= 0 else 1.0 / cluster_count,
        "checks": checks,
        "images": image_reports,
    }


def truth_coverage_image_reports(
    truth_path: str | Path,
    image_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    truth_path = Path(truth_path)
    image_path_value = None if image_path is None else Path(image_path)
    entries = _truth_entries(truth_path, image_path_value)
    return [_image_report(image, points) for image, points in entries]


def write_truth_coverage_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_json


def _truth_entries(truth_path: Path, image_path: Path | None) -> list[tuple[str, list[TruthPoint]]]:
    if image_path is not None:
        return [
            (str(image), read_truth_points(_truth_source_for_image(truth_path, image), image=image))
            for image in iter_images(image_path)
        ]
    if truth_path.is_dir():
        return [
            (str(path.name), read_truth_points(path))
            for path in sorted(truth_path.rglob("*.json"))
            if path.name.endswith(".truth.json") or not path.name.startswith(".")
        ]
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "images" in payload:
        entries = []
        for index, item in enumerate(payload.get("images", [])):
            image = str(item.get("image", f"manifest-entry-{index}"))
            entries.append((image, truth_points_from_payload(item)))
        return entries
    return [(truth_path.name, truth_points_from_payload(payload))]


def _image_report(image: str, points: list[TruthPoint]) -> dict[str, Any]:
    cluster_sizes = _cluster_sizes(points)
    cluster_truth_count = sum(cluster_sizes)
    return {
        "image": image,
        "truth_count": len(points),
        "cluster_count": len(cluster_sizes),
        "cluster_truth_count": cluster_truth_count,
        "cluster_truth_fraction": cluster_truth_count / max(1, len(points)),
        "cluster_sizes": cluster_sizes,
    }


def _cluster_sizes(points: list[TruthPoint]) -> list[int]:
    groups: dict[str, int] = defaultdict(int)
    for point in points:
        if point.group_id is not None:
            groups[point.group_id] += 1
    return sorted((size for size in groups.values() if size >= 2), reverse=True)


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


def _threshold_check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}
