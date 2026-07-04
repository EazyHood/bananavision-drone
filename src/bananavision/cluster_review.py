from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any

from PIL import Image

from .models import Detection
from .truth import TruthPoint, read_truth_points


@dataclass(frozen=True)
class ClusterReviewPaths:
    json: Path
    csv: Path
    crops_manifest: Path | None = None


def build_cluster_review(
    detections_json: str | Path,
    truth_json: str | Path,
    output_json: str | Path,
    tolerance_px: float,
    image_path: str | Path | None = None,
    crops_dir: str | Path | None = None,
    crop_margin_px: int = 48,
) -> dict[str, Any]:
    detections_path = Path(detections_json)
    payload = json.loads(detections_path.read_text(encoding="utf-8"))
    image = _resolve_image_path(image_path, payload, detections_path.parent)
    detections = _detections_from_payload(payload)
    truth = read_truth_points(truth_json, image=image)

    matches, unmatched_prediction_indices = _match_predictions(detections, truth, tolerance_px)
    clusters = _cluster_reports(detections, truth, matches, unmatched_prediction_indices, tolerance_px)
    summary = _summary(clusters, truth, detections)
    report = {
        "schema_version": 1,
        "status": _status(summary),
        "detections_json": str(detections_path),
        "truth_json": str(truth_json),
        "image": None if image is None else str(image),
        "tolerance_px": tolerance_px,
        "summary": summary,
        "clusters": clusters,
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_cluster_review_csv(report, output_json.with_suffix(".csv"))
    if crops_dir is not None:
        manifest = export_cluster_review_crops(report, crops_dir, margin_px=crop_margin_px)
        report["crops_manifest"] = manifest["manifest_path"]
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_cluster_review_csv(report: dict[str, Any], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "group_id",
        "status",
        "issue",
        "truth_count",
        "matched_truth_count",
        "missing_truth_count",
        "extra_prediction_count",
        "priority",
        "review_bbox",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster in report.get("clusters", []):
            writer.writerow(
                {
                    "image": report.get("image") or "",
                    "group_id": cluster["group_id"],
                    "status": cluster["status"],
                    "issue": ",".join(cluster["issues"]),
                    "truth_count": cluster["truth_count"],
                    "matched_truth_count": cluster["matched_truth_count"],
                    "missing_truth_count": cluster["missing_truth_count"],
                    "extra_prediction_count": cluster["extra_prediction_count"],
                    "priority": cluster["priority"],
                    "review_bbox": json.dumps(cluster["review_bbox"]),
                }
            )
    return output_csv


def export_cluster_review_crops(
    report: dict[str, Any],
    output_dir: str | Path,
    margin_px: int = 48,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_value = report.get("image")
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    image_path = None if not image_value else Path(str(image_value))
    if image_path is None or not image_path.exists() or not image_path.is_file():
        skipped.append({"reason": "image_missing", "image": "" if image_path is None else str(image_path)})
    else:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for cluster in report.get("clusters", []):
                if cluster.get("status") == "pass":
                    continue
                crop_box = _expand_bbox(cluster["review_bbox"], image.size, margin_px)
                crop = image.crop(crop_box)
                crop_name = f"{_safe_stem(cluster['group_id'])}_{cluster['status']}.jpg"
                crop_path = output_dir / crop_name
                crop.save(crop_path, quality=92)
                exported.append(
                    {
                        "group_id": cluster["group_id"],
                        "crop": str(crop_path),
                        "crop_box": list(crop_box),
                        "issues": cluster["issues"],
                        "priority": cluster["priority"],
                    }
                )
    manifest = {
        "manifest_path": str(output_dir / "cluster_review_crops_manifest.json"),
        "image": report.get("image"),
        "exported_count": len(exported),
        "skipped_count": len(skipped),
        "items": exported,
        "skipped": skipped,
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _detections_from_payload(payload: dict[str, Any]) -> list[Detection]:
    detections = []
    for item in payload.get("detections", []) or []:
        center = item.get("center", [0.0, 0.0])
        bbox = item.get("bbox", [center[0], center[1], center[0], center[1]])
        detections.append(
            Detection(
                label=str(item.get("label", "banana_plant")),
                score=float(item.get("score", 0.0)),
                bbox=tuple(float(value) for value in bbox[:4]),  # type: ignore[arg-type]
                center=(float(center[0]), float(center[1])),
                area_px=float(item.get("area_px", 0.0)),
                source=str(item.get("source", "review")),
                id=None if item.get("id") is None else str(item.get("id")),
                meta=item.get("meta", {}) or {},
            )
        )
    return detections


def _match_predictions(
    detections: list[Detection],
    truth: list[TruthPoint],
    tolerance_px: float,
) -> tuple[dict[int, int], set[int]]:
    matched_truth: set[int] = set()
    matches: dict[int, int] = {}
    ordered = sorted(enumerate(detections), key=lambda item: item[1].score, reverse=True)
    unmatched_predictions: set[int] = set()
    for pred_index, detection in ordered:
        best_index = None
        best_distance = float("inf")
        for truth_index, truth_point in enumerate(truth):
            if truth_index in matched_truth:
                continue
            distance = _distance(detection.center, truth_point.center)
            if distance < best_distance:
                best_distance = distance
                best_index = truth_index
        if best_index is not None and best_distance <= tolerance_px:
            matched_truth.add(best_index)
            matches[best_index] = pred_index
        else:
            unmatched_predictions.add(pred_index)
    return matches, unmatched_predictions


def _cluster_reports(
    detections: list[Detection],
    truth: list[TruthPoint],
    matches: dict[int, int],
    unmatched_prediction_indices: set[int],
    tolerance_px: float,
) -> list[dict[str, Any]]:
    groups = _groups(truth)
    reports = []
    for group_id, truth_indices in groups.items():
        matched = [index for index in truth_indices if index in matches]
        missing = [index for index in truth_indices if index not in matches]
        extras = _extra_predictions_near_group(detections, truth, truth_indices, unmatched_prediction_indices, tolerance_px)
        issues = []
        if missing:
            issues.append("under_split")
        if extras:
            issues.append("over_split")
        status = "pass" if not issues else "fail"
        bbox = _review_bbox(detections, truth, truth_indices, [matches[index] for index in matched], extras)
        priority = (len(missing) + len(extras)) / max(1, len(truth_indices))
        reports.append(
            {
                "group_id": group_id,
                "status": status,
                "issues": issues,
                "truth_count": len(truth_indices),
                "matched_truth_count": len(matched),
                "missing_truth_count": len(missing),
                "extra_prediction_count": len(extras),
                "priority": round(priority, 6),
                "truth": [_truth_payload(truth[index], matched_prediction=matches.get(index)) for index in truth_indices],
                "matched_prediction_indices": [matches[index] for index in matched],
                "extra_prediction_indices": extras,
                "review_bbox": bbox,
            }
        )
    return sorted(reports, key=lambda item: (item["status"] != "fail", -float(item["priority"]), item["group_id"]))


def _groups(truth: list[TruthPoint]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, point in enumerate(truth):
        if point.group_id is not None:
            groups.setdefault(point.group_id, []).append(index)
    return {group_id: indices for group_id, indices in groups.items() if len(indices) >= 2}


def _extra_predictions_near_group(
    detections: list[Detection],
    truth: list[TruthPoint],
    truth_indices: list[int],
    unmatched_prediction_indices: set[int],
    tolerance_px: float,
) -> list[int]:
    extras = []
    for pred_index in unmatched_prediction_indices:
        center = detections[pred_index].center
        if any(_distance(center, truth[index].center) <= tolerance_px for index in truth_indices):
            extras.append(pred_index)
    return extras


def _review_bbox(
    detections: list[Detection],
    truth: list[TruthPoint],
    truth_indices: list[int],
    matched_prediction_indices: list[int],
    extra_prediction_indices: list[int],
) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for index in truth_indices:
        x, y = truth[index].center
        xs.append(x)
        ys.append(y)
    for index in [*matched_prediction_indices, *extra_prediction_indices]:
        x1, y1, x2, y2 = detections[index].bbox
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


def _summary(clusters: list[dict[str, Any]], truth: list[TruthPoint], detections: list[Detection]) -> dict[str, Any]:
    failed = [cluster for cluster in clusters if cluster["status"] == "fail"]
    return {
        "truth_count": len(truth),
        "prediction_count": len(detections),
        "cluster_count": len(clusters),
        "failed_cluster_count": len(failed),
        "under_split_cluster_count": sum(1 for cluster in clusters if "under_split" in cluster["issues"]),
        "over_split_cluster_count": sum(1 for cluster in clusters if "over_split" in cluster["issues"]),
        "missing_truth_count": sum(int(cluster["missing_truth_count"]) for cluster in clusters),
        "extra_prediction_count": sum(int(cluster["extra_prediction_count"]) for cluster in clusters),
    }


def _status(summary: dict[str, Any]) -> str:
    if int(summary["cluster_count"]) == 0:
        return "warn"
    if int(summary["failed_cluster_count"]) > 0:
        return "fail"
    return "pass"


def _truth_payload(point: TruthPoint, matched_prediction: int | None) -> dict[str, Any]:
    return {
        "x": point.x,
        "y": point.y,
        "group_id": point.group_id,
        "matched_prediction_index": matched_prediction,
    }


def _resolve_image_path(
    image_path: str | Path | None,
    payload: dict[str, Any],
    base_dir: Path,
) -> Path | None:
    value = image_path or payload.get("image")
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return path


def _expand_bbox(
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


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)[:80]
