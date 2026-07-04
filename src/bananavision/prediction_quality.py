from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from .postprocess import bbox_iou
from .runtime import utc_now_iso


@dataclass(frozen=True)
class PredictionQualityThresholds:
    low_confidence: float = 0.45
    high_split_count: int = 3
    edge_margin_px: float = 8.0
    min_center_distance_px: float = 24.0
    duplicate_iou: float = 0.65
    max_review_fraction: float = 0.2
    fail_on_zero_detections: bool = False


def audit_prediction_outputs(
    predictions_dir: str | Path,
    output_json: str | Path,
    thresholds: PredictionQualityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or PredictionQualityThresholds()
    prediction_files = sorted(Path(predictions_dir).rglob("*.detections.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No *.detections.json files found in: {predictions_dir}")
    image_reports = [_audit_prediction_file(path, thresholds) for path in prediction_files]
    detection_count = sum(report["detection_count"] for report in image_reports)
    review_detections = sum(report["review_detection_count"] for report in image_reports)
    zero_detection_images = sum(1 for report in image_reports if report["detection_count"] == 0)
    fail_count = sum(1 for report in image_reports if report["status"] == "fail")
    warn_count = sum(1 for report in image_reports if report["status"] == "warn")
    pass_count = sum(1 for report in image_reports if report["status"] == "pass")
    review_fraction = 0.0 if detection_count == 0 else review_detections / detection_count
    status = "fail" if fail_count else "warn" if warn_count else "pass"
    if detection_count > 0 and review_fraction > thresholds.max_review_fraction:
        status = "fail"
    report = {
        "created_at": utc_now_iso(),
        "predictions_dir": str(predictions_dir),
        "status": status,
        "image_count": len(image_reports),
        "detection_count": detection_count,
        "review_detection_count": review_detections,
        "review_fraction": round(review_fraction, 6),
        "zero_detection_images": zero_detection_images,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "thresholds": asdict(thresholds),
        "images": image_reports,
    }
    write_prediction_quality_report(report, output_json)
    return report


def write_prediction_quality_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_prediction_quality_csv(report, output_json.with_suffix(".csv"))
    return output_json


def _audit_prediction_file(path: Path, thresholds: PredictionQualityThresholds) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    width = int(payload.get("width") or 0)
    height = int(payload.get("height") or 0)
    detections = payload.get("detections", []) or []
    detection_reports = [_audit_detection(detection, width, height, thresholds) for detection in detections]
    _add_pairwise_risks(detection_reports, thresholds)
    review_detection_count = sum(1 for detection in detection_reports if detection["review_required"])
    image_issues = []
    if not detections:
        image_issues.append("zero_detections")
    if review_detection_count:
        image_issues.append(f"{review_detection_count}_detections_require_review")
    status = "pass"
    if image_issues:
        status = "warn"
    if not detections and thresholds.fail_on_zero_detections:
        status = "fail"
    return {
        "prediction_file": str(path),
        "image": str(payload.get("image") or ""),
        "width": width,
        "height": height,
        "detection_count": len(detection_reports),
        "review_detection_count": review_detection_count,
        "status": status,
        "issues": image_issues,
        "detections": detection_reports,
    }


def _audit_detection(
    detection: dict[str, Any],
    width: int,
    height: int,
    thresholds: PredictionQualityThresholds,
) -> dict[str, Any]:
    score = float(detection.get("score", 0.0))
    bbox = [float(value) for value in detection.get("bbox", [0, 0, 0, 0])[:4]]
    meta = detection.get("meta", {}) or {}
    split_count = int(meta.get("component_split_count", 1))
    risks = []
    if score <= thresholds.low_confidence:
        risks.append("low_confidence")
    if split_count >= thresholds.high_split_count:
        risks.append("high_cluster_split")
    if _touches_edge(bbox, width, height, thresholds.edge_margin_px):
        risks.append("edge_detection")
    return {
        "id": detection.get("id"),
        "score": score,
        "bbox": bbox,
        "center": detection.get("center", []),
        "area_px": float(detection.get("area_px", 0.0)),
        "component_split_count": split_count,
        "risks": risks,
        "review_required": bool(risks),
    }


def _add_pairwise_risks(detections: list[dict[str, Any]], thresholds: PredictionQualityThresholds) -> None:
    for left_index, left in enumerate(detections):
        for right in detections[left_index + 1 :]:
            duplicate = bbox_iou(tuple(left["bbox"]), tuple(right["bbox"])) >= thresholds.duplicate_iou
            crowded = _center_distance(left.get("center", []), right.get("center", [])) < thresholds.min_center_distance_px
            if duplicate:
                _add_risk(left, "duplicate_overlap")
                _add_risk(right, "duplicate_overlap")
            if crowded:
                _add_risk(left, "crowded_center")
                _add_risk(right, "crowded_center")


def _touches_edge(bbox: list[float], width: int, height: int, margin: float) -> bool:
    if width <= 0 or height <= 0 or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def _center_distance(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("inf")
    return sqrt((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2)


def _add_risk(detection: dict[str, Any], risk: str) -> None:
    if risk not in detection["risks"]:
        detection["risks"].append(risk)
    detection["review_required"] = True


def _write_prediction_quality_csv(report: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prediction_file",
        "image",
        "detection_id",
        "score",
        "bbox",
        "component_split_count",
        "review_required",
        "risks",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_report in report.get("images", []):
            for detection in image_report.get("detections", []):
                writer.writerow(
                    {
                        "prediction_file": image_report.get("prediction_file", ""),
                        "image": image_report.get("image", ""),
                        "detection_id": detection.get("id", ""),
                        "score": detection.get("score", ""),
                        "bbox": json.dumps(detection.get("bbox", [])),
                        "component_split_count": detection.get("component_split_count", ""),
                        "review_required": detection.get("review_required", False),
                        "risks": " | ".join(detection.get("risks", [])),
                    }
                )
