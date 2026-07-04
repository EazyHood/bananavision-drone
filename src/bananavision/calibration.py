from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .metrics import evaluate_points
from .models import Detection
from .truth import read_truth_points


def calibrate_thresholds(
    detections_json: str | Path,
    truth_json: str | Path,
    tolerance_px: float,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    detections_json = Path(detections_json)
    thresholds = thresholds or [round(value / 100.0, 2) for value in range(5, 100, 5)]
    payload = json.loads(detections_json.read_text(encoding="utf-8"))
    image = payload.get("image")
    detections = [_detection_from_dict(item) for item in payload.get("detections", [])]
    truth = read_truth_points(truth_json, image=image)

    rows = []
    for threshold in thresholds:
        filtered = [detection for detection in detections if detection.score >= threshold]
        metrics = evaluate_points(filtered, truth, tolerance_px=tolerance_px)
        rows.append({"threshold": threshold, "count": len(filtered), "metrics": asdict(metrics)})
    best = min(rows, key=_ranking_key) if rows else None
    return {
        "detections_json": str(detections_json),
        "truth_json": str(truth_json),
        "tolerance_px": tolerance_px,
        "best": best,
        "rows": rows,
    }


def write_calibration_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_json


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = row["metrics"]
    return (
        float(metrics["count_error_rate"]),
        -float(metrics.get("cluster_recall", 0.0)),
        -float(metrics["f1"]),
        -float(metrics["precision"]),
    )


def _detection_from_dict(item: dict[str, Any]) -> Detection:
    bbox_values = item.get("bbox", [0, 0, 0, 0])
    center_values = item.get("center", [0, 0])
    return Detection(
        label=str(item.get("label", "banana_plant")),
        score=float(item.get("score", 0.0)),
        bbox=tuple(float(value) for value in bbox_values[:4]),  # type: ignore[arg-type]
        center=(float(center_values[0]), float(center_values[1])),
        area_px=float(item.get("area_px", 0.0)),
        source=str(item.get("source", "json")),
        id=item.get("id"),
        meta=item.get("meta", {}) or {},
    )
