from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inventory import update_inventory
from .mission_quality import MissionQualityThresholds, audit_mission_images
from .models import InferenceConfig
from .pipeline import predict_path
from .prediction_quality import PredictionQualityThresholds, audit_prediction_outputs
from .reporting import build_field_report
from .runtime import runtime_fingerprint, utc_now_iso


def process_mission(
    input_path: str | Path,
    output_dir: str | Path,
    config: InferenceConfig,
    mission_quality_thresholds: MissionQualityThresholds | None = None,
    prediction_quality_thresholds: PredictionQualityThresholds | None = None,
    inventory_dir: str | Path | None = None,
    inventory_distance_threshold: float = 1.2,
    id_prefix: str = "banana-plant",
    observed_at: str | None = None,
    report_title: str = "BananaVision Mission Report",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mission_quality_path = output_dir / "mission_quality" / "mission_quality_report.json"
    infer_dir = output_dir / "infer"
    prediction_quality_path = output_dir / "prediction_quality" / "prediction_quality_report.json"
    field_report_path = output_dir / "field_report.html"

    mission_quality = audit_mission_images(
        input_path,
        mission_quality_path,
        mission_quality_thresholds or MissionQualityThresholds(),
    )
    results = predict_path(input_path, infer_dir, config)
    prediction_quality = audit_prediction_outputs(
        infer_dir,
        prediction_quality_path,
        prediction_quality_thresholds or PredictionQualityThresholds(),
    )
    inventory_summary = None
    if inventory_dir is not None:
        inventory_summary = update_inventory(
            infer_dir / "mission.detections.geojson",
            inventory_dir,
            distance_threshold=inventory_distance_threshold,
            id_prefix=id_prefix,
            observed_at=observed_at,
        )

    build_field_report(
        field_report_path,
        run_manifest=infer_dir / "run_manifest.json",
        mission_quality_report=mission_quality_path,
        prediction_quality_report=prediction_quality_path,
        title=report_title,
    )
    manifest = {
        "created_at": utc_now_iso(),
        "status": _overall_status(mission_quality["status"], prediction_quality["status"]),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "runtime": runtime_fingerprint(config),
        "image_count": len(results),
        "total_detections": sum(result.count for result in results),
        "artifacts": {
            "mission_quality_report": str(mission_quality_path),
            "run_manifest": str(infer_dir / "run_manifest.json"),
            "prediction_quality_report": str(prediction_quality_path),
            "field_report": str(field_report_path),
        },
        "mission_quality": _summary(mission_quality),
        "prediction_quality": _summary(prediction_quality),
        "inventory": inventory_summary,
    }
    manifest_path = output_dir / "mission_process_manifest.json"
    manifest["artifacts"]["mission_process_manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _overall_status(*statuses: str) -> str:
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "status",
        "image_count",
        "pass_count",
        "warn_count",
        "fail_count",
        "detection_count",
        "review_detection_count",
        "review_fraction",
    ]
    return {key: report[key] for key in keys if key in report}
