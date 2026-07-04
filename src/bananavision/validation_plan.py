from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .runtime import utc_now_iso


def build_validation_plan(
    target_count_error_rate: float = 0.01,
    target_cluster_recall_loss: float = 0.01,
    target_cluster_full_detection_loss: float = 0.01,
    farms: int = 3,
    flight_dates: int = 3,
    gsd_bands: int = 2,
    cultivars: int = 1,
    min_plants_per_condition: int = 50,
    min_cluster_mats_per_condition: int = 10,
    min_cluster_truth_fraction: float = 0.20,
) -> dict[str, Any]:
    _validate_rate("target_count_error_rate", target_count_error_rate)
    _validate_rate("target_cluster_recall_loss", target_cluster_recall_loss)
    _validate_rate("target_cluster_full_detection_loss", target_cluster_full_detection_loss)
    _validate_rate("min_cluster_truth_fraction", min_cluster_truth_fraction, allow_zero=True)
    _validate_positive_int("farms", farms)
    _validate_positive_int("flight_dates", flight_dates)
    _validate_positive_int("gsd_bands", gsd_bands)
    _validate_positive_int("cultivars", cultivars)
    _validate_non_negative_int("min_plants_per_condition", min_plants_per_condition)
    _validate_non_negative_int("min_cluster_mats_per_condition", min_cluster_mats_per_condition)

    condition_count = farms * flight_dates * gsd_bands * cultivars
    truth_for_error_resolution = math.ceil(1.0 / target_count_error_rate)
    truth_for_condition_coverage = condition_count * min_plants_per_condition
    minimum_truth_count = max(truth_for_error_resolution, truth_for_condition_coverage)

    cluster_truth_for_recall_resolution = math.ceil(1.0 / target_cluster_recall_loss)
    cluster_mats_for_full_detection_resolution = math.ceil(1.0 / target_cluster_full_detection_loss)
    cluster_mats_for_condition_coverage = condition_count * min_cluster_mats_per_condition
    minimum_cluster_count = max(
        cluster_mats_for_full_detection_resolution,
        cluster_mats_for_condition_coverage,
    )
    minimum_cluster_truth_count = max(
        cluster_truth_for_recall_resolution,
        math.ceil(minimum_truth_count * min_cluster_truth_fraction),
    )
    minimum_cluster_images = condition_count if minimum_cluster_count > 0 else 0

    return {
        "created_at": utc_now_iso(),
        "status": "plan",
        "targets": {
            "target_count_error_rate": target_count_error_rate,
            "target_cluster_recall_loss": target_cluster_recall_loss,
            "target_cluster_full_detection_loss": target_cluster_full_detection_loss,
            "min_cluster_truth_fraction": min_cluster_truth_fraction,
        },
        "operating_domain": {
            "farms": farms,
            "flight_dates": flight_dates,
            "gsd_bands": gsd_bands,
            "cultivars": cultivars,
            "condition_count": condition_count,
            "min_plants_per_condition": min_plants_per_condition,
            "min_cluster_mats_per_condition": min_cluster_mats_per_condition,
        },
        "minimum_support": {
            "truth_count": minimum_truth_count,
            "cluster_count": minimum_cluster_count,
            "cluster_truth_count": minimum_cluster_truth_count,
            "cluster_image_count": minimum_cluster_images,
            "plants_per_condition": _ceil_div(minimum_truth_count, condition_count),
            "cluster_mats_per_condition": _ceil_div(minimum_cluster_count, condition_count),
            "cluster_truth_per_condition": _ceil_div(minimum_cluster_truth_count, condition_count),
        },
        "resolution": {
            "min_detectable_count_error_rate": 1.0 / minimum_truth_count,
            "min_detectable_cluster_recall_loss": 1.0 / minimum_cluster_truth_count,
            "min_detectable_cluster_full_detection_loss": 1.0 / minimum_cluster_count,
        },
        "recommended_acceptance_args": {
            "--target-count-error-rate": target_count_error_rate,
            "--max-count-error-rate": target_count_error_rate,
            "--min-truth-count": minimum_truth_count,
            "--min-cluster-count": minimum_cluster_count,
            "--min-cluster-truth-count": minimum_cluster_truth_count,
            "--min-cluster-images": minimum_cluster_images,
            "--min-cluster-truth-fraction": min_cluster_truth_fraction,
            "--min-cluster-recall": 1.0 - target_cluster_recall_loss,
            "--min-cluster-full-detection-rate": 1.0 - target_cluster_full_detection_loss,
        },
        "notes": [
            "This is a validation sampling plan, not model evidence.",
            "Collect and lock the holdout before tuning final thresholds.",
            "Use separate calibration and final acceptance sets.",
            "Increase support when the farm has stronger cultivar, age, canopy, slope, lighting, or sensor variation.",
        ],
    }


def write_validation_plan(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_json


def _validate_rate(name: str, value: float, allow_zero: bool = False) -> None:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value >= 1.0:
        lower = "0 <= " if allow_zero else "0 < "
        raise ValueError(f"{name} must satisfy {lower}{name} < 1")


def _validate_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _ceil_div(numerator: int, denominator: int) -> int:
    return math.ceil(numerator / denominator)
