from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .io import write_bundle
from .metrics import (
    BatchMetrics,
    PointMetrics,
    aggregate_point_metrics,
    batch_statistical_evidence,
    evaluate_points,
)
from .models import PredictionResult
from .pipeline import iter_images, make_detector, predict_image
from .truth import TruthPoint, read_truth_points


def evaluate_result(
    result: PredictionResult,
    truth_path: str | Path,
    tolerance_px: float,
) -> PointMetrics:
    truth = read_truth_points(truth_path, image=result.image_path)
    return evaluate_points(result.detections, truth, tolerance_px=tolerance_px)


def evaluate_image(
    image_path: str | Path,
    truth_path: str | Path,
    config,
    tolerance_px: float,
) -> tuple[PredictionResult, PointMetrics]:
    detector = make_detector(config)
    result = predict_image(image_path, config, detector=detector)
    return result, evaluate_result(result, truth_path, tolerance_px)


def acceptance_passed(
    metrics: PointMetrics,
    max_count_error_rate: float,
    min_precision: float,
    min_recall: float,
    min_f1: float,
    min_cluster_recall: float | None = None,
    min_cluster_full_detection_rate: float | None = None,
    min_cluster_count: int | None = None,
) -> bool:
    cluster_support_ok = True if min_cluster_count is None else metrics.cluster_count >= min_cluster_count
    cluster_recall_ok = (
        True
        if min_cluster_recall is None
        else metrics.cluster_truth_count > 0 and metrics.cluster_recall >= min_cluster_recall
    )
    cluster_full_ok = (
        True
        if min_cluster_full_detection_rate is None
        else metrics.cluster_count > 0 and metrics.fully_detected_cluster_rate >= min_cluster_full_detection_rate
    )
    return (
        metrics.count_error_rate <= max_count_error_rate
        and metrics.precision >= min_precision
        and metrics.recall >= min_recall
        and metrics.f1 >= min_f1
        and cluster_support_ok
        and cluster_recall_ok
        and cluster_full_ok
    )


def batch_acceptance_passed(
    metrics: BatchMetrics,
    max_count_error_rate: float,
    min_precision: float,
    min_recall: float,
    min_f1: float,
    max_mean_image_count_error_rate: float | None = None,
    min_truth_count: int | None = None,
    min_precision_ci_lower: float | None = None,
    min_recall_ci_lower: float | None = None,
    max_mean_image_count_error_rate_ci_upper: float | None = None,
    min_cluster_recall: float | None = None,
    min_cluster_full_detection_rate: float | None = None,
    min_cluster_count: int | None = None,
    statistics: dict[str, Any] | None = None,
) -> bool:
    mean_ok = (
        True
        if max_mean_image_count_error_rate is None
        else metrics.mean_abs_image_count_error_rate <= max_mean_image_count_error_rate
    )
    truth_count_ok = True if min_truth_count is None else metrics.truth_count >= min_truth_count
    cluster_support_ok = True if min_cluster_count is None else metrics.cluster_count >= min_cluster_count
    cluster_recall_ok = (
        True
        if min_cluster_recall is None
        else metrics.cluster_truth_count > 0 and metrics.cluster_recall >= min_cluster_recall
    )
    cluster_full_ok = (
        True
        if min_cluster_full_detection_rate is None
        else metrics.cluster_count > 0 and metrics.fully_detected_cluster_rate >= min_cluster_full_detection_rate
    )
    precision_ci_ok = True
    recall_ci_ok = True
    mean_ci_ok = True
    if statistics is not None:
        precision_ci = statistics.get("precision_wilson_ci", {})
        recall_ci = statistics.get("recall_wilson_ci", {})
        mean_ci = statistics.get("mean_abs_image_count_error_rate_ci", {})
        if min_precision_ci_lower is not None:
            precision_ci_ok = float(precision_ci.get("lower", 0.0)) >= min_precision_ci_lower
        if min_recall_ci_lower is not None:
            recall_ci_ok = float(recall_ci.get("lower", 0.0)) >= min_recall_ci_lower
        if max_mean_image_count_error_rate_ci_upper is not None:
            mean_ci_ok = float(mean_ci.get("upper", float("inf"))) <= max_mean_image_count_error_rate_ci_upper
    elif any(
        value is not None
        for value in [min_precision_ci_lower, min_recall_ci_lower, max_mean_image_count_error_rate_ci_upper]
    ):
        return False
    return (
        metrics.count_error_rate <= max_count_error_rate
        and metrics.precision >= min_precision
        and metrics.recall >= min_recall
        and metrics.f1 >= min_f1
        and mean_ok
        and truth_count_ok
        and cluster_support_ok
        and cluster_recall_ok
        and cluster_full_ok
        and precision_ci_ok
        and recall_ci_ok
        and mean_ci_ok
    )


def evaluate_path(
    input_path: str | Path,
    truth_path: str | Path,
    config,
    tolerance_px: float,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    detector = make_detector(config)
    image_reports: list[dict[str, Any]] = []
    point_metrics: list[PointMetrics] = []
    truth_counts: list[int] = []
    prediction_counts: list[int] = []

    for image_path in iter_images(input_path):
        truth_centers = read_truth_for_image(truth_path, image_path)
        result = predict_image(image_path, config, detector=detector)
        if output_dir is not None:
            write_bundle(result, output_dir)
        metrics = evaluate_points(result.detections, truth_centers, tolerance_px=tolerance_px)
        point_metrics.append(metrics)
        truth_counts.append(len(truth_centers))
        prediction_counts.append(result.count)
        image_reports.append(
            {
                "image": str(image_path),
                "truth_count": len(truth_centers),
                "prediction_count": result.count,
                "metrics": asdict(metrics),
                "elapsed_ms": result.meta.get("elapsed_ms"),
            }
        )

    batch_metrics = aggregate_point_metrics(point_metrics, truth_counts, prediction_counts)
    return {
        "input_path": str(input_path),
        "truth_path": str(truth_path),
        "tolerance_px": tolerance_px,
        "metrics": asdict(batch_metrics),
        "images": image_reports,
    }


def read_truth_for_image(truth_path: str | Path, image_path: str | Path) -> list[TruthPoint]:
    truth_path = Path(truth_path)
    image_path = Path(image_path)
    if truth_path.is_dir():
        candidates = [
            truth_path / f"{image_path.stem}.truth.json",
            truth_path / f"{image_path.stem}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return read_truth_points(candidate)
        raise FileNotFoundError(f"No truth file found for {image_path.name} in {truth_path}")
    return read_truth_points(truth_path, image=image_path)


def write_evaluation_report(
    path: str | Path,
    result: PredictionResult,
    metrics: PointMetrics,
    thresholds: dict[str, float],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "image": str(result.image_path),
        "count": result.count,
        "metrics": asdict(metrics),
        "thresholds": thresholds,
        "passed": acceptance_passed(
            metrics,
            max_count_error_rate=thresholds["max_count_error_rate"],
            min_precision=thresholds["min_precision"],
            min_recall=thresholds["min_recall"],
            min_f1=thresholds["min_f1"],
            min_cluster_recall=thresholds.get("min_cluster_recall"),
            min_cluster_full_detection_rate=thresholds.get("min_cluster_full_detection_rate"),
            min_cluster_count=thresholds.get("min_cluster_count"),
        ),
        "runtime": result.meta.get("runtime", {}),
        "elapsed_ms": result.meta.get("elapsed_ms"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_batch_evaluation_report(
    path: str | Path,
    report: dict[str, Any],
    thresholds: dict[str, float],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = BatchMetrics(**report["metrics"])
    confidence_level = float(thresholds.get("confidence_level", 0.95))
    statistics = batch_statistical_evidence(
        metrics,
        [float(image["metrics"]["count_error_rate"]) for image in report.get("images", [])],
        confidence_level=confidence_level,
    )
    payload = {
        **report,
        "thresholds": thresholds,
        "statistics": statistics,
        "passed": batch_acceptance_passed(
            metrics,
            max_count_error_rate=thresholds["max_count_error_rate"],
            min_precision=thresholds["min_precision"],
            min_recall=thresholds["min_recall"],
            min_f1=thresholds["min_f1"],
            max_mean_image_count_error_rate=thresholds.get("max_mean_image_count_error_rate"),
            min_truth_count=thresholds.get("min_truth_count"),
            min_precision_ci_lower=thresholds.get("min_precision_ci_lower"),
            min_recall_ci_lower=thresholds.get("min_recall_ci_lower"),
            max_mean_image_count_error_rate_ci_upper=thresholds.get("max_mean_image_count_error_rate_ci_upper"),
            min_cluster_recall=thresholds.get("min_cluster_recall"),
            min_cluster_full_detection_rate=thresholds.get("min_cluster_full_detection_rate"),
            min_cluster_count=thresholds.get("min_cluster_count"),
            statistics=statistics,
        ),
    }
    report["statistics"] = statistics
    report["passed"] = payload["passed"]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
