from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot, sqrt
from statistics import NormalDist
from typing import Any

from .models import Detection
from .truth import TruthPoint


@dataclass(frozen=True)
class PointMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    count_error: int
    count_error_rate: float
    cluster_count: int = 0
    cluster_truth_count: int = 0
    cluster_matched_count: int = 0
    cluster_recall: float = 0.0
    fully_detected_cluster_count: int = 0
    fully_detected_cluster_rate: float = 0.0
    under_split_cluster_count: int = 0
    over_split_cluster_count: int = 0
    cluster_extra_prediction_count: int = 0


@dataclass(frozen=True)
class BatchMetrics:
    images: int
    truth_count: int
    prediction_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    count_error: int
    count_error_rate: float
    mean_abs_image_count_error_rate: float
    worst_image_count_error_rate: float
    cluster_count: int = 0
    cluster_truth_count: int = 0
    cluster_matched_count: int = 0
    cluster_recall: float = 0.0
    fully_detected_cluster_count: int = 0
    fully_detected_cluster_rate: float = 0.0
    under_split_cluster_count: int = 0
    over_split_cluster_count: int = 0
    cluster_extra_prediction_count: int = 0


def evaluate_points(
    predictions: list[Detection],
    truth_centers: Sequence[TruthPoint | tuple[float, float] | list[Any]],
    tolerance_px: float,
) -> PointMetrics:
    truth_points = [_as_truth_point(point) for point in truth_centers]
    matched_truth: set[int] = set()
    true_positives = 0
    false_positives = 0
    unmatched_prediction_centers: list[tuple[float, float]] = []
    ordered = sorted(predictions, key=lambda det: det.score, reverse=True)
    for prediction in ordered:
        best_index = None
        best_distance = float("inf")
        px, py = prediction.center
        for index, truth_point in enumerate(truth_points):
            if index in matched_truth:
                continue
            tx, ty = truth_point.center
            distance = hypot(px - tx, py - ty)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None and best_distance <= tolerance_px:
            true_positives += 1
            matched_truth.add(best_index)
        else:
            false_positives += 1
            unmatched_prediction_centers.append(prediction.center)
    false_negatives = len(truth_points) - len(matched_truth)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    count_error = len(predictions) - len(truth_points)
    count_error_rate = abs(count_error) / max(1, len(truth_points))
    cluster_metrics = _cluster_metrics(
        truth_points,
        matched_truth,
        unmatched_prediction_centers,
        tolerance_px=tolerance_px,
    )
    return PointMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        count_error=count_error,
        count_error_rate=count_error_rate,
        **cluster_metrics,
    )


def aggregate_point_metrics(metrics: list[PointMetrics], truth_counts: list[int], prediction_counts: list[int]) -> BatchMetrics:
    true_positives = sum(item.true_positives for item in metrics)
    false_positives = sum(item.false_positives for item in metrics)
    false_negatives = sum(item.false_negatives for item in metrics)
    truth_count = sum(truth_counts)
    prediction_count = sum(prediction_counts)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    count_error = prediction_count - truth_count
    count_error_rate = abs(count_error) / max(1, truth_count)
    image_rates = [item.count_error_rate for item in metrics]
    mean_abs = sum(image_rates) / max(1, len(image_rates))
    worst = max(image_rates) if image_rates else 0.0
    cluster_count = sum(item.cluster_count for item in metrics)
    cluster_truth_count = sum(item.cluster_truth_count for item in metrics)
    cluster_matched_count = sum(item.cluster_matched_count for item in metrics)
    fully_detected_cluster_count = sum(item.fully_detected_cluster_count for item in metrics)
    under_split_cluster_count = sum(item.under_split_cluster_count for item in metrics)
    over_split_cluster_count = sum(item.over_split_cluster_count for item in metrics)
    cluster_extra_prediction_count = sum(item.cluster_extra_prediction_count for item in metrics)
    return BatchMetrics(
        images=len(metrics),
        truth_count=truth_count,
        prediction_count=prediction_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        count_error=count_error,
        count_error_rate=count_error_rate,
        mean_abs_image_count_error_rate=mean_abs,
        worst_image_count_error_rate=worst,
        cluster_count=cluster_count,
        cluster_truth_count=cluster_truth_count,
        cluster_matched_count=cluster_matched_count,
        cluster_recall=cluster_matched_count / max(1, cluster_truth_count),
        fully_detected_cluster_count=fully_detected_cluster_count,
        fully_detected_cluster_rate=fully_detected_cluster_count / max(1, cluster_count),
        under_split_cluster_count=under_split_cluster_count,
        over_split_cluster_count=over_split_cluster_count,
        cluster_extra_prediction_count=cluster_extra_prediction_count,
    )


def batch_statistical_evidence(
    metrics: BatchMetrics,
    image_count_error_rates: list[float],
    confidence_level: float = 0.95,
) -> dict:
    z = _z_score(confidence_level)
    precision_trials = metrics.true_positives + metrics.false_positives
    recall_trials = metrics.true_positives + metrics.false_negatives
    return {
        "confidence_level": confidence_level,
        "sample_support": {
            "images": metrics.images,
            "truth_count": metrics.truth_count,
            "prediction_count": metrics.prediction_count,
            "cluster_count": metrics.cluster_count,
            "cluster_truth_count": metrics.cluster_truth_count,
            "min_detectable_count_error_rate": 0.0 if metrics.truth_count <= 0 else 1.0 / metrics.truth_count,
            "min_detectable_cluster_recall_loss": (
                0.0 if metrics.cluster_truth_count <= 0 else 1.0 / metrics.cluster_truth_count
            ),
            "min_detectable_cluster_full_detection_loss": (
                0.0 if metrics.cluster_count <= 0 else 1.0 / metrics.cluster_count
            ),
        },
        "precision_wilson_ci": _wilson_interval(metrics.true_positives, precision_trials, z),
        "recall_wilson_ci": _wilson_interval(metrics.true_positives, recall_trials, z),
        "cluster_recall_wilson_ci": _wilson_interval(
            metrics.cluster_matched_count,
            metrics.cluster_truth_count,
            z,
        ),
        "cluster_full_detection_wilson_ci": _wilson_interval(
            metrics.fully_detected_cluster_count,
            metrics.cluster_count,
            z,
        ),
        "mean_abs_image_count_error_rate_ci": _mean_interval(image_count_error_rates, z),
    }


def _as_truth_point(point: TruthPoint | tuple[float, float] | list[Any]) -> TruthPoint:
    if isinstance(point, TruthPoint):
        return point
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        group_id = None if len(point) < 3 or point[2] is None else str(point[2])
        return TruthPoint(float(point[0]), float(point[1]), group_id)
    raise ValueError(f"Invalid truth point: {point!r}")


def _cluster_metrics(
    truth_points: list[TruthPoint],
    matched_truth: set[int],
    unmatched_prediction_centers: list[tuple[float, float]],
    tolerance_px: float,
) -> dict[str, int | float]:
    groups = _cluster_groups(truth_points)
    cluster_count = len(groups)
    cluster_truth_count = sum(len(indices) for indices in groups.values())
    cluster_matched_count = sum(1 for indices in groups.values() for index in indices if index in matched_truth)
    fully_detected_cluster_count = sum(
        1 for indices in groups.values() if all(index in matched_truth for index in indices)
    )
    under_split_cluster_count = sum(
        1 for indices in groups.values() if any(index not in matched_truth for index in indices)
    )
    extra_prediction_count = 0
    groups_with_extra: set[str] = set()
    for center in unmatched_prediction_centers:
        group_id = _nearest_cluster_group(truth_points, groups, center, tolerance_px)
        if group_id is not None:
            extra_prediction_count += 1
            groups_with_extra.add(group_id)
    return {
        "cluster_count": cluster_count,
        "cluster_truth_count": cluster_truth_count,
        "cluster_matched_count": cluster_matched_count,
        "cluster_recall": cluster_matched_count / max(1, cluster_truth_count),
        "fully_detected_cluster_count": fully_detected_cluster_count,
        "fully_detected_cluster_rate": fully_detected_cluster_count / max(1, cluster_count),
        "under_split_cluster_count": under_split_cluster_count,
        "over_split_cluster_count": len(groups_with_extra),
        "cluster_extra_prediction_count": extra_prediction_count,
    }


def _cluster_groups(truth_points: list[TruthPoint]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, point in enumerate(truth_points):
        if point.group_id is None:
            continue
        groups.setdefault(point.group_id, []).append(index)
    return {group_id: indices for group_id, indices in groups.items() if len(indices) >= 2}


def _nearest_cluster_group(
    truth_points: list[TruthPoint],
    groups: dict[str, list[int]],
    center: tuple[float, float],
    tolerance_px: float,
) -> str | None:
    px, py = center
    best_group = None
    best_distance = float("inf")
    for group_id, indices in groups.items():
        for index in indices:
            tx, ty = truth_points[index].center
            distance = hypot(px - tx, py - ty)
            if distance < best_distance:
                best_distance = distance
                best_group = group_id
    if best_group is None or best_distance > tolerance_px:
        return None
    return best_group


def _wilson_interval(successes: int, total: int, z: float) -> dict[str, float | int]:
    if total <= 0:
        return {"estimate": 0.0, "lower": 0.0, "upper": 0.0, "successes": successes, "total": total}
    estimate = successes / total
    denominator = 1.0 + (z * z / total)
    center = (estimate + (z * z / (2.0 * total))) / denominator
    half_width = (z / denominator) * sqrt((estimate * (1.0 - estimate) / total) + (z * z / (4.0 * total * total)))
    return {
        "estimate": estimate,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
        "successes": successes,
        "total": total,
    }


def _mean_interval(values: list[float], z: float) -> dict[str, float | int]:
    if not values:
        return {"estimate": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    estimate = sum(values) / len(values)
    if len(values) == 1:
        return {"estimate": estimate, "lower": estimate, "upper": estimate, "n": 1}
    variance = sum((value - estimate) ** 2 for value in values) / (len(values) - 1)
    margin = z * sqrt(variance / len(values))
    return {
        "estimate": estimate,
        "lower": max(0.0, estimate - margin),
        "upper": estimate + margin,
        "n": len(values),
    }


def _z_score(confidence_level: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
