from bananavision.metrics import (
    aggregate_point_metrics,
    batch_statistical_evidence,
    evaluate_points,
)
from bananavision.models import Detection
from bananavision.truth import TruthPoint


def test_evaluate_points() -> None:
    predictions = [
        Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "test"),
        Detection("banana_plant", 0.8, (50, 50, 60, 60), (55, 55), 100, "test"),
    ]
    metrics = evaluate_points(predictions, [(6, 5), (100, 100)], tolerance_px=3)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_evaluate_points_exposes_failed_banana_cluster_split() -> None:
    predictions = [
        Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "test"),
        Detection("banana_plant", 0.8, (1, 0, 11, 10), (6, 5), 100, "test"),
    ]
    truth = [
        TruthPoint(5, 5, "mat-a"),
        TruthPoint(20, 5, "mat-a"),
    ]

    metrics = evaluate_points(predictions, truth, tolerance_px=2)

    assert metrics.count_error_rate == 0.0
    assert metrics.cluster_count == 1
    assert metrics.cluster_truth_count == 2
    assert metrics.cluster_matched_count == 1
    assert metrics.cluster_recall == 0.5
    assert metrics.fully_detected_cluster_rate == 0.0
    assert metrics.under_split_cluster_count == 1
    assert metrics.over_split_cluster_count == 1
    assert metrics.cluster_extra_prediction_count == 1


def test_evaluate_points_exposes_complete_banana_cluster_split() -> None:
    predictions = [
        Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "test"),
        Detection("banana_plant", 0.8, (15, 0, 25, 10), (20, 5), 100, "test"),
        Detection("banana_plant", 0.7, (30, 0, 40, 10), (35, 5), 100, "test"),
    ]
    truth = [
        TruthPoint(5, 5, "mat-a"),
        TruthPoint(20, 5, "mat-a"),
        TruthPoint(35, 5, "mat-a"),
    ]

    metrics = evaluate_points(predictions, truth, tolerance_px=2)

    assert metrics.cluster_recall == 1.0
    assert metrics.fully_detected_cluster_count == 1
    assert metrics.fully_detected_cluster_rate == 1.0
    assert metrics.under_split_cluster_count == 0
    assert metrics.over_split_cluster_count == 0


def test_aggregate_point_metrics() -> None:
    first = evaluate_points(
        [Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "test")],
        [(5, 5)],
        tolerance_px=2,
    )
    second = evaluate_points(
        [Detection("banana_plant", 0.9, (0, 0, 10, 10), (50, 50), 100, "test")],
        [(5, 5)],
        tolerance_px=2,
    )
    batch = aggregate_point_metrics([first, second], truth_counts=[1, 1], prediction_counts=[1, 1])
    assert batch.images == 2
    assert batch.true_positives == 1
    assert batch.false_positives == 1
    assert batch.false_negatives == 1
    assert batch.precision == 0.5


def test_aggregate_point_metrics_keeps_cluster_support() -> None:
    failed_cluster = evaluate_points(
        [Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "test")],
        [TruthPoint(5, 5, "mat-a"), TruthPoint(20, 5, "mat-a")],
        tolerance_px=2,
    )
    complete_cluster = evaluate_points(
        [
            Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "test"),
            Detection("banana_plant", 0.8, (15, 0, 25, 10), (20, 5), 100, "test"),
        ],
        [TruthPoint(5, 5, "mat-b"), TruthPoint(20, 5, "mat-b")],
        tolerance_px=2,
    )

    batch = aggregate_point_metrics(
        [failed_cluster, complete_cluster],
        truth_counts=[2, 2],
        prediction_counts=[1, 2],
    )

    assert batch.cluster_count == 2
    assert batch.cluster_truth_count == 4
    assert batch.cluster_matched_count == 3
    assert batch.cluster_recall == 0.75
    assert batch.fully_detected_cluster_count == 1
    assert batch.fully_detected_cluster_rate == 0.5


def test_batch_statistical_evidence() -> None:
    first = evaluate_points(
        [Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "test")],
        [(5, 5)],
        tolerance_px=2,
    )
    second = evaluate_points(
        [Detection("banana_plant", 0.9, (0, 0, 10, 10), (50, 50), 100, "test")],
        [(5, 5)],
        tolerance_px=2,
    )
    batch = aggregate_point_metrics([first, second], truth_counts=[1, 1], prediction_counts=[1, 1])
    statistics = batch_statistical_evidence(
        batch,
        [first.count_error_rate, second.count_error_rate],
        confidence_level=0.95,
    )
    assert statistics["sample_support"]["truth_count"] == 2
    assert statistics["sample_support"]["min_detectable_count_error_rate"] == 0.5
    assert statistics["sample_support"]["cluster_count"] == 0
    assert statistics["precision_wilson_ci"]["lower"] < batch.precision
    assert statistics["recall_wilson_ci"]["upper"] <= 1.0
    assert statistics["mean_abs_image_count_error_rate_ci"]["n"] == 2
