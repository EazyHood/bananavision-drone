import json
from pathlib import Path

from bananavision.reporting import build_field_report


def test_build_field_report(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"image_count": 1, "total_detections": 3, "elapsed_ms": 12.5}), encoding="utf-8")
    mission_audit = tmp_path / "mission_audit.json"
    mission_audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": {
                    "image_count": 1,
                    "total_detections": 3,
                    "mission_quality_status": "pass",
                    "prediction_quality_status": "pass",
                },
            }
        ),
        encoding="utf-8",
    )
    mission_quality = tmp_path / "mission_quality.json"
    mission_quality.write_text(
        json.dumps({"status": "pass", "image_count": 2, "pass_count": 2, "warn_count": 0, "fail_count": 0}),
        encoding="utf-8",
    )
    prediction_quality = tmp_path / "prediction_quality.json"
    prediction_quality.write_text(
        json.dumps(
            {
                "status": "warn",
                "detection_count": 12,
                "review_detection_count": 2,
                "review_fraction": 0.166667,
            }
        ),
        encoding="utf-8",
    )
    flight_check = tmp_path / "flight_check.json"
    flight_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "profile": {
                    "observed_gsd_cm": 2.0,
                    "front_overlap": 75.0,
                    "side_overlap": 72.0,
                },
            }
        ),
        encoding="utf-8",
    )
    flight_log = tmp_path / "flight_log.json"
    flight_log.write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": {"row_count": 2, "pass_count": 2, "warn_count": 0, "fail_count": 0, "max_gsd_cm": 2.1},
            }
        ),
        encoding="utf-8",
    )
    capture_coverage = tmp_path / "capture_coverage.json"
    capture_coverage.write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": {
                    "row_count": 4,
                    "missing_image_count": 0,
                    "position_count": 4,
                    "max_step_distance_m": 12.5,
                    "duplicate_position_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    domain_check = tmp_path / "domain_check.json"
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "image_count": 4,
                "outlier_count": 0,
                "outlier_fraction": 0.0,
            }
        ),
        encoding="utf-8",
    )
    geo_accuracy = tmp_path / "geo_accuracy.json"
    geo_accuracy.write_text(
        json.dumps(
            {
                "status": "pass",
                "metrics": {
                    "truth_count": 10,
                    "matched_count": 10,
                    "rmse_m": 0.25,
                    "p95_m": 0.4,
                    "recall": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    validation_plan = tmp_path / "validation_plan.json"
    validation_plan.write_text(
        json.dumps(
            {
                "targets": {"target_count_error_rate": 0.01},
                "minimum_support": {
                    "truth_count": 1000,
                    "cluster_count": 100,
                    "cluster_truth_count": 300,
                    "cluster_image_count": 30,
                },
            }
        ),
        encoding="utf-8",
    )
    truth_quality = tmp_path / "truth_quality.json"
    truth_quality.write_text(
        json.dumps(
            {
                "status": "pass",
                "image_count": 30,
                "truth_count": 1000,
                "bounded_image_count": 30,
                "issue_count": 0,
            }
        ),
        encoding="utf-8",
    )
    truth_coverage = tmp_path / "truth_coverage.json"
    truth_coverage.write_text(
        json.dumps(
            {
                "status": "pass",
                "truth_count": 1000,
                "cluster_count": 100,
                "cluster_truth_count": 300,
                "cluster_image_count": 30,
                "cluster_truth_fraction": 0.3,
            }
        ),
        encoding="utf-8",
    )
    stratified_truth_coverage = tmp_path / "stratified_truth_coverage.json"
    stratified_truth_coverage.write_text(
        json.dumps({"status": "pass", "stratum_count": 2, "failed_stratum_count": 0}),
        encoding="utf-8",
    )
    tuning = tmp_path / "tuning.json"
    tuning.write_text(
        json.dumps({"best": {"count": 3, "metrics": {"count_error_rate": 0.0, "f1": 1.0}}}),
        encoding="utf-8",
    )
    stratified_acceptance = tmp_path / "stratified_acceptance.json"
    stratified_acceptance.write_text(
        json.dumps({"status": "pass", "stratum_count": 2, "failed_stratum_count": 0}),
        encoding="utf-8",
    )
    cluster_review = tmp_path / "cluster_review.json"
    cluster_review.write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": {"cluster_count": 2, "failed_cluster_count": 0},
                "clusters": [],
            }
        ),
        encoding="utf-8",
    )
    release_audit = tmp_path / "release_audit.json"
    release_audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "gates": [
                    {"name": "acceptance_passed", "status": "pass"},
                    {"name": "truth_quality_passed", "status": "pass"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = build_field_report(
        tmp_path / "report.html",
        run_manifest=manifest,
        mission_audit_report=mission_audit,
        mission_quality_report=mission_quality,
        prediction_quality_report=prediction_quality,
        flight_check_report=flight_check,
        flight_log_report=flight_log,
        capture_coverage_report=capture_coverage,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        validation_plan_report=validation_plan,
        truth_quality_report=truth_quality,
        truth_coverage_report=truth_coverage,
        stratified_truth_coverage_report=stratified_truth_coverage,
        stratified_acceptance_report=stratified_acceptance,
        tuning_report=tuning,
        cluster_review_report=cluster_review,
        release_audit_report=release_audit,
        title="Demo",
    )
    text = output.read_text(encoding="utf-8")
    assert "Demo" in text
    assert "Run Manifest" in text
    assert "Mission Audit" in text
    assert "Mission Quality" in text
    assert "Prediction Quality" in text
    assert "Flight Check" in text
    assert "Flight Log Audit" in text
    assert "Capture Coverage" in text
    assert "Domain Check" in text
    assert "Geo Accuracy" in text
    assert "Validation Plan" in text
    assert "Truth Quality" in text
    assert "Truth Coverage" in text
    assert "Stratified Truth Coverage" in text
    assert "Stratified Acceptance" in text
    assert "Tuning" in text
    assert "Cluster Review" in text
    assert "Release Audit" in text
    assert "GSD cm/px" in text
    assert "Max GSD cm" in text
    assert "Max step m" in text
    assert "Outlier frac" in text
    assert "RMSE m" in text
    assert "Truth minimum" in text
    assert "Cluster frac" in text
    assert "Detections" in text
    assert "Review frac" in text
