import json
from pathlib import Path

from bananavision.release_audit import audit_release


def test_audit_release_passes_complete_evidence(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    stratified_acceptance = tmp_path / "stratified_acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    mission_quality = tmp_path / "mission_quality.json"
    prediction_quality = tmp_path / "prediction_quality.json"
    holdout_verify = tmp_path / "holdout_verify.json"
    validation_plan = tmp_path / "validation_plan.json"
    truth_quality = tmp_path / "truth_quality.json"
    truth_coverage = tmp_path / "truth_coverage.json"
    stratified_truth_coverage = tmp_path / "stratified_truth_coverage.json"
    flight_check = tmp_path / "flight_check.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    model_manifest = tmp_path / "model_manifest.json"
    model_card = tmp_path / "MODEL_CARD.md"
    field_report = tmp_path / "field_report.html"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "truth_count": 2000,
                    "prediction_count": 2000,
                    "count_error_rate": 0.0,
                    "precision": 0.995,
                    "recall": 0.995,
                    "f1": 0.995,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "cluster_matched_count": 60,
                    "cluster_recall": 1.0,
                    "fully_detected_cluster_count": 20,
                    "fully_detected_cluster_rate": 1.0,
                },
                "statistics": {
                    "sample_support": {
                        "truth_count": 2000,
                        "cluster_count": 20,
                        "cluster_truth_count": 60,
                        "min_detectable_count_error_rate": 0.0005,
                    },
                    "precision_wilson_ci": {"lower": 0.985},
                    "recall_wilson_ci": {"lower": 0.986},
                },
            }
        ),
        encoding="utf-8",
    )
    stratified_acceptance.write_text(
        json.dumps(
            {
                "status": "pass",
                "stratum_count": 2,
                "failed_stratum_count": 0,
                "missing_metadata_count": 0,
                "strata": [
                    {"stratum": {"farm": "farm-a"}, "passed": True},
                    {"stratum": {"farm": "farm-b"}, "passed": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")
    mission_quality.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    prediction_quality.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    holdout_verify.write_text(
        json.dumps(
            {
                "status": "pass",
                "truth_count": 2000,
                "cluster_count": 20,
                "cluster_truth_count": 60,
                "min_detectable_count_error_rate": 0.0005,
                "issue_count": 0,
            }
        ),
        encoding="utf-8",
    )
    _write_validation_plan(validation_plan)
    _write_truth_quality(truth_quality)
    truth_coverage.write_text(
        json.dumps(
            {
                "status": "pass",
                "truth_count": 2000,
                "cluster_count": 20,
                "cluster_truth_count": 60,
                "cluster_image_count": 10,
                "cluster_truth_fraction": 0.03,
            }
        ),
        encoding="utf-8",
    )
    _write_stratified_truth_coverage(stratified_truth_coverage)
    flight_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "checks": [
                    {"name": "gsd_available", "status": "pass"},
                    {"name": "gsd_within_validated_range", "status": "pass"},
                    {"name": "front_overlap", "status": "pass"},
                    {"name": "side_overlap", "status": "pass"},
                    {"name": "motion_blur", "status": "pass"},
                ],
            }
        ),
        encoding="utf-8",
    )
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "reference_image_count": 50,
                "outlier_count": 0,
                "outlier_fraction": 0.0,
                "thresholds": {"max_outlier_fraction": 0.0, "min_reference_images": 10},
                "checks": [
                    {"name": "reference_support", "status": "pass"},
                    {"name": "outlier_fraction", "status": "pass"},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_geo_accuracy(geo_accuracy, max_p95_m=0.75)
    model_manifest.write_text(
        json.dumps({"model_sha256": "abc123", "promotion": {"status": "promoted"}}),
        encoding="utf-8",
    )
    model_card.write_text("# Model Card\n\n## Claim Status\n", encoding="utf-8")
    field_report.write_text("<html></html>", encoding="utf-8")

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        stratified_acceptance_report=stratified_acceptance,
        benchmark_report=benchmark,
        mission_quality_report=mission_quality,
        prediction_quality_report=prediction_quality,
        holdout_verify_report=holdout_verify,
        validation_plan_report=validation_plan,
        truth_quality_report=truth_quality,
        truth_coverage_report=truth_coverage,
        stratified_truth_coverage_report=stratified_truth_coverage,
        flight_check_report=flight_check,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        model_manifest=model_manifest,
        model_card=model_card,
        field_report=field_report,
        max_count_error_rate=0.01,
        min_truth_count=1000,
        min_precision_ci_lower=0.98,
        min_recall_ci_lower=0.98,
        min_cluster_count=20,
        min_cluster_truth_count=60,
        min_cluster_images=10,
        min_cluster_truth_fraction=0.03,
        min_cluster_recall=0.99,
        min_cluster_full_detection_rate=0.99,
        max_p95_ms=50,
        max_geo_p95_m=1.0,
    )

    assert report["status"] == "pass"
    assert all(gate["status"] == "pass" for gate in report["gates"])


def test_audit_release_requires_stratified_truth_coverage_for_one_percent_claim(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    stratified_acceptance = tmp_path / "stratified_acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    holdout_verify = tmp_path / "holdout_verify.json"
    validation_plan = tmp_path / "validation_plan.json"
    truth_quality = tmp_path / "truth_quality.json"
    truth_coverage = tmp_path / "truth_coverage.json"
    flight_check = tmp_path / "flight_check.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "truth_count": 2000,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "count_error_rate": 0.0,
                },
                "statistics": {
                    "sample_support": {
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stratified_acceptance.write_text(
        json.dumps({"status": "pass", "stratum_count": 1, "failed_stratum_count": 0, "missing_metadata_count": 0}),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")
    holdout_verify.write_text(
        json.dumps({"status": "pass", "truth_count": 2000, "min_detectable_count_error_rate": 0.0005}),
        encoding="utf-8",
    )
    _write_validation_plan(validation_plan)
    _write_truth_quality(truth_quality)
    truth_coverage.write_text(
        json.dumps(
            {
                "status": "pass",
                "truth_count": 2000,
                "cluster_count": 20,
                "cluster_truth_count": 60,
                "cluster_image_count": 10,
                "cluster_truth_fraction": 0.03,
            }
        ),
        encoding="utf-8",
    )
    flight_check.write_text(json.dumps({"status": "pass", "checks": []}), encoding="utf-8")
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "reference_image_count": 50,
                "outlier_fraction": 0.0,
                "thresholds": {"max_outlier_fraction": 0.0, "min_reference_images": 10},
            }
        ),
        encoding="utf-8",
    )
    _write_geo_accuracy(geo_accuracy)

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        stratified_acceptance_report=stratified_acceptance,
        benchmark_report=benchmark,
        holdout_verify_report=holdout_verify,
        validation_plan_report=validation_plan,
        truth_quality_report=truth_quality,
        truth_coverage_report=truth_coverage,
        flight_check_report=flight_check,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        max_count_error_rate=0.01,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "stratified_truth_coverage_report_present" in failed


def test_audit_release_fails_insufficient_sample_support(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "truth_count": 20,
                    "prediction_count": 20,
                    "count_error_rate": 0.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                },
                "statistics": {
                    "sample_support": {
                        "truth_count": 20,
                        "min_detectable_count_error_rate": 0.05,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        max_count_error_rate=0.01,
        min_truth_count=1000,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "sample_support_resolution" in failed
    assert "min_truth_count" in failed


def test_audit_release_requires_validation_plan_for_one_percent_claim(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    holdout_verify = tmp_path / "holdout_verify.json"
    truth_coverage = tmp_path / "truth_coverage.json"
    flight_check = tmp_path / "flight_check.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "truth_count": 2000,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "count_error_rate": 0.0,
                },
                "statistics": {
                    "sample_support": {
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")
    holdout_verify.write_text(
        json.dumps({"status": "pass", "truth_count": 2000, "min_detectable_count_error_rate": 0.0005}),
        encoding="utf-8",
    )
    truth_coverage.write_text(
        json.dumps(
            {
                "status": "pass",
                "truth_count": 2000,
                "cluster_count": 20,
                "cluster_truth_count": 60,
                "cluster_image_count": 10,
                "cluster_truth_fraction": 0.03,
            }
        ),
        encoding="utf-8",
    )
    flight_check.write_text(json.dumps({"status": "pass", "checks": []}), encoding="utf-8")
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "reference_image_count": 50,
                "outlier_fraction": 0.0,
                "thresholds": {"max_outlier_fraction": 0.0, "min_reference_images": 10},
            }
        ),
        encoding="utf-8",
    )
    _write_geo_accuracy(geo_accuracy)

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        holdout_verify_report=holdout_verify,
        truth_coverage_report=truth_coverage,
        flight_check_report=flight_check,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        max_count_error_rate=0.01,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "validation_plan_report_present" in failed


def test_audit_release_requires_truth_quality_for_one_percent_claim(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    holdout_verify = tmp_path / "holdout_verify.json"
    validation_plan = tmp_path / "validation_plan.json"
    truth_coverage = tmp_path / "truth_coverage.json"
    flight_check = tmp_path / "flight_check.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "truth_count": 2000,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "count_error_rate": 0.0,
                },
                "statistics": {
                    "sample_support": {
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")
    holdout_verify.write_text(
        json.dumps({"status": "pass", "truth_count": 2000, "min_detectable_count_error_rate": 0.0005}),
        encoding="utf-8",
    )
    _write_validation_plan(validation_plan)
    truth_coverage.write_text(
        json.dumps(
            {
                "status": "pass",
                "truth_count": 2000,
                "cluster_count": 20,
                "cluster_truth_count": 60,
                "cluster_image_count": 10,
                "cluster_truth_fraction": 0.03,
            }
        ),
        encoding="utf-8",
    )
    flight_check.write_text(json.dumps({"status": "pass", "checks": []}), encoding="utf-8")
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "reference_image_count": 50,
                "outlier_fraction": 0.0,
                "thresholds": {"max_outlier_fraction": 0.0, "min_reference_images": 10},
            }
        ),
        encoding="utf-8",
    )
    _write_geo_accuracy(geo_accuracy)

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        holdout_verify_report=holdout_verify,
        validation_plan_report=validation_plan,
        truth_coverage_report=truth_coverage,
        flight_check_report=flight_check,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        max_count_error_rate=0.01,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "truth_quality_report_present" in failed


def test_audit_release_fails_bad_flight_check(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    holdout_verify = tmp_path / "holdout_verify.json"
    flight_check = tmp_path / "flight_check.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {"truth_count": 2000, "count_error_rate": 0.0},
                "statistics": {
                    "sample_support": {
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")
    holdout_verify.write_text(
        json.dumps({"status": "pass", "truth_count": 2000, "min_detectable_count_error_rate": 0.0005}),
        encoding="utf-8",
    )
    flight_check.write_text(
        json.dumps(
            {
                "status": "fail",
                "checks": [
                    {"name": "gsd_within_validated_range", "status": "fail"},
                    {"name": "front_overlap", "status": "pass"},
                ],
            }
        ),
        encoding="utf-8",
    )
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "reference_image_count": 50,
                "outlier_fraction": 0.0,
                "thresholds": {"max_outlier_fraction": 0.0, "min_reference_images": 10},
            }
        ),
        encoding="utf-8",
    )
    _write_geo_accuracy(geo_accuracy)

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        holdout_verify_report=holdout_verify,
        flight_check_report=flight_check,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        max_count_error_rate=0.01,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "flight_check_passed" in failed
    assert "flight_check_failures" in failed


def test_audit_release_fails_bad_domain_check(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    holdout_verify = tmp_path / "holdout_verify.json"
    flight_check = tmp_path / "flight_check.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {"truth_count": 2000, "count_error_rate": 0.0},
                "statistics": {
                    "sample_support": {
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")
    holdout_verify.write_text(
        json.dumps({"status": "pass", "truth_count": 2000, "min_detectable_count_error_rate": 0.0005}),
        encoding="utf-8",
    )
    flight_check.write_text(json.dumps({"status": "pass", "checks": []}), encoding="utf-8")
    domain_check.write_text(
        json.dumps(
            {
                "status": "fail",
                "reference_image_count": 5,
                "outlier_fraction": 0.4,
                "thresholds": {"max_outlier_fraction": 0.0, "min_reference_images": 10},
            }
        ),
        encoding="utf-8",
    )
    _write_geo_accuracy(geo_accuracy)

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        holdout_verify_report=holdout_verify,
        flight_check_report=flight_check,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        max_count_error_rate=0.01,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "domain_check_passed" in failed
    assert "domain_outlier_fraction" in failed
    assert "domain_reference_support" in failed


def test_audit_release_fails_bad_flight_log_when_provided(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    holdout_verify = tmp_path / "holdout_verify.json"
    flight_check = tmp_path / "flight_check.json"
    flight_log = tmp_path / "flight_log.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {"truth_count": 2000, "count_error_rate": 0.0},
                "statistics": {
                    "sample_support": {
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")
    holdout_verify.write_text(
        json.dumps({"status": "pass", "truth_count": 2000, "min_detectable_count_error_rate": 0.0005}),
        encoding="utf-8",
    )
    flight_check.write_text(json.dumps({"status": "pass", "checks": []}), encoding="utf-8")
    flight_log.write_text(
        json.dumps({"status": "fail", "summary": {"row_count": 5, "pass_count": 4, "warn_count": 0, "fail_count": 1}}),
        encoding="utf-8",
    )
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "reference_image_count": 50,
                "outlier_fraction": 0.0,
                "thresholds": {"max_outlier_fraction": 0.0, "min_reference_images": 10},
            }
        ),
        encoding="utf-8",
    )
    _write_geo_accuracy(geo_accuracy)

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        holdout_verify_report=holdout_verify,
        flight_check_report=flight_check,
        flight_log_report=flight_log,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        max_count_error_rate=0.01,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "flight_log_passed" in failed
    assert "flight_log_failures" in failed


def test_audit_release_fails_bad_geo_accuracy(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    holdout_verify = tmp_path / "holdout_verify.json"
    flight_check = tmp_path / "flight_check.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {"truth_count": 2000, "count_error_rate": 0.0},
                "statistics": {
                    "sample_support": {
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 45.0}}), encoding="utf-8")
    holdout_verify.write_text(
        json.dumps({"status": "pass", "truth_count": 2000, "min_detectable_count_error_rate": 0.0005}),
        encoding="utf-8",
    )
    flight_check.write_text(json.dumps({"status": "pass", "checks": []}), encoding="utf-8")
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "reference_image_count": 50,
                "outlier_fraction": 0.0,
                "thresholds": {"max_outlier_fraction": 0.0, "min_reference_images": 10},
            }
        ),
        encoding="utf-8",
    )
    _write_geo_accuracy(geo_accuracy, status="fail", rmse_m=1.8, recall=0.96, matched_count=1920)

    report = audit_release(
        tmp_path / "release_audit.json",
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        holdout_verify_report=holdout_verify,
        flight_check_report=flight_check,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        max_count_error_rate=0.01,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "geo_accuracy_passed" in failed
    assert "geo_rmse" in failed
    assert "geo_recall" in failed


def _write_geo_accuracy(
    path: Path,
    status: str = "pass",
    rmse_m: float = 0.25,
    p95_m: float = 0.5,
    recall: float = 1.0,
    truth_count: int = 2000,
    matched_count: int = 2000,
    max_rmse_m: float = 1.0,
    max_p95_m: float | None = None,
    min_recall: float = 0.99,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "thresholds": {
                    "tolerance_m": 1.0,
                    "max_rmse_m": max_rmse_m,
                    "max_p95_m": max_p95_m,
                    "min_recall": min_recall,
                },
                "metrics": {
                    "prediction_count": matched_count,
                    "truth_count": truth_count,
                    "matched_count": matched_count,
                    "unmatched_prediction_count": 0,
                    "unmatched_truth_count": truth_count - matched_count,
                    "precision": 1.0,
                    "recall": recall,
                    "f1": 1.0,
                    "rmse_m": rmse_m,
                    "p95_m": p95_m,
                },
                "checks": [],
            }
        ),
        encoding="utf-8",
    )


def _write_validation_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "plan",
                "targets": {"target_count_error_rate": 0.01},
                "operating_domain": {"condition_count": 2},
                "minimum_support": {
                    "truth_count": 1000,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "cluster_image_count": 10,
                    "plants_per_condition": 50,
                    "cluster_mats_per_condition": 10,
                    "cluster_truth_per_condition": 30,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_stratified_truth_coverage(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "stratum_count": 2,
                "failed_stratum_count": 0,
                "missing_metadata_count": 0,
                "thresholds": {
                    "min_truth_count": 50,
                    "min_cluster_count": 10,
                    "min_cluster_truth_count": 30,
                    "min_cluster_images": 5,
                    "min_cluster_truth_fraction": 0.2,
                },
                "strata": [
                    {"stratum": {"farm": "farm-a"}, "status": "pass"},
                    {"stratum": {"farm": "farm-b"}, "status": "pass"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_truth_quality(path: Path, status: str = "pass", issue_count: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "image_count": 10,
                "truth_count": 2000,
                "bounded_image_count": 10,
                "issue_count": issue_count,
                "checks": [
                    {"name": "truth_images_present", "status": "pass"},
                    {"name": "truth_quality_issues", "status": "pass" if issue_count == 0 else "fail"},
                ],
            }
        ),
        encoding="utf-8",
    )
