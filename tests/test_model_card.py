import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.model_card import build_model_card

runner = CliRunner()


def test_build_model_card_from_evidence(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    mission_quality = tmp_path / "mission_quality.json"
    prediction_quality = tmp_path / "prediction_quality.json"
    flight_log = tmp_path / "flight_log.json"
    domain_check = tmp_path / "domain_check.json"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    validation_plan = tmp_path / "validation_plan.json"
    stratified_acceptance = tmp_path / "stratified_acceptance.json"
    stratified_truth_coverage = tmp_path / "stratified_truth_coverage.json"
    truth_quality = tmp_path / "truth_quality.json"
    truth_coverage = tmp_path / "truth_coverage.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "images": 2,
                    "truth_count": 18,
                    "prediction_count": 12,
                    "count_error_rate": 0.333333,
                    "precision": 0.666667,
                    "recall": 0.444444,
                    "f1": 0.533333,
                },
                "statistics": {
                    "sample_support": {"images": 2, "truth_count": 18, "min_detectable_count_error_rate": 0.055556},
                    "precision_wilson_ci": {"lower": 0.39, "upper": 0.86},
                    "recall_wilson_ci": {"lower": 0.24, "upper": 0.66},
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"latency_ms": {"median": 10.0, "p95": 20.0, "max": 30.0}}), encoding="utf-8")
    mission_quality.write_text(json.dumps({"status": "pass", "image_count": 2}), encoding="utf-8")
    prediction_quality.write_text(
        json.dumps({"status": "warn", "detection_count": 12, "review_detection_count": 2}),
        encoding="utf-8",
    )
    flight_log.write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": {"row_count": 2, "pass_count": 2, "warn_count": 0, "fail_count": 0, "max_gsd_cm": 2.1},
            }
        ),
        encoding="utf-8",
    )
    domain_check.write_text(
        json.dumps(
            {
                "status": "pass",
                "image_count": 2,
                "reference_image_count": 20,
                "outlier_count": 0,
                "outlier_fraction": 0.0,
            }
        ),
        encoding="utf-8",
    )
    geo_accuracy.write_text(
        json.dumps(
            {
                "status": "pass",
                "metrics": {
                    "truth_count": 18,
                    "matched_count": 18,
                    "rmse_m": 0.35,
                    "p95_m": 0.7,
                    "recall": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    validation_plan.write_text(
        json.dumps(
            {
                "status": "plan",
                "targets": {"target_count_error_rate": 0.01},
                "minimum_support": {
                    "truth_count": 100,
                    "cluster_count": 10,
                    "cluster_truth_count": 30,
                    "cluster_image_count": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_stratified_acceptance(stratified_acceptance)
    _write_stratified_truth_coverage(stratified_truth_coverage)
    truth_quality.write_text(
        json.dumps(
            {
                "status": "pass",
                "image_count": 2,
                "truth_count": 18,
                "bounded_image_count": 2,
                "issue_count": 0,
            }
        ),
        encoding="utf-8",
    )
    truth_coverage.write_text(
        json.dumps(
            {
                "status": "pass",
                "truth_count": 18,
                "cluster_count": 4,
                "cluster_truth_count": 12,
                "cluster_image_count": 2,
                "cluster_truth_fraction": 0.666667,
            }
        ),
        encoding="utf-8",
    )

    output = build_model_card(
        tmp_path / "MODEL_CARD.generated.md",
        model_name="Banana test model",
        version="v1",
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        mission_quality_report=mission_quality,
        prediction_quality_report=prediction_quality,
        flight_log_report=flight_log,
        domain_check_report=domain_check,
        geo_accuracy_report=geo_accuracy,
        validation_plan_report=validation_plan,
        stratified_acceptance_report=stratified_acceptance,
        truth_quality_report=truth_quality,
        truth_coverage_report=truth_coverage,
        stratified_truth_coverage_report=stratified_truth_coverage,
    )

    text = output.read_text(encoding="utf-8")
    assert "# Model Card: Banana test model" in text
    assert "accepted but 1% claim not proven" in text
    assert "Precision Wilson CI" in text
    assert "Mission quality: status=pass" in text
    assert "Flight log audit: status=pass" in text
    assert "Domain check: status=pass" in text
    assert "Geo accuracy: status=pass" in text
    assert "Validation plan: status=plan" in text
    assert "Stratified acceptance: status=pass" in text
    assert "Stratified truth coverage: status=pass" in text
    assert "Truth quality: status=pass" in text
    assert "Truth coverage: status=pass" in text
    assert "rmse_m=0.3500" in text
    assert "P95 latency ms: 20.0000" in text


def test_build_model_card_can_mark_one_percent_supported(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    validation_plan = tmp_path / "validation_plan.json"
    stratified_acceptance = tmp_path / "stratified_acceptance.json"
    stratified_truth_coverage = tmp_path / "stratified_truth_coverage.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "images": 40,
                    "truth_count": 2000,
                    "prediction_count": 2000,
                    "count_error_rate": 0.0,
                    "precision": 0.995,
                    "recall": 0.995,
                    "f1": 0.995,
                    "cluster_count": 30,
                    "cluster_truth_count": 90,
                    "cluster_recall": 1.0,
                    "fully_detected_cluster_rate": 1.0,
                },
                "statistics": {
                    "sample_support": {
                        "images": 40,
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    validation_plan.write_text(
        json.dumps(
            {
                "status": "plan",
                "targets": {"target_count_error_rate": 0.01},
                "minimum_support": {
                    "truth_count": 1000,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "cluster_image_count": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_stratified_acceptance(stratified_acceptance)
    _write_stratified_truth_coverage(stratified_truth_coverage)

    output = build_model_card(
        tmp_path / "MODEL_CARD.generated.md",
        model_name="Banana production model",
        version="v1",
        acceptance_report=acceptance,
        validation_plan_report=validation_plan,
        stratified_acceptance_report=stratified_acceptance,
        stratified_truth_coverage_report=stratified_truth_coverage,
    )

    assert "1% claim supported by provided reports" in output.read_text(encoding="utf-8")


def test_model_card_cli_accepts_stratified_acceptance_report(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    validation_plan = tmp_path / "validation_plan.json"
    stratified_acceptance = tmp_path / "stratified_acceptance.json"
    stratified_truth_coverage = tmp_path / "stratified_truth_coverage.json"
    output = tmp_path / "MODEL_CARD.generated.md"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "images": 40,
                    "truth_count": 2000,
                    "prediction_count": 2000,
                    "count_error_rate": 0.0,
                    "precision": 0.995,
                    "recall": 0.995,
                    "f1": 0.995,
                    "cluster_count": 30,
                    "cluster_truth_count": 90,
                    "cluster_recall": 1.0,
                    "fully_detected_cluster_rate": 1.0,
                },
                "statistics": {
                    "sample_support": {
                        "images": 40,
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    validation_plan.write_text(
        json.dumps(
            {
                "status": "plan",
                "targets": {"target_count_error_rate": 0.01},
                "minimum_support": {
                    "truth_count": 1000,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "cluster_image_count": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_stratified_acceptance(stratified_acceptance)
    _write_stratified_truth_coverage(stratified_truth_coverage)

    result = runner.invoke(
        app,
        [
            "model-card",
            "--output",
            str(output),
            "--model-name",
            "Banana CLI model",
            "--version",
            "v1",
            "--acceptance-report",
            str(acceptance),
            "--validation-plan-report",
            str(validation_plan),
            "--stratified-acceptance-report",
            str(stratified_acceptance),
            "--stratified-truth-coverage-report",
            str(stratified_truth_coverage),
        ],
    )

    assert result.exit_code == 0
    assert "Stratified acceptance: status=pass" in output.read_text(encoding="utf-8")


def test_build_model_card_does_not_mark_one_percent_without_stratified_acceptance(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    validation_plan = tmp_path / "validation_plan.json"
    stratified_truth_coverage = tmp_path / "stratified_truth_coverage.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "images": 40,
                    "truth_count": 2000,
                    "prediction_count": 2000,
                    "count_error_rate": 0.0,
                    "precision": 0.995,
                    "recall": 0.995,
                    "f1": 0.995,
                    "cluster_count": 30,
                    "cluster_truth_count": 90,
                    "cluster_recall": 1.0,
                    "fully_detected_cluster_rate": 1.0,
                },
                "statistics": {
                    "sample_support": {
                        "images": 40,
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    validation_plan.write_text(
        json.dumps(
            {
                "status": "plan",
                "targets": {"target_count_error_rate": 0.01},
                "minimum_support": {
                    "truth_count": 1000,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "cluster_image_count": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_stratified_truth_coverage(stratified_truth_coverage)

    output = build_model_card(
        tmp_path / "MODEL_CARD.generated.md",
        model_name="Banana production model",
        version="v1",
        acceptance_report=acceptance,
        validation_plan_report=validation_plan,
        stratified_truth_coverage_report=stratified_truth_coverage,
    )

    assert "accepted but stratified acceptance not provided" in output.read_text(encoding="utf-8")


def test_build_model_card_does_not_mark_one_percent_without_stratified_truth_coverage(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    validation_plan = tmp_path / "validation_plan.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "images": 40,
                    "truth_count": 2000,
                    "prediction_count": 2000,
                    "count_error_rate": 0.0,
                    "precision": 0.995,
                    "recall": 0.995,
                    "f1": 0.995,
                    "cluster_count": 30,
                    "cluster_truth_count": 90,
                    "cluster_recall": 1.0,
                    "fully_detected_cluster_rate": 1.0,
                },
                "statistics": {
                    "sample_support": {
                        "images": 40,
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    validation_plan.write_text(
        json.dumps(
            {
                "status": "plan",
                "targets": {"target_count_error_rate": 0.01},
                "minimum_support": {
                    "truth_count": 1000,
                    "cluster_count": 20,
                    "cluster_truth_count": 60,
                    "cluster_image_count": 10,
                },
            }
        ),
        encoding="utf-8",
    )

    output = build_model_card(
        tmp_path / "MODEL_CARD.generated.md",
        model_name="Banana production model",
        version="v1",
        acceptance_report=acceptance,
        validation_plan_report=validation_plan,
    )

    assert "accepted but stratified truth coverage not provided" in output.read_text(encoding="utf-8")


def test_build_model_card_does_not_mark_one_percent_without_validation_plan(tmp_path: Path) -> None:
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "metrics": {
                    "images": 40,
                    "truth_count": 2000,
                    "prediction_count": 2000,
                    "count_error_rate": 0.0,
                    "precision": 0.995,
                    "recall": 0.995,
                    "f1": 0.995,
                },
                "statistics": {
                    "sample_support": {
                        "images": 40,
                        "truth_count": 2000,
                        "min_detectable_count_error_rate": 0.0005,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output = build_model_card(
        tmp_path / "MODEL_CARD.generated.md",
        model_name="Banana production model",
        version="v1",
        acceptance_report=acceptance,
    )

    assert "accepted but 1% validation plan not provided" in output.read_text(encoding="utf-8")


def _write_stratified_acceptance(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "stratum_count": 2,
                "failed_stratum_count": 0,
                "missing_metadata_count": 0,
                "thresholds": {"max_count_error_rate": 0.01},
                "strata": [
                    {"stratum": {"farm": "farm-a", "flight_date": "2026-07-01"}, "passed": True},
                    {"stratum": {"farm": "farm-b", "flight_date": "2026-07-02"}, "passed": True},
                ],
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
                "thresholds": {"min_truth_count": 50},
                "strata": [
                    {"stratum": {"farm": "farm-a", "flight_date": "2026-07-01"}, "status": "pass"},
                    {"stratum": {"farm": "farm-b", "flight_date": "2026-07-02"}, "status": "pass"},
                ],
            }
        ),
        encoding="utf-8",
    )
