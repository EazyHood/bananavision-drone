import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.validation_plan import build_validation_plan, write_validation_plan

runner = CliRunner()


def test_validation_plan_computes_one_percent_support() -> None:
    report = build_validation_plan(
        target_count_error_rate=0.01,
        target_cluster_recall_loss=0.01,
        target_cluster_full_detection_loss=0.05,
        farms=1,
        flight_dates=1,
        gsd_bands=1,
        cultivars=1,
        min_plants_per_condition=0,
        min_cluster_mats_per_condition=0,
        min_cluster_truth_fraction=0.0,
    )

    assert report["minimum_support"]["truth_count"] == 100
    assert report["minimum_support"]["cluster_truth_count"] == 100
    assert report["minimum_support"]["cluster_count"] == 20
    assert report["resolution"]["min_detectable_count_error_rate"] == 0.01
    assert report["recommended_acceptance_args"]["--min-truth-count"] == 100
    assert report["recommended_acceptance_args"]["--min-cluster-recall"] == 0.99


def test_validation_plan_keeps_condition_coverage() -> None:
    report = build_validation_plan(
        target_count_error_rate=0.01,
        farms=3,
        flight_dates=3,
        gsd_bands=2,
        cultivars=1,
        min_plants_per_condition=50,
        min_cluster_mats_per_condition=10,
    )

    assert report["operating_domain"]["condition_count"] == 18
    assert report["minimum_support"]["truth_count"] == 900
    assert report["minimum_support"]["cluster_count"] == 180
    assert report["minimum_support"]["cluster_truth_count"] == 180
    assert report["minimum_support"]["cluster_image_count"] == 18
    assert report["minimum_support"]["plants_per_condition"] == 50
    assert report["recommended_acceptance_args"]["--min-cluster-truth-fraction"] == 0.20


def test_validation_plan_write_report(tmp_path: Path) -> None:
    report = build_validation_plan(farms=1, flight_dates=1, gsd_bands=1)
    output = write_validation_plan(report, tmp_path / "validation_plan.json")

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "plan"


def test_validation_plan_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError):
        build_validation_plan(target_count_error_rate=0)
    with pytest.raises(ValueError):
        build_validation_plan(min_cluster_truth_fraction=1)


def test_validation_plan_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "validation_plan.json"

    result = runner.invoke(
        app,
        [
            "validation-plan",
            "--output",
            str(output),
            "--target-count-error-rate",
            "0.01",
            "--farms",
            "1",
            "--flight-dates",
            "1",
            "--gsd-bands",
            "1",
            "--min-plants-per-condition",
            "0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["minimum_support"]["truth_count"] == 100


def test_validation_plan_cli_exits_nonzero_for_invalid_args(tmp_path: Path) -> None:
    output = tmp_path / "validation_plan.json"

    result = runner.invoke(
        app,
        [
            "validation-plan",
            "--output",
            str(output),
            "--target-count-error-rate",
            "0",
        ],
    )

    assert result.exit_code == 2
    assert not output.exists()
