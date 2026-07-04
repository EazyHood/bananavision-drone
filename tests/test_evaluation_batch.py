import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.evaluation import acceptance_passed, evaluate_path, write_batch_evaluation_report
from bananavision.holdout import lock_holdout
from bananavision.metrics import PointMetrics
from bananavision.models import InferenceConfig
from bananavision.synthetic import generate_scene

runner = CliRunner()


def test_evaluate_path_with_truth_manifest(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_a = image_dir / "a.jpg"
    image_b = image_dir / "b.jpg"
    truth_a = tmp_path / "a.truth.json"
    truth_b = tmp_path / "b.truth.json"
    generate_scene(image_a, truth_a, width=220, height=160, plant_count=5, seed=1)
    generate_scene(image_b, truth_b, width=220, height=160, plant_count=5, seed=2)
    manifest = tmp_path / "truth_manifest.json"
    centers_a = json.loads(truth_a.read_text(encoding="utf-8"))["centers"]
    centers_b = json.loads(truth_b.read_text(encoding="utf-8"))["centers"]
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {"image": "a.jpg", "centers": centers_a},
                    {"image": "b.jpg", "centers": centers_b},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)
    report = evaluate_path(image_dir, manifest, config, tolerance_px=80, output_dir=tmp_path / "out")
    assert report["metrics"]["images"] == 2
    assert len(report["images"]) == 2
    output = write_batch_evaluation_report(
        tmp_path / "acceptance_batch_report.json",
        report,
        {
            "tolerance_px": 80,
            "max_count_error_rate": 10.0,
            "min_precision": 0.0,
            "min_recall": 0.0,
            "min_f1": 0.0,
            "confidence_level": 0.95,
        },
    )
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "statistics" in payload
    assert payload["statistics"]["sample_support"]["truth_count"] == 10
    assert payload["passed"] is True


def test_write_batch_evaluation_report_can_fail_on_sample_support(tmp_path: Path) -> None:
    report = {
        "input_path": "images",
        "truth_path": "truth.json",
        "tolerance_px": 24,
        "metrics": {
            "images": 1,
            "truth_count": 10,
            "prediction_count": 10,
            "true_positives": 10,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "count_error": 0,
            "count_error_rate": 0.0,
            "mean_abs_image_count_error_rate": 0.0,
            "worst_image_count_error_rate": 0.0,
        },
        "images": [{"metrics": {"count_error_rate": 0.0}}],
    }
    output = write_batch_evaluation_report(
        tmp_path / "acceptance_batch_report.json",
        report,
        {
            "tolerance_px": 24,
            "max_count_error_rate": 0.01,
            "min_precision": 0.99,
            "min_recall": 0.99,
            "min_f1": 0.99,
            "min_truth_count": 100,
        },
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False


def test_acceptance_can_fail_on_cluster_split_support() -> None:
    metrics = PointMetrics(
        true_positives=1,
        false_positives=1,
        false_negatives=1,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        count_error=0,
        count_error_rate=0.0,
        cluster_count=1,
        cluster_truth_count=2,
        cluster_matched_count=1,
        cluster_recall=0.5,
        fully_detected_cluster_count=0,
        fully_detected_cluster_rate=0.0,
        under_split_cluster_count=1,
        over_split_cluster_count=1,
        cluster_extra_prediction_count=1,
    )

    assert not acceptance_passed(
        metrics,
        max_count_error_rate=0.01,
        min_precision=0.0,
        min_recall=0.0,
        min_f1=0.0,
        min_cluster_recall=0.99,
        min_cluster_full_detection_rate=0.99,
        min_cluster_count=1,
    )


def test_write_batch_evaluation_report_can_gate_cluster_split(tmp_path: Path) -> None:
    report = {
        "input_path": "images",
        "truth_path": "truth.json",
        "tolerance_px": 24,
        "metrics": {
            "images": 1,
            "truth_count": 3,
            "prediction_count": 3,
            "true_positives": 2,
            "false_positives": 1,
            "false_negatives": 1,
            "precision": 0.666667,
            "recall": 0.666667,
            "f1": 0.666667,
            "count_error": 0,
            "count_error_rate": 0.0,
            "mean_abs_image_count_error_rate": 0.0,
            "worst_image_count_error_rate": 0.0,
            "cluster_count": 1,
            "cluster_truth_count": 3,
            "cluster_matched_count": 2,
            "cluster_recall": 0.666667,
            "fully_detected_cluster_count": 0,
            "fully_detected_cluster_rate": 0.0,
            "under_split_cluster_count": 1,
            "over_split_cluster_count": 1,
            "cluster_extra_prediction_count": 1,
        },
        "images": [{"metrics": {"count_error_rate": 0.0}}],
    }

    output = write_batch_evaluation_report(
        tmp_path / "acceptance_batch_report.json",
        report,
        {
            "tolerance_px": 24,
            "max_count_error_rate": 0.01,
            "min_precision": 0.0,
            "min_recall": 0.0,
            "min_f1": 0.0,
            "min_cluster_recall": 0.99,
            "min_cluster_full_detection_rate": 0.99,
            "min_cluster_count": 1,
        },
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["statistics"]["sample_support"]["cluster_truth_count"] == 3
    assert payload["statistics"]["cluster_recall_wilson_ci"]["successes"] == 2


def test_acceptance_batch_cli_verifies_holdout_lock(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "scene.truth.json"
    lock = tmp_path / "holdout_lock.json"
    output_dir = tmp_path / "acceptance"
    generate_scene(image, truth, width=220, height=160, plant_count=5, seed=1)
    lock_holdout(image, truth, lock, target_count_error_rate=1.0)

    result = runner.invoke(
        app,
        [
            "acceptance-batch",
            str(image),
            str(truth),
            "--output",
            str(output_dir),
            "--holdout-lock",
            str(lock),
            "--max-count-error-rate",
            "10",
            "--min-precision",
            "0",
            "--min-recall",
            "0",
            "--min-f1",
            "0",
        ],
    )

    assert result.exit_code == 0
    report = json.loads((output_dir / "acceptance_batch_report.json").read_text(encoding="utf-8"))
    assert report["holdout_verification"]["status"] == "pass"
    assert (output_dir / "holdout_verify.json").exists()
