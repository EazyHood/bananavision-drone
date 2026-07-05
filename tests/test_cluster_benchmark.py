import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.cluster_benchmark import run_cluster_benchmark
from bananavision.models import InferenceConfig

runner = CliRunner()


def test_run_cluster_benchmark_writes_evidence(tmp_path: Path) -> None:
    config = InferenceConfig(
        gsd_cm=2.0,
        expected_crown_diameter_m=0.55,
        min_component_area_px=20,
        rgb_threshold_quantile=0.78,
    )

    report = run_cluster_benchmark(tmp_path / "cluster_benchmark", config, scenes=2)

    assert report["status"] == "pass"
    assert report["truth_coverage"]["cluster_count"] == 6
    assert report["metrics"]["cluster_count"] == 6
    # Regression guard for the base RGB detector on SYNTHETIC scenes. The exact
    # value is sensitive to the numpy/scikit-image version (splitting of adjacent
    # clumps), so the threshold is a wide floor, not a field metric (the real
    # model is the one used in production).
    assert report["metrics"]["cluster_recall"] >= 0.80
    assert report["metrics"]["fully_detected_cluster_rate"] >= 0.66
    assert (tmp_path / "cluster_benchmark" / "cluster_benchmark_report.json").exists()
    assert (tmp_path / "cluster_benchmark" / "cluster_acceptance_report.json").exists()
    assert (tmp_path / "cluster_benchmark" / "truth_coverage_report.json").exists()


def test_cluster_benchmark_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "cluster_benchmark"

    result = runner.invoke(
        app,
        [
            "cluster-benchmark",
            "--output",
            str(output),
            "--scenes",
            "1",
            "--max-count-error-rate",
            "1",
            "--max-mean-image-count-error-rate",
            "1",
            "--min-precision",
            "0",
            "--min-recall",
            "0",
            "--min-f1",
            "0",
            "--min-cluster-recall",
            "0",
            "--min-cluster-full-detection-rate",
            "0",
        ],
    )

    payload = json.loads((output / "cluster_benchmark_report.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert payload["passed"] is True
    assert payload["truth_coverage"]["cluster_count"] == 3
