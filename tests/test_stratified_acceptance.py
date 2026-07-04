import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.stratified_acceptance import build_stratified_acceptance_report

runner = CliRunner()


def test_stratified_acceptance_fails_bad_field_condition(tmp_path: Path) -> None:
    acceptance = _write_acceptance(tmp_path, failing=True)
    metadata = _write_metadata(tmp_path)

    report = build_stratified_acceptance_report(
        acceptance,
        metadata,
        tmp_path / "stratified_acceptance.json",
        strata_keys=["farm", "flight_date", "gsd_band"],
        max_count_error_rate=0.01,
        min_precision=0.99,
        min_recall=0.99,
        min_f1=0.99,
        min_truth_count=2,
    )

    strata = {tuple(row["stratum"].values()): row for row in report["strata"]}
    assert report["status"] == "fail"
    assert report["failed_stratum_count"] == 1
    assert strata[("farm-a", "2026-07-01", "2cm")]["passed"] is True
    assert strata[("farm-b", "2026-07-02", "2cm")]["passed"] is False
    assert (tmp_path / "stratified_acceptance.csv").exists()


def test_stratified_acceptance_passes_all_conditions(tmp_path: Path) -> None:
    acceptance = _write_acceptance(tmp_path, failing=False)
    metadata = _write_metadata(tmp_path)

    report = build_stratified_acceptance_report(
        acceptance,
        metadata,
        tmp_path / "stratified_acceptance.json",
        strata_keys=["farm"],
        max_count_error_rate=0.01,
        min_precision=0.99,
        min_recall=0.99,
        min_f1=0.99,
        min_truth_count=2,
    )

    assert report["status"] == "pass"
    assert report["failed_stratum_count"] == 0


def test_stratified_acceptance_cli_exits_nonzero_when_condition_fails(tmp_path: Path) -> None:
    acceptance = _write_acceptance(tmp_path, failing=True)
    metadata = _write_metadata(tmp_path)
    output = tmp_path / "stratified_acceptance.json"

    result = runner.invoke(
        app,
        [
            "stratified-acceptance",
            str(acceptance),
            str(metadata),
            "--output",
            str(output),
            "--strata",
            "farm",
            "--strata",
            "flight_date",
            "--strata",
            "gsd_band",
            "--max-count-error-rate",
            "0.01",
            "--min-precision",
            "0.99",
            "--min-recall",
            "0.99",
            "--min-f1",
            "0.99",
            "--min-truth-count",
            "2",
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.exit_code == 2
    assert payload["status"] == "fail"


def test_stratified_acceptance_fails_missing_metadata(tmp_path: Path) -> None:
    acceptance = _write_acceptance(tmp_path, failing=False)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("image,farm\nblock-a.jpg,farm-a\n", encoding="utf-8")

    report = build_stratified_acceptance_report(
        acceptance,
        metadata,
        tmp_path / "stratified_acceptance.json",
        strata_keys=["farm"],
        max_count_error_rate=0.01,
        min_precision=0.99,
        min_recall=0.99,
        min_f1=0.99,
    )

    assert report["status"] == "fail"
    assert report["missing_metadata_count"] == 1


def _write_acceptance(tmp_path: Path, failing: bool) -> Path:
    path = tmp_path / "acceptance_batch_report.json"
    block_b_metrics = _metrics(tp=1, fp=0, fn=1, truth=2, predictions=1) if failing else _metrics()
    payload = {
        "passed": not failing,
        "metrics": {
            "images": 2,
            "truth_count": 4,
            "prediction_count": 3 if failing else 4,
            "true_positives": 3 if failing else 4,
            "false_positives": 0,
            "false_negatives": 1 if failing else 0,
            "precision": 1.0,
            "recall": 0.75 if failing else 1.0,
            "f1": 0.857143 if failing else 1.0,
            "count_error": -1 if failing else 0,
            "count_error_rate": 0.25 if failing else 0.0,
            "mean_abs_image_count_error_rate": 0.25 if failing else 0.0,
            "worst_image_count_error_rate": 0.5 if failing else 0.0,
        },
        "images": [
            {
                "image": "block-a.jpg",
                "truth_count": 2,
                "prediction_count": 2,
                "metrics": _metrics(),
            },
            {
                "image": "block-b.jpg",
                "truth_count": 2,
                "prediction_count": 1 if failing else 2,
                "metrics": block_b_metrics,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_metadata(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "image,farm,flight_date,gsd_band,cultivar\n"
        "block-a.jpg,farm-a,2026-07-01,2cm,cavendish\n"
        "block-b.jpg,farm-b,2026-07-02,2cm,cavendish\n",
        encoding="utf-8",
    )
    return path


def _metrics(tp: int = 2, fp: int = 0, fn: int = 0, truth: int = 2, predictions: int = 2) -> dict[str, float | int]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_error": predictions - truth,
        "count_error_rate": abs(predictions - truth) / truth,
        "cluster_count": 0,
        "cluster_truth_count": 0,
        "cluster_matched_count": 0,
        "cluster_recall": 0.0,
        "fully_detected_cluster_count": 0,
        "fully_detected_cluster_rate": 0.0,
        "under_split_cluster_count": 0,
        "over_split_cluster_count": 0,
        "cluster_extra_prediction_count": 0,
    }
