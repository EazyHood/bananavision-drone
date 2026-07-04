import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.stratified_truth_coverage import build_stratified_truth_coverage_report

runner = CliRunner()


def test_stratified_truth_coverage_fails_undercovered_condition(tmp_path: Path) -> None:
    truth = _write_truth(tmp_path, failing=True)
    metadata = _write_metadata(tmp_path)

    report = build_stratified_truth_coverage_report(
        truth,
        metadata,
        tmp_path / "stratified_truth_coverage.json",
        strata_keys=["farm", "flight_date", "gsd_band"],
        min_truth_count=2,
        min_cluster_count=1,
        min_cluster_truth_count=2,
        min_cluster_images=1,
        min_cluster_truth_fraction=0.5,
    )

    strata = {tuple(row["stratum"].values()): row for row in report["strata"]}
    assert report["status"] == "fail"
    assert report["failed_stratum_count"] == 1
    assert strata[("farm-a", "2026-07-01", "2cm")]["status"] == "pass"
    failed = strata[("farm-b", "2026-07-02", "2cm")]
    assert failed["status"] == "fail"
    assert failed["deficits"]["cluster_count"] == 1
    assert failed["deficits"]["cluster_truth_count"] == 2
    assert failed["deficits"]["cluster_image_count"] == 1
    assert failed["deficits"]["cluster_truth_fraction_cluster_truth"] == 1
    assert "annotate at least 1 additional grouped banana mat" in failed["actions"]
    assert (tmp_path / "stratified_truth_coverage.csv").exists()


def test_stratified_truth_coverage_passes_all_conditions(tmp_path: Path) -> None:
    truth = _write_truth(tmp_path, failing=False)
    metadata = _write_metadata(tmp_path)

    report = build_stratified_truth_coverage_report(
        truth,
        metadata,
        tmp_path / "stratified_truth_coverage.json",
        strata_keys=["farm"],
        min_truth_count=2,
        min_cluster_count=1,
        min_cluster_truth_count=2,
        min_cluster_images=1,
        min_cluster_truth_fraction=0.5,
    )

    assert report["status"] == "pass"
    assert report["stratum_count"] == 2
    assert report["failed_stratum_count"] == 0


def test_stratified_truth_coverage_cli_exits_nonzero_when_condition_fails(tmp_path: Path) -> None:
    truth = _write_truth(tmp_path, failing=True)
    metadata = _write_metadata(tmp_path)
    output = tmp_path / "stratified_truth_coverage.json"

    result = runner.invoke(
        app,
        [
            "stratified-truth-coverage",
            str(truth),
            str(metadata),
            "--output",
            str(output),
            "--strata",
            "farm",
            "--strata",
            "flight_date",
            "--strata",
            "gsd_band",
            "--min-truth-count",
            "2",
            "--min-cluster-count",
            "1",
            "--min-cluster-truth-count",
            "2",
            "--min-cluster-images",
            "1",
            "--min-cluster-truth-fraction",
            "0.5",
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.exit_code == 2
    assert payload["status"] == "fail"


def test_stratified_truth_coverage_fails_missing_metadata(tmp_path: Path) -> None:
    truth = _write_truth(tmp_path, failing=False)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("image,farm\nblock-a.jpg,farm-a\n", encoding="utf-8")

    report = build_stratified_truth_coverage_report(
        truth,
        metadata,
        tmp_path / "stratified_truth_coverage.json",
        strata_keys=["farm"],
        min_truth_count=1,
    )

    assert report["status"] == "fail"
    assert report["missing_metadata_count"] == 1


def _write_truth(tmp_path: Path, failing: bool) -> Path:
    path = tmp_path / "truth_manifest.json"
    block_b_centers = (
        [{"x": 20, "y": 20, "group_id": "mat-b"}, {"x": 22, "y": 20, "group_id": "mat-b"}]
        if not failing
        else [{"x": 20, "y": 20}, {"x": 22, "y": 20}]
    )
    payload = {
        "images": [
            {
                "image": "block-a.jpg",
                "centers": [
                    {"x": 10, "y": 10, "group_id": "mat-a"},
                    {"x": 12, "y": 10, "group_id": "mat-a"},
                ],
            },
            {"image": "block-b.jpg", "centers": block_b_centers},
        ]
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
