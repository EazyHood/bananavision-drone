import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.truth_coverage import audit_truth_coverage, write_truth_coverage_report

runner = CliRunner()


def test_truth_coverage_passes_clustered_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "truth_manifest.json")

    report = audit_truth_coverage(
        manifest,
        min_truth_count=6,
        min_cluster_count=2,
        min_cluster_truth_count=5,
        min_cluster_images=2,
        min_cluster_truth_fraction=0.8,
    )
    output = write_truth_coverage_report(report, tmp_path / "truth_coverage.json")

    assert report["status"] == "pass"
    assert report["truth_count"] == 6
    assert report["cluster_count"] == 2
    assert report["cluster_truth_count"] == 5
    assert report["cluster_image_count"] == 2
    assert report["max_cluster_size"] == 3
    assert output.exists()


def test_truth_coverage_fails_when_holdout_lacks_grouped_banana_mats(tmp_path: Path) -> None:
    manifest = tmp_path / "truth_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image": "a.jpg",
                        "centers": [[1, 2], [3, 4], [5, 6]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_truth_coverage(manifest, min_truth_count=3, min_cluster_count=1, min_cluster_truth_count=2)
    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}

    assert report["status"] == "fail"
    assert "cluster_count" in failed
    assert "cluster_truth_count" in failed


def test_truth_coverage_cli_exits_nonzero_on_failed_threshold(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "truth_manifest.json")
    output = tmp_path / "truth_coverage.json"

    result = runner.invoke(
        app,
        [
            "truth-coverage",
            str(manifest),
            "--output",
            str(output),
            "--min-cluster-count",
            "3",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"


def _write_manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image": "a.jpg",
                        "centers": [
                            {"x": 1, "y": 2, "group_id": "mat-a"},
                            {"x": 3, "y": 4, "group_id": "mat-a"},
                            {"x": 5, "y": 6, "group_id": "mat-a"},
                        ],
                    },
                    {
                        "image": "b.jpg",
                        "centers": [
                            {"x": 7, "y": 8, "group_id": "mat-b"},
                            {"x": 9, "y": 10, "group_id": "mat-b"},
                            {"x": 11, "y": 12},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path
