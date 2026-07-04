import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.truth_quality import TruthQualityThresholds, audit_truth_quality

runner = CliRunner()


def test_truth_quality_passes_clean_grouped_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "truth_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image": "a.jpg",
                        "width": 100,
                        "height": 100,
                        "centers": [
                            {"x": 10, "y": 10, "group_id": "mat-a"},
                            {"x": 18, "y": 10, "group_id": "mat-a"},
                            {"x": 26, "y": 10, "group_id": "mat-a"},
                            {"x": 80, "y": 80},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = audit_truth_quality(manifest, tmp_path / "truth_quality.json")

    assert report["status"] == "pass"
    assert report["truth_count"] == 4
    assert report["issue_count"] == 0
    assert (tmp_path / "truth_quality.csv").exists()


def test_truth_quality_fails_duplicate_singleton_oversize_and_bounds(tmp_path: Path) -> None:
    manifest = tmp_path / "truth_manifest.json"
    centers = [
        {"x": 10, "y": 10, "group_id": "mat-a"},
        {"x": 10.5, "y": 10.5, "group_id": "mat-a"},
        {"x": 40, "y": 40, "group_id": "singleton"},
        {"x": 60, "y": 60, "group_id": "too-big"},
        {"x": 62, "y": 60, "group_id": "too-big"},
        {"x": 64, "y": 60, "group_id": "too-big"},
        {"x": 66, "y": 60, "group_id": "too-big"},
        {"x": 68, "y": 60, "group_id": "too-big"},
        {"x": 120, "y": 50},
    ]
    manifest.write_text(
        json.dumps({"images": [{"image": "a.jpg", "width": 100, "height": 100, "centers": centers}]}),
        encoding="utf-8",
    )

    report = audit_truth_quality(
        manifest,
        tmp_path / "truth_quality.json",
        thresholds=TruthQualityThresholds(min_center_distance_px=2.0, max_group_size=4),
    )

    issue_types = {issue["type"] for issue in report["issues"]}
    assert report["status"] == "fail"
    assert "duplicate_or_too_close_center" in issue_types
    assert "singleton_group" in issue_types
    assert "oversized_group" in issue_types
    assert "center_out_of_bounds" in issue_types


def test_truth_quality_uses_image_dimensions_when_images_are_provided(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    truth_dir = tmp_path / "truth"
    image_dir.mkdir()
    truth_dir.mkdir()
    Image.new("RGB", (50, 40), "green").save(image_dir / "a.jpg")
    (truth_dir / "a.truth.json").write_text(
        json.dumps({"centers": [{"x": 49, "y": 39}, {"x": 51, "y": 10}]}),
        encoding="utf-8",
    )

    report = audit_truth_quality(truth_dir, tmp_path / "truth_quality.json", image_path=image_dir)

    assert report["status"] == "fail"
    assert report["bounded_image_count"] == 1
    assert {issue["type"] for issue in report["issues"]} == {"center_out_of_bounds"}


def test_truth_quality_cli_exits_nonzero_on_issues(tmp_path: Path) -> None:
    manifest = tmp_path / "truth_manifest.json"
    output = tmp_path / "truth_quality.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image": "a.jpg",
                        "width": 100,
                        "height": 100,
                        "centers": [
                            {"x": 10, "y": 10, "group_id": "mat-a"},
                            {"x": 10.5, "y": 10.5, "group_id": "mat-a"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["truth-quality", str(manifest), "--output", str(output)])

    assert result.exit_code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "fail"
