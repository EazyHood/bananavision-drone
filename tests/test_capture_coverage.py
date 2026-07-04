import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.capture_coverage import CaptureCoverageThresholds, audit_capture_coverage
from bananavision.cli import app

runner = CliRunner()


def test_capture_coverage_passes_complete_capture_log(tmp_path: Path) -> None:
    image_dir = _write_images(tmp_path, ["frame-1.jpg", "frame-2.jpg", "frame-3.jpg"])
    log = tmp_path / "capture_log.csv"
    log.write_text(
        "image,x,y,timestamp\n"
        "frame-1.jpg,0,0,0\n"
        "frame-2.jpg,10,0,2\n"
        "frame-3.jpg,20,0,4\n",
        encoding="utf-8",
    )

    report = audit_capture_coverage(
        log,
        tmp_path / "capture_coverage.json",
        image_dir=image_dir,
        thresholds=CaptureCoverageThresholds(
            min_images=3,
            max_position_gap_m=15,
            max_time_gap_s=3,
            require_image_files=True,
        ),
    )

    assert report["status"] == "pass"
    assert report["summary"]["row_count"] == 3
    assert report["summary"]["missing_image_count"] == 0
    assert report["summary"]["max_step_distance_m"] == 10
    assert (tmp_path / "capture_coverage.csv").exists()


def test_capture_coverage_fails_missing_image_and_position_gap(tmp_path: Path) -> None:
    image_dir = _write_images(tmp_path, ["frame-1.jpg", "frame-2.jpg"])
    log = tmp_path / "capture_log.csv"
    log.write_text(
        "image,x,y,timestamp\n"
        "frame-1.jpg,0,0,0\n"
        "frame-2.jpg,10,0,2\n"
        "frame-3.jpg,90,0,4\n",
        encoding="utf-8",
    )

    report = audit_capture_coverage(
        log,
        tmp_path / "capture_coverage.json",
        image_dir=image_dir,
        thresholds=CaptureCoverageThresholds(
            min_images=3,
            max_position_gap_m=20,
            require_image_files=True,
        ),
    )

    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert report["status"] == "fail"
    assert "image_files_present" in failed
    assert "position_gaps" in failed
    assert report["summary"]["missing_image_count"] == 1


def test_capture_coverage_supports_lonlat_steps(tmp_path: Path) -> None:
    log = tmp_path / "capture_log.csv"
    log.write_text(
        "image,lat,lon\n"
        "frame-1.jpg,4.000000,-74.000000\n"
        "frame-2.jpg,4.000090,-74.000000\n",
        encoding="utf-8",
    )

    report = audit_capture_coverage(
        log,
        tmp_path / "capture_coverage.json",
        thresholds=CaptureCoverageThresholds(max_position_gap_m=20, require_image_files=False),
    )

    assert report["status"] == "warn"
    assert 9 <= report["summary"]["max_step_distance_m"] <= 11


def test_capture_coverage_cli_writes_report(tmp_path: Path) -> None:
    image_dir = _write_images(tmp_path, ["frame-1.jpg", "frame-2.jpg"])
    log = tmp_path / "capture_log.csv"
    output = tmp_path / "capture_coverage.json"
    log.write_text(
        "image,x,y\n"
        "frame-1.jpg,0,0\n"
        "frame-2.jpg,10,0\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "capture-coverage",
            str(log),
            "--output",
            str(output),
            "--images",
            str(image_dir),
            "--require-image-files",
            "--min-images",
            "2",
            "--max-position-gap-m",
            "15",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "warn"
    assert output.with_suffix(".csv").exists()


def _write_images(tmp_path: Path, names: list[str]) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in names:
        (image_dir / name).write_bytes(b"image")
    return image_dir
