import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.flight_profile import (
    FlightEnvelope,
    FlightProfile,
    audit_flight_log,
    audit_flight_profile,
    estimate_gsd_cm,
)
from bananavision.models import InferenceConfig

runner = CliRunner()


def test_estimate_gsd_from_camera_geometry() -> None:
    gsd = estimate_gsd_cm(
        altitude_m=60.0,
        sensor_width_mm=13.2,
        focal_length_mm=8.8,
        image_width_px=5472,
    )

    assert round(gsd, 3) == 1.645


def test_audit_flight_profile_passes_valid_profile(tmp_path: Path) -> None:
    report = audit_flight_profile(
        FlightProfile(
            gsd_cm=2.0,
            front_overlap=75,
            side_overlap=72,
            speed_mps=4.0,
            exposure_ms=4.0,
        ),
        FlightEnvelope.from_config(InferenceConfig(gsd_cm=2.0)),
        tmp_path / "flight_check.json",
    )

    assert report["status"] == "pass"
    assert all(check["status"] == "pass" for check in report["checks"])
    assert (tmp_path / "flight_check.json").exists()


def test_audit_flight_profile_fails_off_domain_gsd_and_overlap() -> None:
    report = audit_flight_profile(
        FlightProfile(gsd_cm=3.2, front_overlap=55, side_overlap=0.60),
        FlightEnvelope.from_config(InferenceConfig(gsd_cm=2.0), max_gsd_drift_ratio=0.20),
    )

    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert report["status"] == "fail"
    assert "gsd_within_validated_range" in failed
    assert "front_overlap" in failed
    assert "side_overlap" in failed


def test_flight_check_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "flight_check_report.json"
    result = runner.invoke(
        app,
        [
            "flight-check",
            "--output",
            str(output),
            "--gsd-cm",
            "2.0",
            "--front-overlap",
            "75",
            "--side-overlap",
            "72",
            "--speed-mps",
            "4",
            "--exposure-ms",
            "4",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"


def test_flight_check_cli_fails_without_gsd(tmp_path: Path) -> None:
    output = tmp_path / "flight_check_report.json"
    result = runner.invoke(
        app,
        [
            "flight-check",
            "--output",
            str(output),
            "--front-overlap",
            "75",
            "--side-overlap",
            "72",
        ],
    )

    assert result.exit_code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "fail"


def test_audit_flight_log_passes_valid_rows(tmp_path: Path) -> None:
    log = tmp_path / "flight_log.csv"
    log.write_text(
        "image,gsd_cm,front_overlap,side_overlap,speed_mps,exposure_ms\n"
        "frame-1.jpg,2.0,75,72,4,4\n"
        "frame-2.jpg,2.1,76,73,4,4\n",
        encoding="utf-8",
    )

    report = audit_flight_log(
        log,
        FlightEnvelope.from_config(InferenceConfig(gsd_cm=2.0)),
        tmp_path / "flight_log_audit.json",
    )

    assert report["status"] == "pass"
    assert report["summary"]["row_count"] == 2
    assert report["summary"]["fail_count"] == 0
    assert (tmp_path / "flight_log_audit.csv").exists()


def test_audit_flight_log_fails_off_domain_rows(tmp_path: Path) -> None:
    log = tmp_path / "flight_log.csv"
    log.write_text(
        "image,gsd_cm,front_overlap,side_overlap,speed_mps,exposure_ms\n"
        "frame-1.jpg,3.2,55,60,12,8\n",
        encoding="utf-8",
    )

    report = audit_flight_log(log, FlightEnvelope.from_config(InferenceConfig(gsd_cm=2.0)))

    assert report["status"] == "fail"
    assert report["summary"]["fail_count"] == 1
    assert "gsd_within_validated_range" in report["rows"][0]["failed_checks"]
    assert "front_overlap" in report["rows"][0]["failed_checks"]


def test_flight_log_audit_cli_writes_report(tmp_path: Path) -> None:
    log = tmp_path / "flight_log.csv"
    output = tmp_path / "flight_log_audit.json"
    log.write_text(
        "image,gsd_cm,front_overlap,side_overlap,speed_mps,exposure_ms\n"
        "frame-1.jpg,2.0,75,72,4,4\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "flight-log-audit",
            str(log),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert output.with_suffix(".csv").exists()
