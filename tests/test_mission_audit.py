import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.mission_audit import audit_mission_delivery

runner = CliRunner()


def test_audit_mission_delivery_passes_complete_evidence(tmp_path: Path) -> None:
    evidence = _write_mission_evidence(tmp_path)

    report = audit_mission_delivery(
        tmp_path / "mission_audit.json",
        run_manifest=evidence["run"],
        mission_quality_report=evidence["mission_quality"],
        prediction_quality_report=evidence["prediction_quality"],
        flight_check_report=evidence["flight_check"],
        domain_check_report=evidence["domain_check"],
        flight_log_report=evidence["flight_log"],
        capture_coverage_report=evidence["capture_coverage"],
        geo_accuracy_report=evidence["geo_accuracy"],
        require_capture_coverage=True,
        require_geo_accuracy=True,
    )

    assert report["status"] == "pass"
    assert report["summary"]["total_detections"] == 12


def test_audit_mission_delivery_fails_missing_required_domain_check(tmp_path: Path) -> None:
    evidence = _write_mission_evidence(tmp_path)

    report = audit_mission_delivery(
        tmp_path / "mission_audit.json",
        run_manifest=evidence["run"],
        mission_quality_report=evidence["mission_quality"],
        prediction_quality_report=evidence["prediction_quality"],
        flight_check_report=evidence["flight_check"],
        domain_check_report=None,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "domain_check_report_present" in failed


def test_audit_mission_delivery_fails_missing_required_capture_coverage(tmp_path: Path) -> None:
    evidence = _write_mission_evidence(tmp_path)

    report = audit_mission_delivery(
        tmp_path / "mission_audit.json",
        run_manifest=evidence["run"],
        mission_quality_report=evidence["mission_quality"],
        prediction_quality_report=evidence["prediction_quality"],
        flight_check_report=evidence["flight_check"],
        domain_check_report=evidence["domain_check"],
        capture_coverage_report=None,
        require_capture_coverage=True,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "capture_coverage_report_present" in failed


def test_mission_audit_cli_writes_report(tmp_path: Path) -> None:
    evidence = _write_mission_evidence(tmp_path)
    output = tmp_path / "mission_audit.json"

    result = runner.invoke(
        app,
        [
            "mission-audit",
            str(evidence["run"]),
            str(evidence["mission_quality"]),
            str(evidence["prediction_quality"]),
            "--output",
            str(output),
            "--flight-check-report",
            str(evidence["flight_check"]),
            "--domain-check-report",
            str(evidence["domain_check"]),
            "--flight-log-report",
            str(evidence["flight_log"]),
            "--capture-coverage-report",
            str(evidence["capture_coverage"]),
            "--require-capture-coverage",
            "--geo-accuracy-report",
            str(evidence["geo_accuracy"]),
            "--require-geo-accuracy",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def _write_mission_evidence(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "run": tmp_path / "run_manifest.json",
        "mission_quality": tmp_path / "mission_quality.json",
        "prediction_quality": tmp_path / "prediction_quality.json",
        "flight_check": tmp_path / "flight_check.json",
        "flight_log": tmp_path / "flight_log.json",
        "capture_coverage": tmp_path / "capture_coverage.json",
        "domain_check": tmp_path / "domain_check.json",
        "geo_accuracy": tmp_path / "geo_accuracy.json",
    }
    paths["run"].write_text(
        json.dumps({"image_count": 3, "total_detections": 12, "runtime": {"config_sha256": "cfg123"}}),
        encoding="utf-8",
    )
    paths["mission_quality"].write_text(json.dumps({"status": "pass", "fail_count": 0}), encoding="utf-8")
    paths["prediction_quality"].write_text(json.dumps({"status": "pass", "fail_count": 0}), encoding="utf-8")
    paths["flight_check"].write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    paths["flight_log"].write_text(
        json.dumps({"status": "pass", "summary": {"row_count": 3, "pass_count": 3, "warn_count": 0, "fail_count": 0}}),
        encoding="utf-8",
    )
    paths["capture_coverage"].write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": {
                    "row_count": 3,
                    "missing_image_count": 0,
                    "position_count": 3,
                    "duplicate_position_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    paths["domain_check"].write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    paths["geo_accuracy"].write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    return paths
