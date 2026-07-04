import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.evidence_manifest import build_evidence_manifest

runner = CliRunner()


def test_build_evidence_manifest_passes_required_artifacts(tmp_path: Path) -> None:
    acceptance = _write_json(tmp_path / "acceptance_report.json", {"passed": True, "metrics": {"count_error_rate": 0.0}})
    release_audit = _write_json(tmp_path / "release_audit.json", {"status": "pass"})
    model = tmp_path / "model.engine"
    model.write_text("serialized model", encoding="utf-8")

    report = build_evidence_manifest(
        tmp_path / "evidence_manifest.json",
        {
            "acceptance_report": acceptance,
            "release_audit_report": release_audit,
            "model": model,
        },
        required_labels=["acceptance_report", "release_audit_report", "model"],
    )

    artifacts = {entry["label"]: entry for entry in report["artifacts"]}
    assert report["status"] == "pass"
    assert report["missing_required_count"] == 0
    assert artifacts["acceptance_report"]["reported_status"] == "pass"
    assert artifacts["release_audit_report"]["reported_status"] == "pass"
    assert artifacts["model"]["sha256"]
    assert (tmp_path / "evidence_manifest.json").exists()


def test_build_evidence_manifest_fails_missing_required_artifact(tmp_path: Path) -> None:
    report = build_evidence_manifest(
        tmp_path / "evidence_manifest.json",
        {},
        required_labels=["benchmark_report"],
    )

    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert report["status"] == "fail"
    assert report["missing_required_count"] == 1
    assert "required_artifact:benchmark_report" in failed


def test_build_evidence_manifest_fails_reported_failed_artifact(tmp_path: Path) -> None:
    truth_quality = _write_json(tmp_path / "truth_quality_report.json", {"status": "fail", "issue_count": 1})

    report = build_evidence_manifest(
        tmp_path / "evidence_manifest.json",
        {"truth_quality_report": truth_quality},
    )

    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert report["status"] == "fail"
    assert report["failed_artifact_count"] == 1
    assert "artifact_status:truth_quality_report" in failed


def test_evidence_manifest_cli_writes_passing_report(tmp_path: Path) -> None:
    acceptance = _write_json(tmp_path / "acceptance_report.json", {"passed": True})
    output = tmp_path / "evidence_manifest.json"

    result = runner.invoke(
        app,
        [
            "evidence-manifest",
            "--output",
            str(output),
            "--acceptance-report",
            str(acceptance),
            "--require",
            "acceptance_report",
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert payload["status"] == "pass"
    assert payload["present_count"] == 1


def test_evidence_manifest_cli_exits_nonzero_for_missing_required(tmp_path: Path) -> None:
    output = tmp_path / "evidence_manifest.json"

    result = runner.invoke(
        app,
        [
            "evidence-manifest",
            "--output",
            str(output),
            "--require",
            "acceptance_report",
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.exit_code == 2
    assert payload["status"] == "fail"


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
