import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.release_package import (
    DEPLOYMENT_REQUIRED_ARTIFACTS,
    PackageArtifact,
    build_release_package,
    parse_artifact_specs,
    verify_release_package,
)

runner = CliRunner()


def test_build_release_package_writes_manifest_and_zip(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path / "release_audit.json", status="pass")
    model = tmp_path / "best.pt"
    config = tmp_path / "banana.yaml"
    model.write_bytes(b"pretend weights")
    config.write_text("detector: yolo-seg\n", encoding="utf-8")

    manifest = build_release_package(
        tmp_path / "dist",
        audit,
        artifacts=[
            PackageArtifact("model", model),
            PackageArtifact("config", config),
        ],
        package_name="banana-v1",
    )

    package_root = Path(manifest["package_root"])
    manifest_path = package_root / "release_package_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = {artifact["label"] for artifact in payload["artifacts"]}

    assert payload["package_status"] == "release"
    assert payload["release_audit_status"] == "pass"
    assert payload["manifest_sha256"]
    assert labels == {"release_audit", "model", "config", "package_readme"}
    assert (tmp_path / "dist" / "banana-v1.zip").exists()
    assert manifest["zip_sha256"]

    verify_report = verify_release_package(package_root)
    zip_verify_report = verify_release_package(tmp_path / "dist" / "banana-v1.zip")
    assert verify_report["status"] == "pass"
    assert verify_report["verified_artifact_count"] == 4
    assert zip_verify_report["status"] == "pass"
    assert zip_verify_report["verified_artifact_count"] == 4

    deployment_report = verify_release_package(package_root, require_deployment_artifacts=True)
    failed = {check["name"] for check in deployment_report["checks"] if check["status"] == "fail"}
    assert deployment_report["status"] == "fail"
    assert "deployment_artifact:model_manifest" in failed


def test_verify_release_package_can_require_deployment_artifacts(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path / "release_audit.json", status="pass")
    artifacts = _write_deployment_artifacts(tmp_path)

    manifest = build_release_package(
        tmp_path / "dist",
        audit,
        artifacts=artifacts,
        package_name="banana-deploy-v1",
    )

    package_root = Path(manifest["package_root"])
    folder_report = verify_release_package(package_root, require_deployment_artifacts=True)
    zip_report = verify_release_package(tmp_path / "dist" / "banana-deploy-v1.zip", require_deployment_artifacts=True)

    assert folder_report["status"] == "pass"
    assert zip_report["status"] == "pass"
    assert folder_report["deployment_artifacts_required"] is True


def test_build_release_package_rejects_failed_audit(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path / "release_audit.json", status="fail")

    with pytest.raises(ValueError, match="Release audit status"):
        build_release_package(tmp_path / "dist", audit, package_name="banana-v1")


def test_build_release_package_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path / "release_audit.json", status="pass")
    build_release_package(tmp_path / "dist", audit, package_name="banana-v1", create_zip=False)

    with pytest.raises(FileExistsError, match="already exists"):
        build_release_package(tmp_path / "dist", audit, package_name="banana-v1", create_zip=False)

    manifest = build_release_package(
        tmp_path / "dist",
        audit,
        package_name="banana-v1",
        create_zip=False,
        overwrite=True,
    )
    assert manifest["package_status"] == "release"


def test_build_release_package_can_create_exploratory_bundle(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path / "release_audit.json", status="fail")

    manifest = build_release_package(
        tmp_path / "dist",
        audit,
        package_name="debug-v1",
        allow_failed_audit=True,
        create_zip=False,
    )

    assert manifest["package_status"] == "exploratory"
    assert manifest["zip_path"] is None

    strict_report = verify_release_package(tmp_path / "dist" / "debug-v1")
    exploratory_report = verify_release_package(tmp_path / "dist" / "debug-v1", allow_exploratory=True)
    assert strict_report["status"] == "fail"
    assert exploratory_report["status"] == "warn"


def test_verify_release_package_fails_tampered_artifact(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path / "release_audit.json", status="pass")
    model = tmp_path / "best.pt"
    model.write_bytes(b"pretend weights")
    build_release_package(
        tmp_path / "dist",
        audit,
        artifacts=[PackageArtifact("model", model)],
        package_name="banana-v1",
        create_zip=False,
    )
    packaged_model = tmp_path / "dist" / "banana-v1" / "artifacts" / "model.pt"
    packaged_model.write_bytes(b"changed weights")

    report = verify_release_package(tmp_path / "dist" / "banana-v1")

    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert report["status"] == "fail"
    assert "artifact:model:sha256" in failed


def test_parse_artifact_specs_requires_label_path_format() -> None:
    artifacts = parse_artifact_specs(["weights=best.pt", "config=configs/banana_uav.yaml"])
    assert [artifact.label for artifact in artifacts] == ["weights", "config"]

    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_artifact_specs(["broken"])


def test_release_package_cli_writes_package(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path / "release_audit.json", status="pass")
    extra = tmp_path / "extra.txt"
    geo_accuracy = tmp_path / "geo_accuracy.json"
    flight_log = tmp_path / "flight_log.json"
    mission_audit = tmp_path / "mission_audit.json"
    extra.write_text("field note", encoding="utf-8")
    geo_accuracy.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    flight_log.write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    mission_audit.write_text(json.dumps({"status": "pass"}), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "release-package",
            str(audit),
            "--output",
            str(tmp_path / "dist"),
            "--package-name",
            "cli-v1",
            "--artifact",
            f"field_note={extra}",
            "--flight-log-report",
            str(flight_log),
            "--mission-audit-report",
            str(mission_audit),
            "--geo-accuracy-report",
            str(geo_accuracy),
            "--no-zip",
        ],
    )

    assert result.exit_code == 0
    manifest = tmp_path / "dist" / "cli-v1" / "release_package_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["package_status"] == "release"
    assert any(artifact["label"] == "field_note" for artifact in payload["artifacts"])
    assert any(artifact["label"] == "flight_log_report" for artifact in payload["artifacts"])
    assert any(artifact["label"] == "mission_audit_report" for artifact in payload["artifacts"])
    assert any(artifact["label"] == "geo_accuracy_report" for artifact in payload["artifacts"])


def test_release_package_verify_cli_writes_report(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path / "release_audit.json", status="pass")
    build_release_package(tmp_path / "dist", audit, package_name="cli-v1", create_zip=False)
    output = tmp_path / "verify.json"

    result = runner.invoke(
        app,
        [
            "release-package-verify",
            str(tmp_path / "dist" / "cli-v1"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"


def _write_audit(path: Path, status: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "gates": [
                    {"name": "acceptance_passed", "status": status, "detail": ""},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_deployment_artifacts(tmp_path: Path) -> list[PackageArtifact]:
    artifacts = []
    for label in DEPLOYMENT_REQUIRED_ARTIFACTS:
        if label in {"release_audit", "package_readme"}:
            continue
        suffix = ".engine" if label == "model" else ".yaml" if label == "config" else ".json"
        path = tmp_path / f"{label}{suffix}"
        path.write_text(f"{label}\n", encoding="utf-8")
        artifacts.append(PackageArtifact(label, path))
    return artifacts
