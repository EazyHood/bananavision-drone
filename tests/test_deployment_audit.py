import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.deploy import build_deployment_manifest
from bananavision.deployment_audit import audit_deployment
from bananavision.release_package import (
    DEPLOYMENT_REQUIRED_ARTIFACTS,
    PackageArtifact,
    build_release_package,
)

runner = CliRunner()


def test_audit_deployment_passes_complete_install_evidence(tmp_path: Path) -> None:
    package_root, preflight, deployment_manifest, smoke = _write_complete_deployment_fixture(tmp_path)

    report = audit_deployment(
        tmp_path / "deployment_audit.json",
        release_package=package_root,
        preflight_report=preflight,
        deployment_manifest=deployment_manifest,
        deployment_smoke_report=smoke,
    )

    assert report["status"] == "pass"
    assert report["package_summary"]["verified_artifact_count"] == report["package_summary"]["artifact_count"]


def test_audit_deployment_fails_bad_preflight(tmp_path: Path) -> None:
    package_root, preflight, deployment_manifest, smoke = _write_complete_deployment_fixture(tmp_path)
    payload = json.loads(preflight.read_text(encoding="utf-8"))
    payload["status"] = "fail"
    payload["checks"][0]["status"] = "fail"
    preflight.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_deployment(
        tmp_path / "deployment_audit.json",
        release_package=package_root,
        preflight_report=preflight,
        deployment_manifest=deployment_manifest,
        deployment_smoke_report=smoke,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "preflight_passed" in failed
    assert "preflight_config_valid" in failed


def test_audit_deployment_fails_api_without_security_options(tmp_path: Path) -> None:
    package_root, preflight, deployment_manifest, smoke = _write_complete_deployment_fixture(tmp_path)
    payload = json.loads(deployment_manifest.read_text(encoding="utf-8"))
    payload["services"]["api"]["start_command"] = "bananavision serve --config configs/banana_uav.yaml"
    deployment_manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_deployment(
        tmp_path / "deployment_audit.json",
        release_package=package_root,
        preflight_report=preflight,
        deployment_manifest=deployment_manifest,
        deployment_smoke_report=smoke,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "deployment_api_key_env" in failed
    assert "deployment_api_upload_limit" in failed


def test_audit_deployment_fails_missing_smoke_report_by_default(tmp_path: Path) -> None:
    package_root, preflight, deployment_manifest, _smoke = _write_complete_deployment_fixture(tmp_path)

    report = audit_deployment(
        tmp_path / "deployment_audit.json",
        release_package=package_root,
        preflight_report=preflight,
        deployment_manifest=deployment_manifest,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "deployment_smoke_report_present" in failed


def test_deployment_audit_cli_writes_report(tmp_path: Path) -> None:
    package_root, preflight, deployment_manifest, smoke = _write_complete_deployment_fixture(tmp_path)
    output = tmp_path / "deployment_audit.json"

    result = runner.invoke(
        app,
        [
            "deployment-audit",
            str(package_root),
            str(preflight),
            str(deployment_manifest),
            "--output",
            str(output),
            "--deployment-smoke-report",
            str(smoke),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def _write_complete_deployment_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    audit = tmp_path / "release_audit.json"
    audit.write_text(json.dumps({"status": "pass", "gates": [{"name": "ok", "status": "pass"}]}), encoding="utf-8")
    deployment_manifest = tmp_path / "deployment_manifest.json"
    deployment_manifest.write_text(
        json.dumps(
            build_deployment_manifest(
                install_dir="/opt/bananavision-drone",
                user="bananavision",
                bananavision_bin="/opt/bananavision-drone/.venv/bin/bananavision",
                config_path="/opt/bananavision-drone/configs/banana_uav.yaml",
                detector="yolo-seg",
                model_path="/models/best.engine",
                watch_dir="/data/mission/incoming",
                mission_output_dir="/data/mission/output",
                api_host="0.0.0.0",
                api_port=8080,
                generated_files=[
                    "bananavision-mission-watch.service",
                    "bananavision-api.service",
                    "deployment_manifest.json",
                ],
            )
        ),
        encoding="utf-8",
    )
    artifacts = _write_package_artifacts(tmp_path, deployment_manifest)
    manifest = build_release_package(
        tmp_path / "dist",
        audit,
        artifacts=artifacts,
        package_name="banana-deploy-v1",
        create_zip=False,
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "pass",
                "runtime": {"config_sha256": "cfg123", "model": {"model_sha256": "model123"}},
                "checks": [
                    {"name": "config", "status": "pass", "detail": "Configuration is valid"},
                    {"name": "output_dir", "status": "pass", "detail": "Writable"},
                    {"name": "model_file", "status": "pass", "detail": "/models/best.engine"},
                ],
            }
        ),
        encoding="utf-8",
    )
    smoke = _write_deployment_smoke(tmp_path)
    return Path(manifest["package_root"]), preflight, deployment_manifest, smoke


def _write_deployment_smoke(tmp_path: Path) -> Path:
    artifacts = tmp_path / "smoke_artifacts"
    artifacts.mkdir()
    for name in ["run_manifest.json", "mission.detections.csv", "mission.detections.geojson", "mission.detections.kml"]:
        (artifacts / name).write_text(name, encoding="utf-8")
    smoke = tmp_path / "deployment_smoke.json"
    smoke.write_text(
        json.dumps(
            {
                "status": "pass",
                "runtime": {"config_sha256": "cfg123"},
                "image_count": 1,
                "total_detections": 3,
                "latency_ms": {"max": 42.0, "mean": 42.0},
                "artifacts": {
                    "run_manifest": str(artifacts / "run_manifest.json"),
                    "mission_csv": str(artifacts / "mission.detections.csv"),
                    "mission_geojson": str(artifacts / "mission.detections.geojson"),
                    "mission_kml": str(artifacts / "mission.detections.kml"),
                },
            }
        ),
        encoding="utf-8",
    )
    return smoke


def _write_package_artifacts(tmp_path: Path, deployment_manifest: Path) -> list[PackageArtifact]:
    artifacts = []
    for label in DEPLOYMENT_REQUIRED_ARTIFACTS:
        if label in {"release_audit", "package_readme"}:
            continue
        if label == "deployment_manifest":
            artifacts.append(PackageArtifact(label, deployment_manifest))
            continue
        suffix = ".engine" if label == "model" else ".yaml" if label == "config" else ".json"
        path = tmp_path / f"{label}{suffix}"
        path.write_text(f"{label}\n", encoding="utf-8")
        artifacts.append(PackageArtifact(label, path))
    return artifacts
