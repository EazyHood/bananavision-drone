import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.deploy import build_deployment_manifest
from bananavision.drone_ready import run_drone_ready_check
from bananavision.models import InferenceConfig
from bananavision.release_package import (
    DEPLOYMENT_REQUIRED_ARTIFACTS,
    PackageArtifact,
    build_release_package,
)
from bananavision.synthetic import generate_scene

runner = CliRunner()


def test_run_drone_ready_check_writes_all_gate_reports(tmp_path: Path) -> None:
    package_root, deployment_manifest = _write_release_package(tmp_path)
    smoke_image = _write_smoke_image(tmp_path)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)

    report = run_drone_ready_check(
        tmp_path / "drone_ready",
        release_package=package_root,
        deployment_manifest=deployment_manifest,
        smoke_image=smoke_image,
        config=config,
        min_detections=1,
        max_image_latency_ms=10_000,
        min_free_gb=0,
        allow_warn_preflight=True,
    )

    assert report["status"] in {"pass", "warn"}
    assert Path(report["artifacts"]["preflight_report"]).exists()
    assert Path(report["artifacts"]["deployment_smoke_report"]).exists()
    assert Path(report["artifacts"]["deployment_audit_report"]).exists()
    assert Path(report["artifacts"]["evidence_manifest"]).exists()
    assert Path(report["artifacts"]["drone_ready_report"]).exists()


def test_drone_ready_cli_writes_summary(tmp_path: Path) -> None:
    package_root, deployment_manifest = _write_release_package(tmp_path)
    smoke_image = _write_smoke_image(tmp_path)
    output_dir = tmp_path / "drone_ready"

    result = runner.invoke(
        app,
        [
            "drone-ready",
            str(package_root),
            str(deployment_manifest),
            str(smoke_image),
            "--output",
            str(output_dir),
            "--gsd-cm",
            "2.0",
            "--crown-m",
            "0.55",
            "--min-component-area-px",
            "20",
            "--max-image-latency-ms",
            "10000",
            "--min-free-gb",
            "0",
            "--allow-warn-preflight",
        ],
    )

    payload = json.loads((output_dir / "drone_ready_report.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert payload["status"] in {"pass", "warn"}


def _write_release_package(tmp_path: Path) -> tuple[Path, Path]:
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
                detector="rgb-canopy",
                model_path=None,
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
    manifest = build_release_package(
        tmp_path / "dist",
        audit,
        artifacts=artifacts,
        package_name="banana-deploy-v1",
        create_zip=False,
    )
    return Path(manifest["package_root"]), deployment_manifest


def _write_smoke_image(tmp_path: Path) -> Path:
    image = tmp_path / "smoke_scene.jpg"
    generate_scene(image, tmp_path / "truth.json", width=320, height=240, plant_count=8, seed=4)
    image.with_suffix(".jgw").write_text("0.02\n0\n0\n-0.02\n100\n100\n", encoding="utf-8")
    return image
