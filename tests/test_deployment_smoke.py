import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.deployment_smoke import run_deployment_smoke_test, write_deployment_smoke_report
from bananavision.models import InferenceConfig
from bananavision.synthetic import generate_scene

runner = CliRunner()


def test_run_deployment_smoke_test_passes_on_known_image(tmp_path: Path) -> None:
    image = _write_smoke_image(tmp_path)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)

    report = run_deployment_smoke_test(
        image,
        tmp_path / "artifacts",
        config,
        min_detections=1,
        max_image_latency_ms=10_000,
        min_free_gb=0,
    )
    path = write_deployment_smoke_report(report, tmp_path / "deployment_smoke.json")

    assert report["status"] == "pass"
    assert report["image_count"] == 1
    assert report["total_detections"] >= 1
    assert Path(report["artifacts"]["run_manifest"]).exists()
    assert path.exists()


def test_run_deployment_smoke_test_fails_detection_threshold(tmp_path: Path) -> None:
    image = _write_smoke_image(tmp_path)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)

    report = run_deployment_smoke_test(
        image,
        tmp_path / "artifacts",
        config,
        min_detections=999,
        min_free_gb=0,
    )

    failed = {gate["name"] for gate in report["gates"] if gate["status"] == "fail"}
    assert report["status"] == "fail"
    assert "smoke_min_detections" in failed


def test_deployment_smoke_test_cli_writes_report(tmp_path: Path) -> None:
    image = _write_smoke_image(tmp_path)
    output = tmp_path / "deployment_smoke.json"

    result = runner.invoke(
        app,
        [
            "deployment-smoke-test",
            str(image),
            "--output",
            str(output),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--gsd-cm",
            "2.0",
            "--crown-m",
            "0.55",
            "--min-component-area-px",
            "20",
            "--min-free-gb",
            "0",
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert payload["status"] == "pass"


def _write_smoke_image(tmp_path: Path) -> Path:
    image = tmp_path / "smoke_scene.jpg"
    generate_scene(image, tmp_path / "truth.json", width=320, height=240, plant_count=8, seed=4)
    return image
