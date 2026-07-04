from pathlib import Path

from PIL import Image

from bananavision.models import InferenceConfig
from bananavision.readiness import run_preflight, write_preflight_report
from bananavision.synthetic import generate_scene


def test_run_preflight_with_rgb_baseline(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    generate_scene(image, tmp_path / "truth.json", width=160, height=120, plant_count=4)
    report = run_preflight(InferenceConfig(), input_path=image, output_dir=tmp_path / "out", min_free_gb=0)
    assert report["status"] in {"pass", "warn"}
    names = {check["name"] for check in report["checks"]}
    assert "input_images" in names
    assert "output_dir" in names
    path = write_preflight_report(report, tmp_path / "preflight.json")
    assert path.exists()


def test_run_preflight_fails_missing_model(tmp_path: Path) -> None:
    config = InferenceConfig(detector="yolo-seg", model_path=str(tmp_path / "missing.pt"))
    report = run_preflight(config, output_dir=tmp_path / "out", min_free_gb=0)
    assert report["status"] == "fail"
    assert any(check["name"] == "model_file" and check["status"] == "fail" for check in report["checks"])


def test_run_preflight_accepts_exif_gps_geotag(tmp_path: Path) -> None:
    image = tmp_path / "geotagged.jpg"
    _write_exif_geotagged_image(image)
    report = run_preflight(InferenceConfig(), input_path=image, output_dir=tmp_path / "out", min_free_gb=0)
    georef = next(check for check in report["checks"] if check["name"] == "georeference")
    assert georef["status"] == "pass"
    assert "EXIF GPS" in georef["detail"]


def _write_exif_geotagged_image(path: Path) -> None:
    image = Image.new("RGB", (128, 128), (80, 160, 90))
    exif = Image.Exif()
    exif[34853] = {
        1: "N",
        2: (4.0, 36.0, 0.0),
        3: "W",
        4: (74.0, 4.0, 0.0),
        5: 0,
        6: 123.0,
    }
    image.save(path, exif=exif)
