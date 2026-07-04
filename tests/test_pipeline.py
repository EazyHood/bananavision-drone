from pathlib import Path

from bananavision.models import InferenceConfig
from bananavision.pipeline import iter_images, predict_image, predict_path
from bananavision.synthetic import generate_scene


def test_predict_image_on_synthetic_scene(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.jpg"
    generate_scene(image_path, tmp_path / "truth.json", width=320, height=240, plant_count=8, seed=4)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)
    result = predict_image(image_path, config)
    assert result.width == 320
    assert result.height == 240
    assert result.count > 0


def test_predict_path_writes_manifest_and_mission_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.jpg"
    output_dir = tmp_path / "out"
    generate_scene(image_path, tmp_path / "truth.json", width=320, height=240, plant_count=8, seed=4)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)
    results = predict_path(image_path, output_dir, config)
    assert len(results) == 1
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "mission.detections.csv").exists()
    assert (output_dir / "mission.detections.geojson").exists()
    assert (output_dir / "mission.detections.kml").exists()


def test_iter_images_rejects_missing_path(tmp_path: Path) -> None:
    try:
        iter_images(tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "Input path does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
