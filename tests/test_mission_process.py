from pathlib import Path

from bananavision.mission_process import process_mission
from bananavision.mission_quality import MissionQualityThresholds
from bananavision.models import InferenceConfig
from bananavision.prediction_quality import PredictionQualityThresholds
from bananavision.synthetic import generate_scene


def test_process_mission_writes_operational_artifacts(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    output_dir = tmp_path / "mission"
    inventory_dir = tmp_path / "inventory"
    generate_scene(image, tmp_path / "truth.json", width=320, height=240, plant_count=8, seed=4)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)

    manifest = process_mission(
        image,
        output_dir,
        config,
        mission_quality_thresholds=MissionQualityThresholds(
            min_width=64,
            min_height=64,
            min_focus_score=0.0,
            require_georef=False,
        ),
        prediction_quality_thresholds=PredictionQualityThresholds(max_review_fraction=1.0),
        inventory_dir=inventory_dir,
        observed_at="2026-07-03T12:00:00Z",
    )

    assert manifest["status"] in {"pass", "warn"}
    assert manifest["image_count"] == 1
    assert manifest["total_detections"] > 0
    assert Path(manifest["artifacts"]["mission_process_manifest"]).exists()
    assert Path(manifest["artifacts"]["mission_quality_report"]).exists()
    assert Path(manifest["artifacts"]["prediction_quality_report"]).exists()
    assert Path(manifest["artifacts"]["run_manifest"]).exists()
    assert Path(manifest["artifacts"]["field_report"]).exists()
    assert manifest["inventory"] is not None
    assert (inventory_dir / "inventory.json").exists()
