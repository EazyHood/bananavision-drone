from pathlib import Path

import yaml

from bananavision.models import InferenceConfig
from bananavision.synthetic import generate_scene
from bananavision.tuning import tune_config, write_tuned_config, write_tuning_report


def test_tune_config_writes_report_and_config(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "truth.json"
    generate_scene(image, truth, width=240, height=180, plant_count=6, seed=3)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)
    report = tune_config(
        image,
        truth,
        config,
        tolerance_px=80,
        crown_diameters_m=[0.45, 0.55],
        min_center_distance_ratios=[0.35],
        center_distance_weights=[0.2, 0.6],
        canopy_fill_ratios=[0.58],
        rgb_threshold_quantiles=[0.78],
        max_split_instances=[8, 12],
    )
    assert report["best"] is not None
    assert len(report["rows"]) == 8
    assert report["search_space"]["max_split_instances"] == [8, 12]
    assert report["search_space"]["center_distance_weight"] == [0.2, 0.6]
    report_path = write_tuning_report(report, tmp_path / "tuning.json")
    config_path = write_tuned_config(report, tmp_path / "tuned.yaml")
    assert report_path.exists()
    tuned = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert tuned["detector"] == "rgb-canopy"
    assert tuned["expected_crown_diameter_m"] in {0.45, 0.55}
    assert tuned["center_distance_weight"] in {0.2, 0.6}
    assert tuned["max_split_instances"] in {8, 12}
