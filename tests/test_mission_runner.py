import json
from pathlib import Path

from bananavision.mission_runner import watch_mission
from bananavision.models import InferenceConfig
from bananavision.synthetic import generate_scene


def test_watch_mission_once(tmp_path: Path) -> None:
    watch = tmp_path / "watch"
    output = tmp_path / "out"
    watch.mkdir()
    generate_scene(watch / "scene.jpg", tmp_path / "truth.json", width=240, height=160, plant_count=5)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)
    manifest = watch_mission(watch, output, config, poll_interval=0, settle_seconds=0, max_cycles=1)
    assert manifest["mode"] == "watch"
    assert manifest["image_count"] == 1
    assert (output / "mission_watch_state.json").exists()
    assert (output / "run_manifest.json").exists()


def test_watch_mission_resumes_processed_state(tmp_path: Path) -> None:
    watch = tmp_path / "watch"
    output = tmp_path / "out"
    watch.mkdir()
    generate_scene(watch / "scene.jpg", tmp_path / "truth.json", width=240, height=160, plant_count=5)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)
    first = watch_mission(watch, output, config, poll_interval=0, settle_seconds=0, max_cycles=1)
    second = watch_mission(watch, output, config, poll_interval=0, settle_seconds=0, max_cycles=1)
    state = json.loads((output / "mission_watch_state.json").read_text(encoding="utf-8"))
    assert first["image_count"] == 1
    assert second["image_count"] == 0
    assert state["processed_count"] == 1


def test_watch_mission_can_disable_resume(tmp_path: Path) -> None:
    watch = tmp_path / "watch"
    output = tmp_path / "out"
    watch.mkdir()
    generate_scene(watch / "scene.jpg", tmp_path / "truth.json", width=240, height=160, plant_count=5)
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.55, min_component_area_px=20)
    watch_mission(watch, output, config, poll_interval=0, settle_seconds=0, max_cycles=1)
    second = watch_mission(watch, output, config, poll_interval=0, settle_seconds=0, max_cycles=1, resume=False)
    assert second["image_count"] == 1
