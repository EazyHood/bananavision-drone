import runpy
import sys
from pathlib import Path

from bananavision.synthetic import generate_scene

ROOT = Path(__file__).resolve().parents[1]


def test_mavlink_folder_bridge_runs_once(tmp_path: Path, monkeypatch, capsys) -> None:
    watch = tmp_path / "watch"
    output = tmp_path / "out"
    watch.mkdir()
    generate_scene(watch / "scene.jpg", tmp_path / "truth.json", width=180, height=140, plant_count=4)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mission_counter.py",
            "--watch",
            str(watch),
            "--output",
            str(output),
            "--settle-seconds",
            "0",
            "--once",
        ],
    )
    runpy.run_path(str(ROOT / "edge" / "mavlink" / "mission_counter.py"), run_name="__main__")

    captured = capsys.readouterr()
    assert "image(s)" in captured.out
    assert (output / "mission_watch_state.json").exists()
    assert (output / "run_manifest.json").exists()


def test_ros2_template_imports_without_ros_runtime() -> None:
    module = runpy.run_path(str(ROOT / "edge" / "ros2" / "banana_counter_node.py"))
    assert "main" in module
