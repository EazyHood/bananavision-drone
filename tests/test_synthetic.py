import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.synthetic import generate_scene
from bananavision.truth import read_truth_points, truth_cluster_summary
from bananavision.truth_coverage import audit_truth_coverage

runner = CliRunner()


def test_generate_scene_can_write_grouped_banana_mats(tmp_path: Path) -> None:
    image = tmp_path / "clustered.jpg"
    truth = tmp_path / "clustered.truth.json"

    generate_scene(
        image,
        truth,
        width=260,
        height=180,
        plant_count=7,
        seed=11,
        clustered_mats=2,
        min_plants_per_mat=3,
        max_plants_per_mat=3,
    )

    points = read_truth_points(truth)
    summary = truth_cluster_summary(points)
    report = audit_truth_coverage(truth, min_truth_count=7, min_cluster_count=2, min_cluster_truth_count=6)

    assert image.exists()
    assert len(points) == 7
    assert summary == {"cluster_count": 2, "cluster_truth_count": 6}
    assert report["status"] == "pass"


def test_synthetic_cli_can_generate_clustered_truth(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "scene.truth.json"

    result = runner.invoke(
        app,
        [
            "synthetic",
            "--image",
            str(image),
            "--truth",
            str(truth),
            "--width",
            "260",
            "--height",
            "180",
            "--plants",
            "6",
            "--clustered-mats",
            "2",
            "--min-plants-per-mat",
            "3",
            "--max-plants-per-mat",
            "3",
        ],
    )

    payload = json.loads(truth.read_text(encoding="utf-8"))
    grouped = [center for center in payload["centers"] if "group_id" in center]

    assert result.exit_code == 0
    assert image.exists()
    assert len(grouped) == 6
