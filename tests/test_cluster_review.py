import json
from pathlib import Path

from PIL import Image, ImageDraw
from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.cluster_review import build_cluster_review

runner = CliRunner()


def test_build_cluster_review_flags_under_and_over_split(tmp_path: Path) -> None:
    detections, truth, image = _write_review_fixture(tmp_path)

    report = build_cluster_review(
        detections,
        truth,
        tmp_path / "cluster_review.json",
        tolerance_px=4,
        image_path=image,
        crops_dir=tmp_path / "crops",
    )

    clusters = {cluster["group_id"]: cluster for cluster in report["clusters"]}
    assert report["status"] == "fail"
    assert report["summary"]["cluster_count"] == 2
    assert report["summary"]["failed_cluster_count"] == 1
    assert clusters["mat-a"]["status"] == "fail"
    assert clusters["mat-a"]["issues"] == ["under_split", "over_split"]
    assert clusters["mat-a"]["missing_truth_count"] == 1
    assert clusters["mat-a"]["extra_prediction_count"] == 1
    assert clusters["mat-b"]["status"] == "pass"
    assert (tmp_path / "cluster_review.csv").exists()
    assert (tmp_path / "crops" / "cluster_review_crops_manifest.json").exists()


def test_cluster_review_cli_writes_report_and_exits_nonzero_on_failed_cluster(tmp_path: Path) -> None:
    detections, truth, image = _write_review_fixture(tmp_path)
    output = tmp_path / "cluster_review.json"

    result = runner.invoke(
        app,
        [
            "cluster-review",
            str(detections),
            str(truth),
            "--output",
            str(output),
            "--tolerance-px",
            "4",
            "--image",
            str(image),
            "--crops-dir",
            str(tmp_path / "crops"),
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.exit_code == 2
    assert payload["status"] == "fail"


def test_cluster_review_passes_complete_grouped_mats(tmp_path: Path) -> None:
    detections = tmp_path / "detections.json"
    truth = tmp_path / "truth.json"
    detections.write_text(
        json.dumps(
            {
                "image": "field.jpg",
                "detections": [
                    _det("a1", 10, 10),
                    _det("a2", 20, 10),
                ],
            }
        ),
        encoding="utf-8",
    )
    truth.write_text(
        json.dumps(
            {
                "centers": [
                    {"x": 10, "y": 10, "group_id": "mat-a"},
                    {"x": 20, "y": 10, "group_id": "mat-a"},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_cluster_review(detections, truth, tmp_path / "cluster_review.json", tolerance_px=4)

    assert report["status"] == "pass"
    assert report["summary"]["failed_cluster_count"] == 0


def _write_review_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    image = tmp_path / "field.jpg"
    canvas = Image.new("RGB", (120, 80), (80, 110, 70))
    draw = ImageDraw.Draw(canvas)
    for cx, cy in [(10, 10), (20, 10), (70, 20), (82, 20)]:
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(30, 170, 55))
    canvas.save(image)
    detections = tmp_path / "field.detections.json"
    detections.write_text(
        json.dumps(
            {
                "image": str(image),
                "detections": [
                    _det("a1", 10, 10),
                    _det("a-extra", 12, 10, score=0.8),
                    _det("b1", 70, 20),
                    _det("b2", 82, 20),
                ],
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "field.truth.json"
    truth.write_text(
        json.dumps(
            {
                "centers": [
                    {"x": 10, "y": 10, "group_id": "mat-a"},
                    {"x": 20, "y": 10, "group_id": "mat-a"},
                    {"x": 70, "y": 20, "group_id": "mat-b"},
                    {"x": 82, "y": 20, "group_id": "mat-b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return detections, truth, image


def _det(detection_id: str, x: float, y: float, score: float = 0.9) -> dict[str, object]:
    return {
        "id": detection_id,
        "label": "banana_plant",
        "score": score,
        "center": [x, y],
        "bbox": [x - 4, y - 4, x + 4, y + 4],
        "area_px": 64,
        "source": "test",
    }
