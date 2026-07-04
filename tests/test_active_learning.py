import json
from pathlib import Path

from PIL import Image

from bananavision.active_learning import build_review_queue, export_review_crops


def test_build_review_queue(tmp_path: Path) -> None:
    predictions = {
        "image": "field.jpg",
        "detections": [
            {
                "id": "banana-1",
                "score": 0.3,
                "bbox": [1, 2, 3, 4],
                "meta": {"component_split_count": 3},
            }
        ],
    }
    (tmp_path / "field.detections.json").write_text(json.dumps(predictions), encoding="utf-8")
    output = tmp_path / "queue.json"
    items = build_review_queue(tmp_path, output, low_confidence=0.45, high_split_count=2)
    assert len(items) == 2
    assert output.exists()
    assert output.with_suffix(".csv").exists()
    assert {item.reason for item in items} == {"low_confidence", "cluster_split"}


def test_export_review_crops(tmp_path: Path) -> None:
    image = tmp_path / "field.jpg"
    Image.new("RGB", (100, 80), (30, 160, 50)).save(image)
    queue = tmp_path / "review_queue.json"
    queue.write_text(
        json.dumps(
            {
                "count": 1,
                "items": [
                    {
                        "image": str(image),
                        "reason": "low_confidence",
                        "priority": 0.7,
                        "detection_id": "banana-1",
                        "score": 0.3,
                        "bbox": [20, 20, 50, 50],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = export_review_crops(queue, tmp_path / "out", margin_px=5)
    assert manifest["exported_count"] == 1
    assert Path(manifest["items"][0]["crop"]).exists()
