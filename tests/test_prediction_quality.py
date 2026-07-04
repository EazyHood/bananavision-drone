import json
from pathlib import Path

import pytest

from bananavision.prediction_quality import PredictionQualityThresholds, audit_prediction_outputs


def test_audit_prediction_outputs_flags_review_risks(tmp_path: Path) -> None:
    predictions = {
        "image": "field.jpg",
        "width": 300,
        "height": 200,
        "detections": [
            _detection("banana-1", 0.3, [1, 20, 61, 80], [31, 50], split_count=1),
            _detection("banana-2", 0.8, [100, 80, 160, 140], [130, 110], split_count=3),
            _detection("banana-3", 0.9, [106, 84, 166, 144], [136, 114], split_count=1),
        ],
    }
    (tmp_path / "field.detections.json").write_text(json.dumps(predictions), encoding="utf-8")

    report = audit_prediction_outputs(
        tmp_path,
        tmp_path / "quality" / "prediction_quality_report.json",
        PredictionQualityThresholds(max_review_fraction=0.2),
    )

    assert report["status"] == "fail"
    assert report["image_count"] == 1
    assert report["detection_count"] == 3
    assert report["review_detection_count"] == 3
    assert (tmp_path / "quality" / "prediction_quality_report.json").exists()
    assert (tmp_path / "quality" / "prediction_quality_report.csv").exists()
    detections = report["images"][0]["detections"]
    assert "low_confidence" in detections[0]["risks"]
    assert "edge_detection" in detections[0]["risks"]
    assert "high_cluster_split" in detections[1]["risks"]
    assert "duplicate_overlap" in detections[1]["risks"]
    assert "crowded_center" in detections[2]["risks"]


def test_audit_prediction_outputs_zero_detections_can_fail(tmp_path: Path) -> None:
    predictions = {"image": "empty.jpg", "width": 300, "height": 200, "detections": []}
    (tmp_path / "empty.detections.json").write_text(json.dumps(predictions), encoding="utf-8")

    report = audit_prediction_outputs(
        tmp_path,
        tmp_path / "prediction_quality_report.json",
        PredictionQualityThresholds(fail_on_zero_detections=True),
    )

    assert report["status"] == "fail"
    assert report["zero_detection_images"] == 1
    assert report["images"][0]["issues"] == ["zero_detections"]


def test_audit_prediction_outputs_requires_prediction_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audit_prediction_outputs(tmp_path, tmp_path / "prediction_quality_report.json")


def _detection(
    detection_id: str,
    score: float,
    bbox: list[float],
    center: list[float],
    split_count: int,
) -> dict:
    return {
        "id": detection_id,
        "score": score,
        "bbox": bbox,
        "center": center,
        "area_px": 1800,
        "source": "test",
        "meta": {"component_split_count": split_count},
    }
