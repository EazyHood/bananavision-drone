import json
from pathlib import Path

from bananavision.calibration import calibrate_thresholds, write_calibration_report


def test_calibrate_thresholds(tmp_path: Path) -> None:
    detections = {
        "image": "field.jpg",
        "detections": [
            {
                "id": "a",
                "label": "banana_plant",
                "score": 0.9,
                "bbox": [0, 0, 10, 10],
                "center": [5, 5],
                "area_px": 100,
                "source": "test",
            },
            {
                "id": "b",
                "label": "banana_plant",
                "score": 0.2,
                "bbox": [50, 50, 60, 60],
                "center": [55, 55],
                "area_px": 100,
                "source": "test",
            },
        ],
    }
    truth = {"centers": [[5, 5]]}
    detections_path = tmp_path / "field.detections.json"
    truth_path = tmp_path / "field.truth.json"
    detections_path.write_text(json.dumps(detections), encoding="utf-8")
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    report = calibrate_thresholds(detections_path, truth_path, tolerance_px=3, thresholds=[0.1, 0.5])
    assert report["best"]["threshold"] == 0.5
    output = write_calibration_report(report, tmp_path / "calibration.json")
    assert output.exists()
