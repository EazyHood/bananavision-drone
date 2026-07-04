from pathlib import Path

import yaml
from PIL import Image

from bananavision.quality import audit_dataset_quality, write_quality_report


def test_quality_report_passes_valid_dataset(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "a.jpg", (10, 120, 40))
    _write_image(root / "images" / "val" / "b.jpg", (20, 130, 50))
    _write_label(root / "labels" / "train" / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_label(root / "labels" / "val" / "b.txt", "0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4\n")
    data = _write_data_yaml(root)
    report = audit_dataset_quality(data)
    assert report["status"] == "pass"
    assert report["class_counts"]["0"] == 2
    output = write_quality_report(report, tmp_path / "quality.json")
    assert output.exists()


def test_quality_report_detects_duplicate_and_bad_label(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_image(root / "images" / "train" / "a.jpg", (10, 120, 40))
    _write_image(root / "images" / "val" / "a_copy.jpg", (10, 120, 40))
    _write_label(root / "labels" / "train" / "a.txt", "0 1.2 0.5 0.2 0.2\n")
    _write_label(root / "labels" / "val" / "a_copy.txt", "0 0.5 0.5 0.2 0.2\n")
    data = _write_data_yaml(root)
    report = audit_dataset_quality(data)
    codes = {issue["code"] for issue in report["issues"]}
    assert report["status"] == "fail"
    assert "coordinate_range" in codes
    assert "duplicate_image_across_splits" in codes


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_data_yaml(root: Path) -> Path:
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "banana_plant"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data
