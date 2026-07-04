from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetAudit:
    images: int
    labels: int
    missing_labels: list[str]
    empty_labels: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_labels


def audit_yolo_dataset(data_yaml: str | Path) -> DatasetAudit:
    data_yaml = Path(data_yaml)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    train = root / str(config.get("train", "images/train"))
    image_paths = [
        path
        for path in train.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ]
    missing: list[str] = []
    empty: list[str] = []
    labels = 0
    for image_path in image_paths:
        label_path = _label_path_for_image(image_path)
        if not label_path.exists():
            missing.append(str(image_path))
            continue
        labels += 1
        if label_path.stat().st_size == 0:
            empty.append(str(label_path))
    return DatasetAudit(len(image_paths), labels, missing, empty)


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
