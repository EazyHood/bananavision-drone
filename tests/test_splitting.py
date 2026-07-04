import csv
from pathlib import Path

import yaml
from PIL import Image

from bananavision.splitting import split_yolo_dataset


def test_split_yolo_dataset_keeps_groups_together(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    manifest = tmp_path / "groups.csv"
    rows = []
    for index in range(6):
        name = f"img_{index}.jpg"
        Image.new("RGB", (32, 32), (30, 150, 50)).save(images / name)
        (labels / f"img_{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        rows.append({"image": name, "group": "block_a" if index < 3 else "block_b"})
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "group"])
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "out"
    summary = split_yolo_dataset(images, labels, output, manifest_csv=manifest, train_ratio=0.5, val_ratio=0.5, test_ratio=0)
    assert summary.images == 6
    assignments = list(csv.DictReader((output / "split_assignments.csv").open("r", encoding="utf-8")))
    group_splits: dict[str, set[str]] = {}
    for row in assignments:
        group_splits.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/val"
    assert "test" not in data_yaml


def test_split_yolo_dataset_writes_empty_missing_label(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (32, 32), (30, 150, 50)).save(images / "negative.jpg")
    output = tmp_path / "out"
    summary = split_yolo_dataset(images, labels, output, train_ratio=1, val_ratio=0, test_ratio=0)
    assert summary.missing_labels == 1
    assert (output / "labels" / "train" / "negative.txt").exists()
