from pathlib import Path

import yaml
from PIL import Image

from bananavision.tiling import clip_polygon_to_rect, polygon_area, tile_yolo_dataset


def test_clip_polygon_to_rect() -> None:
    polygon = [(-10, 10), (50, 10), (50, 50), (-10, 50)]
    clipped = clip_polygon_to_rect(polygon, (0, 0, 40, 40))
    assert len(clipped) >= 4
    assert all(0 <= x <= 40 and 0 <= y <= 40 for x, y in clipped)
    assert polygon_area(clipped) > 0


def test_tile_yolo_dataset(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (120, 80), (30, 150, 50)).save(images / "field.jpg")
    # One polygon centered in the image.
    (labels / "field.txt").write_text(
        "0 0.250000 0.250000 0.750000 0.250000 0.750000 0.750000 0.250000 0.750000\n",
        encoding="utf-8",
    )
    output = tmp_path / "tiled"
    summary = tile_yolo_dataset(root, output, split="train", tile_size=64, overlap=16)
    assert summary.source_images == 1
    assert summary.tiles_written > 0
    assert summary.objects_kept > 0
    assert (output / "tiling_summary.json").exists()
    data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["train"] == "images/train"
    assert data_yaml["names"][0] == "banana_plant"
    assert list((output / "images" / "train").glob("*.jpg"))
    assert list((output / "labels" / "train").glob("*.txt"))
