import json
from pathlib import Path

from bananavision.annotations import convert_coco_to_yolo_seg, convert_labelme_to_yolo_seg


def test_convert_coco_to_yolo_seg(tmp_path: Path) -> None:
    coco = {
        "images": [{"id": 1, "file_name": "field.jpg", "width": 100, "height": 50}],
        "categories": [{"id": 7, "name": "banana_plant"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
            }
        ],
    }
    coco_path = tmp_path / "instances.json"
    labels_dir = tmp_path / "labels"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    summary = convert_coco_to_yolo_seg(coco_path, labels_dir, target_names={"banana_plant"})
    assert summary.objects == 1
    label = (labels_dir / "field.txt").read_text(encoding="utf-8").strip()
    assert label == "0 0.100000 0.200000 0.300000 0.200000 0.300000 0.600000 0.100000 0.600000"


def test_convert_labelme_to_yolo_seg(tmp_path: Path) -> None:
    payload = {
        "imagePath": "field.jpg",
        "imageWidth": 100,
        "imageHeight": 50,
        "shapes": [
            {
                "label": "banana_plant",
                "shape_type": "rectangle",
                "points": [[10, 10], [30, 30]],
            }
        ],
    }
    (tmp_path / "field.json").write_text(json.dumps(payload), encoding="utf-8")
    labels_dir = tmp_path / "labels"
    summary = convert_labelme_to_yolo_seg(tmp_path, labels_dir)
    assert summary.objects == 1
    assert (labels_dir / "field.txt").exists()
