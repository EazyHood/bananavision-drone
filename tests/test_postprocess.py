import numpy as np
from PIL import Image, ImageDraw

from bananavision.models import Detection, InferenceConfig
from bananavision.postprocess import (
    bbox_iou,
    mask_center_distance_score,
    nms,
    split_instances_from_mask,
)


def test_bbox_iou() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_center_aware_nms_keeps_close_banana_individuals() -> None:
    detections = [
        Detection("banana_plant", 0.9, (0, 0, 50, 50), (25, 25), 100, "test"),
        Detection("banana_plant", 0.8, (12, 0, 62, 50), (37, 25), 100, "test"),
    ]

    assert len(nms(detections, iou_threshold=0.45)) == 1
    assert len(nms(detections, iou_threshold=0.45, min_center_distance_px=8)) == 2
    assert len(nms(detections, iou_threshold=0.45, min_center_distance_px=15)) == 1


def test_split_three_touching_banana_crowns() -> None:
    width, height = 170, 100
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    centers = [(50, 50), (85, 50), (120, 50)]
    for cx, cy in centers:
        draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=255)
    draw.rectangle((50, 45, 120, 55), fill=255)
    mask = np.asarray(image) > 0
    yy, xx = np.mgrid[0:height, 0:width]
    score = np.zeros((height, width), dtype=np.float32)
    for cx, cy in centers:
        score += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 9**2)))
    score = score / score.max()
    config = InferenceConfig(gsd_cm=2.0, expected_crown_diameter_m=0.48, min_component_area_px=30)
    detections = split_instances_from_mask(mask, score, config, source="test")
    assert len(detections) >= 3
    xs = sorted(d.center[0] for d in detections[:3])
    assert xs[0] < 65
    assert 70 < xs[1] < 100
    assert xs[2] > 105


def test_split_elongated_low_area_banana_mat_uses_extent() -> None:
    width, height = 160, 90
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    centers = [(35, 45), (75, 45), (115, 45)]
    for cx, cy in centers:
        draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=255)
    draw.rectangle((35, 42, 115, 48), fill=255)
    mask = np.asarray(image) > 0
    yy, xx = np.mgrid[0:height, 0:width]
    score = np.zeros((height, width), dtype=np.float32)
    for cx, cy in centers:
        score += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 5**2)))
    score = score / score.max()
    config = InferenceConfig(
        gsd_cm=2.0,
        expected_crown_diameter_m=0.8,
        min_component_area_px=20,
        canopy_fill_ratio=0.95,
        min_center_distance_ratio=0.5,
    )

    detections = split_instances_from_mask(mask, score, config, source="test")

    assert len(detections) == 3
    assert {detection.meta["component_split_count"] for detection in detections} == {3}
    assert all(detection.meta["component_extent_count"] >= 3 for detection in detections)


def test_split_touching_crowns_with_flat_mask_score_uses_distance_peaks() -> None:
    width, height = 160, 90
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    centers = [(35, 45), (75, 45), (115, 45)]
    for cx, cy in centers:
        draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=255)
    draw.rectangle((35, 42, 115, 48), fill=255)
    mask = np.asarray(image) > 0
    score = mask.astype(np.float32)
    config = InferenceConfig(
        gsd_cm=2.0,
        expected_crown_diameter_m=0.8,
        min_component_area_px=20,
        canopy_fill_ratio=0.95,
        min_center_distance_ratio=0.5,
        center_distance_weight=0.8,
    )

    detections = split_instances_from_mask(mask, score, config, source="test")

    assert len(detections) == 3
    xs = sorted(detection.center[0] for detection in detections)
    assert xs[0] < 50
    assert 60 < xs[1] < 90
    assert xs[2] > 100
    assert {detection.meta["center_distance_weight"] for detection in detections} == {0.8}


def test_mask_center_distance_score_prefers_crown_centers() -> None:
    image = Image.new("L", (80, 60), 0)
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 20, 30, 40), fill=255)
    draw.ellipse((42, 20, 62, 40), fill=255)
    mask = np.asarray(image) > 0

    distance = mask_center_distance_score(mask)

    assert distance[30, 20] > distance[20, 20]
    assert distance[30, 52] > distance[20, 52]
