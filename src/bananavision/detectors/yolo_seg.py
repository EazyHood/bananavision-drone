from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bananavision.detectors.base import Detector
from bananavision.models import Detection, InferenceConfig
from bananavision.postprocess import nms, split_instances_from_mask


class YoloSegDetector(Detector):
    def __init__(self, model_path: str | Path):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Ultralytics is required for yolo-seg. Install with: "
                "pip install 'bananavision-drone[ml]'"
            ) from exc
        self.model = YOLO(str(model_path))

    def predict(self, image: Image.Image, config: InferenceConfig) -> list[Detection]:
        result = self.model.predict(
            image,
            conf=config.confidence_threshold,
            iou=config.iou_threshold,
            verbose=False,
        )[0]
        detections: list[Detection] = []
        width, height = image.size
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None:
            return detections

        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(boxes))
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), int)

        if masks is not None and getattr(masks, "data", None) is not None:
            mask_data = masks.data.cpu().numpy()
            for index, raw_mask in enumerate(mask_data):
                label = str(names.get(int(classes[index]), config.class_name))
                if label != config.class_name and config.class_name not in {label, "banana", "banana_mat"}:
                    label = config.class_name
                mask = _resize_mask(raw_mask, (width, height))
                score_map = mask.astype(np.float32)
                split = split_instances_from_mask(
                    mask,
                    score_map,
                    config,
                    source="yolo-seg",
                    base_score=float(confs[index]),
                )
                for detection in split:
                    detection.label = label
                    detection.meta["parent_yolo_bbox"] = [float(v) for v in xyxy[index]]
                detections.extend(split)
            min_distance = config.expected_crown_diameter_px * config.min_center_distance_ratio * 0.65
            return nms(detections, config.iou_threshold, min_center_distance_px=min_distance)

        for index, bbox in enumerate(xyxy):
            x1, y1, x2, y2 = [float(v) for v in bbox]
            detections.append(
                Detection(
                    label=str(names.get(int(classes[index]), config.class_name)),
                    score=float(confs[index]),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    area_px=max(0.0, (x2 - x1) * (y2 - y1)),
                    source="yolo-box",
                )
            )
        return nms(detections, config.iou_threshold)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if mask.shape == (height, width):
        return mask > 0.5
    pil = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    return np.asarray(pil.resize((width, height), Image.Resampling.NEAREST)) > 0
