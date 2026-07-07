from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bananavision.detectors.base import Detector
from bananavision.models import Detection, InferenceConfig
from bananavision.postprocess import nms


class YoloEnsembleDetector(Detector):
    """Weighted-Boxes-Fusion ensemble of several YOLO models.

    Runs every model, then fuses their boxes with Weighted Boxes Fusion (WBF).
    This combines complementary strengths of different architectures (e.g.
    YOLOv8m precision + YOLO11m recall) for a higher F1 than any single model.
    """

    def __init__(
        self,
        model_paths: list[str | Path],
        weights: list[float] | None = None,
        base_conf: float = 0.05,
        wbf_iou: float = 0.55,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Ultralytics is required for yolo-ensemble. Install with: "
                "pip install 'bananavision-drone[ml]'"
            ) from exc
        try:
            import ensemble_boxes  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "ensemble-boxes is required for yolo-ensemble. Install with: "
                "pip install 'bananavision-drone[ml]'"
            ) from exc
        if not model_paths or len(model_paths) < 2:
            raise ValueError("yolo-ensemble needs at least 2 model paths")
        self.models = [YOLO(str(path)) for path in model_paths]
        self.weights = list(weights) if weights else [1.0] * len(self.models)
        self.base_conf = base_conf
        self.wbf_iou = wbf_iou

    def predict(self, image: Image.Image, config: InferenceConfig) -> list[Detection]:
        from ensemble_boxes import weighted_boxes_fusion

        width, height = image.size
        boxes_list: list[list[list[float]]] = []
        scores_list: list[list[float]] = []
        labels_list: list[list[int]] = []
        for model in self.models:
            result = model.predict(
                image, conf=self.base_conf, iou=config.iou_threshold, verbose=False
            )[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [1.0] * len(xyxy)
            normalized = [
                [
                    max(0.0, min(1.0, x1 / width)),
                    max(0.0, min(1.0, y1 / height)),
                    max(0.0, min(1.0, x2 / width)),
                    max(0.0, min(1.0, y2 / height)),
                ]
                for x1, y1, x2, y2 in xyxy
            ]
            boxes_list.append(normalized)
            scores_list.append(scores)
            labels_list.append([0] * len(normalized))

        if not any(len(b) for b in boxes_list):
            return []

        fused_boxes, fused_scores, _ = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=self.weights,
            iou_thr=self.wbf_iou,
            skip_box_thr=0.001,
        )

        detections: list[Detection] = []
        for (x1, y1, x2, y2), score in zip(fused_boxes, fused_scores):
            if float(score) < config.confidence_threshold:
                continue
            px1, py1, px2, py2 = x1 * width, y1 * height, x2 * width, y2 * height
            detections.append(
                Detection(
                    label=config.class_name,
                    score=float(score),
                    bbox=(px1, py1, px2, py2),
                    center=((px1 + px2) / 2.0, (py1 + py2) / 2.0),
                    area_px=max(0.0, (px2 - px1) * (py2 - py1)),
                    source="yolo-ensemble",
                )
            )
        return nms(detections, config.iou_threshold)
