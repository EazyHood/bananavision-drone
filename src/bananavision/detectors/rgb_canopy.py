from __future__ import annotations

import numpy as np
from PIL import Image

from bananavision.detectors.base import Detector
from bananavision.models import Detection, InferenceConfig
from bananavision.postprocess import split_instances_from_mask
from bananavision.preprocess import as_rgb_array, vegetation_score


class RgbCanopyDetector(Detector):
    """Immediate RGB baseline for high-resolution orthomosaics or nadir frames.

    This is not a replacement for a field-trained segmentation model. It gives
    users a usable no-GPU baseline and a sanity check for geospatial workflows.
    """

    def predict(self, image: Image.Image, config: InferenceConfig) -> list[Detection]:
        rgb = as_rgb_array(image)
        score = vegetation_score(rgb)
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        green_gate = (g > r * 1.04) & (g > b * 1.02) & (g > 45)
        adaptive = np.quantile(score, config.rgb_threshold_quantile)
        floor = float(score.mean() + 0.25 * score.std())
        threshold = max(adaptive, floor, 0.18)
        mask = (score >= threshold) & green_gate
        return split_instances_from_mask(mask, score, config, source="rgb-canopy")
