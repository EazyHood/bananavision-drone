"""Crown-center detection inside each plant box.

A banana plant seen from nadir is a ROSETTE of leaves radiating from a central
pseudostem. Banana grows in mats of ~2-3 plants, so a detection box can hold more
than one rosette. This module marks a dot at the CENTER of each rosette inside a
box, so the overlay shows where each banana actually is (1-3 per mat).

Method: within each box, threshold the vegetation (excess green), take its
distance transform, and read the peaks. The distance-transform peak of a green
blob sits at its geometric center, so each dot lands in the middle of a plant's
foliage — not on a leaf edge.

Centers are stored on each Detection as meta["crown_centers"] = [[x, y], ...] in
full-image pixel coordinates.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .models import Detection, InferenceConfig


def _centers_in_box(rgb_box: np.ndarray, crown_px: float, max_centers: int) -> list[tuple[float, float]]:
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    h, w = rgb_box.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0
    if min(h, w) < 6:
        return [(cx0, cy0)]
    a = rgb_box.astype(np.float32) / 255.0
    exg = 2.0 * a[..., 1] - a[..., 0] - a[..., 2]
    mask = exg > 0.04
    if mask.sum() < max(12, 0.03 * h * w):
        return [(cx0, cy0)]

    # Distance transform: high at the center of each green blob, low near edges/gaps.
    dist = ndi.distance_transform_edt(mask).astype(np.float32)
    dist = ndi.gaussian_filter(dist, sigma=max(1.0, 0.12 * crown_px))
    if dist.max() <= 0:
        return [(cx0, cy0)]

    # Centrality prior: the detector's boxes are per-plant, so the crown belongs
    # near the box center. A raised-cosine window suppresses mass that leaks in
    # from NEIGHBORING plants at the box border (the classic edge-dot failure).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt(((xx - cx0) / max(1.0, cx0)) ** 2 + ((yy - cy0) / max(1.0, cy0)) ** 2)
    window = 0.5 * (1.0 + np.cos(np.pi * np.clip(rr, 0.0, 1.0)))
    score = dist * window

    # Peaks only in the interior: never allow a dot on the box border.
    margin = max(2, int(round(0.18 * min(h, w))))
    min_dist = max(3, int(round(0.55 * crown_px)))
    peaks = peak_local_max(
        score,
        min_distance=min_dist,
        num_peaks=max_centers,
        threshold_rel=0.35,
        exclude_border=margin,
    )
    if len(peaks) == 0:
        return [(cx0, cy0)]
    centers = [(float(c), float(r)) for r, c in peaks]  # (x, y)
    if max_centers == 1:
        # Single plant: nudge the box center toward the foliage peak, capped to
        # 30% of the half-size so the dot always stays central.
        px, py = centers[0]
        dx, dy = px - cx0, py - cy0
        lim_x, lim_y = 0.30 * cx0, 0.30 * cy0
        dx = max(-lim_x, min(lim_x, dx))
        dy = max(-lim_y, min(lim_y, dy))
        return [(cx0 + dx, cy0 + dy)]
    return centers


def attach_crown_centers(
    image: Image.Image, detections: list[Detection], config: InferenceConfig
) -> None:
    """Fill detection.meta['crown_centers'] with 1-3 rosette centers per box."""
    if not detections:
        return
    rgb = np.asarray(image.convert("RGB"))
    ih, iw = rgb.shape[:2]
    try:
        crown_px = config.expected_crown_diameter_px
    except Exception:
        crown_px = 40.0
    crown_area = np.pi * (crown_px / 2.0) ** 2
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bx1, by1 = max(0, int(x1)), max(0, int(y1))
        bx2, by2 = min(iw, int(round(x2))), min(ih, int(round(y2)))
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            det.meta["crown_centers"] = [[float(det.center[0]), float(det.center[1])]]
            continue
        box_area = (bx2 - bx1) * (by2 - by1)
        # Most boxes are a single plant; only clearly large boxes can hold 2-3.
        est = int(np.floor(box_area / max(1.0, 0.85 * crown_area)))
        max_centers = min(3, max(1, est))
        local = _centers_in_box(rgb[by1:by2, bx1:bx2], crown_px, max_centers)
        det.meta["crown_centers"] = [[bx1 + cx, by1 + cy] for cx, cy in local]
