"""Crown-center detection inside each plant box.

A banana plant seen from nadir is a ROSETTE: leaves radiate from a central point.
Banana grows in mats of ~2-3 plants, so a single detection box can contain up to
three rosettes. This module finds those crown centers inside each box using the
Fast Radial Symmetry Transform (FRST), so the overlay can mark a small dot on each
plant crown (1-3 per mat), not just the box.

Centers are stored on each Detection as meta["crown_centers"] = [[x, y], ...] in
full-image pixel coordinates.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .models import Detection, InferenceConfig


def _frst(gray: np.ndarray, radii: list[int], alpha: float = 2.0, grad_frac: float = 0.15) -> np.ndarray:
    from scipy import ndimage as ndi

    gray = np.asarray(gray, dtype=np.float32)
    h, w = gray.shape
    gx = ndi.sobel(gray, axis=1, mode="reflect")
    gy = ndi.sobel(gray, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    mmax = float(mag.max())
    if mmax <= 0:
        return np.zeros((h, w), np.float32)
    ys, xs = np.nonzero(mag > grad_frac * mmax)
    if ys.size == 0:
        return np.zeros((h, w), np.float32)
    gmag = mag[ys, xs]
    ux, uy = gx[ys, xs] / gmag, gy[ys, xs] / gmag
    out = np.zeros((h, w), np.float32)
    for n in radii:
        if n < 1:
            continue
        on = np.zeros((h, w), np.float32)
        mn = np.zeros((h, w), np.float32)
        px = np.clip(xs + np.round(ux * n).astype(np.int64), 0, w - 1)
        py = np.clip(ys + np.round(uy * n).astype(np.int64), 0, h - 1)
        np.add.at(on, (py, px), 1.0)
        np.add.at(mn, (py, px), gmag)
        kappa = 9.9 if n > 1 else 8.0
        on = np.clip(np.abs(on), None, kappa)
        fn = (on / kappa) ** alpha * (np.abs(mn) / kappa)
        out += ndi.gaussian_filter(fn, sigma=max(0.5, 0.25 * n))
    return out / max(1, len(radii))


def _centers_in_box(gray_box: np.ndarray, crown_px: float, max_centers: int) -> list[tuple[float, float]]:
    from skimage.feature import peak_local_max

    h, w = gray_box.shape
    if min(h, w) < 6:
        return [(w / 2.0, h / 2.0)]
    r = max(2.0, crown_px / 2.0)
    radii = [int(round(r * f)) for f in (0.5, 0.75, 1.0) if r * f >= 1]
    radii = sorted({max(1, x) for x in radii}) or [2]
    score = _frst(gray_box, radii)
    if score.max() <= 0:
        return [(w / 2.0, h / 2.0)]
    min_dist = max(2, int(r * 0.7))
    pos = score[score > 0]
    ref = float(np.percentile(pos, 99)) if pos.size else float(score.max())
    peaks = peak_local_max(
        score,
        min_distance=min_dist,
        threshold_abs=0.25 * ref,
        num_peaks=max_centers,
        exclude_border=False,
    )
    if len(peaks) == 0:
        return [(w / 2.0, h / 2.0)]
    return [(float(c), float(rr)) for rr, c in peaks]  # (x, y)


def attach_crown_centers(
    image: Image.Image, detections: list[Detection], config: InferenceConfig
) -> None:
    """Fill detection.meta['crown_centers'] with 1-3 rosette centers per box."""
    if not detections:
        return
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ih, iw = gray.shape
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
        # Expect 1-3 plants per mat; scale by how many crowns fit in the box.
        est = int(round(box_area / max(1.0, crown_area)))
        max_centers = min(3, max(1, est))
        local = _centers_in_box(gray[by1:by2, bx1:bx2], crown_px, max_centers)
        det.meta["crown_centers"] = [[bx1 + cx, by1 + cy] for cx, cy in local]
