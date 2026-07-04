from __future__ import annotations

import numpy as np
from PIL import Image


def as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        arr = image.astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
    return np.clip(arr, 0, 255)


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if high - low < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def linear_contrast_stretch(rgb: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    stretched = np.empty_like(rgb, dtype=np.float32)
    for band in range(3):
        lo, hi = np.percentile(rgb[..., band], [lower, upper])
        if hi - lo < 1e-6:
            stretched[..., band] = rgb[..., band]
        else:
            stretched[..., band] = np.clip((rgb[..., band] - lo) * 255.0 / (hi - lo), 0, 255)
    return stretched


def excess_green(rgb: np.ndarray) -> np.ndarray:
    total = np.maximum(rgb.sum(axis=-1), 1.0)
    r = rgb[..., 0] / total
    g = rgb[..., 1] / total
    b = rgb[..., 2] / total
    return normalize01(2.0 * g - r - b)


def triangular_greenness_index(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0] / 255.0
    g = rgb[..., 1] / 255.0
    b = rgb[..., 2] / 255.0
    tgi = -0.5 * ((670.0 - 480.0) * (r - g) - (670.0 - 550.0) * (r - b))
    return normalize01(tgi)


def green_dominance(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    dominance = (g - np.maximum(r, b)) / np.maximum(g + np.maximum(r, b), 1.0)
    return normalize01(dominance)


def vegetation_score(rgb: np.ndarray) -> np.ndarray:
    stretched = linear_contrast_stretch(rgb)
    exg = excess_green(stretched)
    tgi = triangular_greenness_index(stretched)
    dom = green_dominance(stretched)
    score = (0.45 * exg) + (0.35 * tgi) + (0.20 * dom)
    return normalize01(score)


def score_to_uint8(score: np.ndarray) -> np.ndarray:
    return np.clip(score * 255.0, 0, 255).astype(np.uint8)
