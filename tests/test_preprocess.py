import numpy as np

from bananavision.preprocess import (
    excess_green,
    linear_contrast_stretch,
    normalize01,
    vegetation_score,
)


def test_normalize_constant_returns_zero() -> None:
    values = np.ones((4, 4), dtype=np.float32) * 3
    assert np.all(normalize01(values) == 0)


def test_vegetation_score_prefers_green_pixels() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.float32)
    rgb[:, :5] = [40, 190, 50]
    rgb[:, 5:] = [140, 110, 90]
    score = vegetation_score(rgb)
    assert score[:, :5].mean() > score[:, 5:].mean()


def test_contrast_stretch_preserves_shape() -> None:
    rgb = np.random.default_rng(1).integers(0, 255, (16, 12, 3)).astype(np.float32)
    stretched = linear_contrast_stretch(rgb)
    assert stretched.shape == rgb.shape
    assert 0 <= stretched.min() <= stretched.max() <= 255


def test_excess_green_range() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    rgb[..., 1] = 255
    exg = excess_green(rgb)
    assert exg.min() >= 0
    assert exg.max() <= 1
