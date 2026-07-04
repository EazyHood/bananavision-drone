from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pipeline import iter_images
from .runtime import utc_now_iso

FEATURE_KEYS = [
    "width",
    "height",
    "aspect_ratio",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "mean_luma",
    "std_luma",
    "mean_saturation",
]


@dataclass(frozen=True)
class DomainProfileSettings:
    bins: int = 16
    max_side: int = 512


@dataclass(frozen=True)
class DomainShiftThresholds:
    max_histogram_distance: float = 0.20
    max_feature_z: float = 4.0
    max_outlier_fraction: float = 0.0
    min_reference_images: int = 1


def build_domain_profile(
    input_path: str | Path,
    output_json: str | Path | None = None,
    settings: DomainProfileSettings | None = None,
) -> dict[str, Any]:
    settings = settings or DomainProfileSettings()
    _validate_settings(settings)
    images = iter_images(input_path)
    rows = [_extract_image_features(path, settings) for path in images]
    feature_matrix = np.asarray([[row["features"][key] for key in FEATURE_KEYS] for row in rows], dtype=np.float64)
    histogram_stack = np.asarray([row["rgb_histogram"] for row in rows], dtype=np.float64)
    profile = {
        "created_at": utc_now_iso(),
        "type": "domain_profile",
        "input_path": str(input_path),
        "settings": asdict(settings),
        "image_count": len(rows),
        "feature_keys": FEATURE_KEYS,
        "feature_mean": _feature_dict(np.mean(feature_matrix, axis=0)),
        "feature_std": _feature_dict(np.std(feature_matrix, axis=0)),
        "rgb_histogram_mean": _round_nested(np.mean(histogram_stack, axis=0)),
        "images": rows,
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def audit_domain_shift(
    input_path: str | Path,
    profile_json: str | Path,
    output_json: str | Path | None = None,
    thresholds: DomainShiftThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DomainShiftThresholds()
    _validate_thresholds(thresholds)
    profile_path = Path(profile_json)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    reference_count = int(profile.get("image_count", 0))
    settings = DomainProfileSettings(**(profile.get("settings") or {}))
    reference_histogram = np.asarray(profile["rgb_histogram_mean"], dtype=np.float64)
    feature_mean = {key: float(value) for key, value in (profile.get("feature_mean") or {}).items()}
    feature_std = {key: float(value) for key, value in (profile.get("feature_std") or {}).items()}

    rows = []
    for path in iter_images(input_path):
        row = _extract_image_features(path, settings)
        assessment = _assess_row(row, reference_histogram, feature_mean, feature_std, thresholds)
        rows.append({**assessment, "image": row["image"], "features": row["features"]})

    outlier_count = sum(1 for row in rows if row["is_outlier"])
    outlier_fraction = outlier_count / len(rows) if rows else 1.0
    checks = [
        _check(
            "reference_support",
            "pass" if reference_count >= thresholds.min_reference_images else "fail",
            f"{reference_count} >= {thresholds.min_reference_images}",
        ),
        _check(
            "outlier_fraction",
            "pass" if outlier_fraction <= thresholds.max_outlier_fraction else "fail",
            f"{outlier_fraction:.4f} <= {thresholds.max_outlier_fraction:.4f}",
        ),
    ]
    report = {
        "created_at": utc_now_iso(),
        "type": "domain_check",
        "input_path": str(input_path),
        "profile_path": str(profile_path),
        "status": _overall_status(checks),
        "image_count": len(rows),
        "reference_image_count": reference_count,
        "outlier_count": outlier_count,
        "outlier_fraction": round(outlier_fraction, 6),
        "thresholds": asdict(thresholds),
        "checks": checks,
        "images": rows,
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_domain_csv(report, output_path.with_suffix(".csv"))
    return report


def _extract_image_features(path: Path, settings: DomainProfileSettings) -> dict[str, Any]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if max(image.size) > settings.max_side:
            image.thumbnail((settings.max_side, settings.max_side))
        array = np.asarray(image, dtype=np.float32)
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    max_rgb = np.max(array, axis=2)
    min_rgb = np.min(array, axis=2)
    saturation = np.divide(
        max_rgb - min_rgb,
        max_rgb,
        out=np.zeros_like(max_rgb, dtype=np.float32),
        where=max_rgb > 0,
    )
    features = {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height) if height else 0.0,
        "mean_r": float(np.mean(red)),
        "mean_g": float(np.mean(green)),
        "mean_b": float(np.mean(blue)),
        "std_r": float(np.std(red)),
        "std_g": float(np.std(green)),
        "std_b": float(np.std(blue)),
        "mean_luma": float(np.mean(luma)),
        "std_luma": float(np.std(luma)),
        "mean_saturation": float(np.mean(saturation)),
    }
    return {
        "image": str(path),
        "features": {key: round(features[key], 6) for key in FEATURE_KEYS},
        "rgb_histogram": _round_nested(_rgb_histogram(array, settings.bins)),
    }


def _rgb_histogram(array: np.ndarray, bins: int) -> list[list[float]]:
    histograms: list[list[float]] = []
    for channel in range(3):
        counts, _ = np.histogram(array[:, :, channel], bins=bins, range=(0, 255))
        total = float(np.sum(counts))
        histograms.append((counts / total).astype(float).tolist() if total else [0.0] * bins)
    return histograms


def _assess_row(
    row: dict[str, Any],
    reference_histogram: np.ndarray,
    feature_mean: dict[str, float],
    feature_std: dict[str, float],
    thresholds: DomainShiftThresholds,
) -> dict[str, Any]:
    histogram = np.asarray(row["rgb_histogram"], dtype=np.float64)
    histogram_distance = _histogram_distance(histogram, reference_histogram)
    z_scores = {
        key: _z_score(float(row["features"][key]), feature_mean.get(key, 0.0), feature_std.get(key, 0.0))
        for key in FEATURE_KEYS
    }
    worst_feature = max(z_scores, key=lambda key: abs(z_scores[key]))
    max_feature_z = abs(z_scores[worst_feature])
    issues: list[str] = []
    if histogram_distance > thresholds.max_histogram_distance:
        issues.append("histogram_distance")
    if max_feature_z > thresholds.max_feature_z:
        issues.append(f"feature_z:{worst_feature}")
    is_outlier = bool(issues)
    return {
        "status": "fail" if is_outlier else "pass",
        "is_outlier": is_outlier,
        "histogram_distance": round(histogram_distance, 6),
        "max_feature_z": round(max_feature_z, 6),
        "worst_feature": worst_feature,
        "issues": issues,
    }


def _histogram_distance(histogram: np.ndarray, reference: np.ndarray) -> float:
    if histogram.shape != reference.shape:
        raise ValueError("Profile histogram shape does not match image histogram shape")
    per_channel = 0.5 * np.sum(np.abs(histogram - reference), axis=1)
    return float(np.mean(per_channel))


def _z_score(observed: float, mean: float, std: float) -> float:
    denominator = max(abs(std), abs(mean) * 0.05, 1e-6)
    return (observed - mean) / denominator


def _feature_dict(values: np.ndarray) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in zip(FEATURE_KEYS, values, strict=True)}


def _round_nested(values: Any) -> Any:
    if isinstance(values, np.ndarray):
        return _round_nested(values.tolist())
    if isinstance(values, list):
        return [_round_nested(value) for value in values]
    return round(float(values), 8)


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _overall_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"


def _write_domain_csv(report: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "status", "histogram_distance", "max_feature_z", "worst_feature", "issues"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("images", []):
            writer.writerow(
                {
                    "image": row.get("image", ""),
                    "status": row.get("status", ""),
                    "histogram_distance": row.get("histogram_distance", ""),
                    "max_feature_z": row.get("max_feature_z", ""),
                    "worst_feature": row.get("worst_feature", ""),
                    "issues": " | ".join(row.get("issues", [])),
                }
            )


def _validate_settings(settings: DomainProfileSettings) -> None:
    if settings.bins < 4:
        raise ValueError("bins must be at least 4")
    if settings.max_side < 32:
        raise ValueError("max_side must be at least 32")


def _validate_thresholds(thresholds: DomainShiftThresholds) -> None:
    if thresholds.max_histogram_distance < 0:
        raise ValueError("max_histogram_distance must be non-negative")
    if thresholds.max_feature_z <= 0:
        raise ValueError("max_feature_z must be positive")
    if not 0.0 <= thresholds.max_outlier_fraction <= 1.0:
        raise ValueError("max_outlier_fraction must be between 0 and 1")
    if thresholds.min_reference_images < 1:
        raise ValueError("min_reference_images must be at least 1")
