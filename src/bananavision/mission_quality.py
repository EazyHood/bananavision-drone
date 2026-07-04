from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .geo import describe_image_georeference
from .pipeline import iter_images
from .runtime import utc_now_iso


@dataclass(frozen=True)
class MissionQualityThresholds:
    min_width: int = 1024
    min_height: int = 768
    min_focus_score: float = 12.0
    min_mean_luma: float = 25.0
    max_mean_luma: float = 235.0
    max_dark_fraction: float = 0.35
    max_bright_fraction: float = 0.35
    require_georef: bool = False


def audit_mission_images(
    input_path: str | Path,
    output_json: str | Path,
    thresholds: MissionQualityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or MissionQualityThresholds()
    images = iter_images(input_path)
    rows = [_inspect_image(image_path, thresholds) for image_path in images]
    fail_count = sum(1 for row in rows if row["status"] == "fail")
    warn_count = sum(1 for row in rows if row["status"] == "warn")
    pass_count = sum(1 for row in rows if row["status"] == "pass")
    status = "fail" if fail_count else "warn" if warn_count else "pass"
    report = {
        "created_at": utc_now_iso(),
        "input_path": str(input_path),
        "status": status,
        "image_count": len(rows),
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "thresholds": asdict(thresholds),
        "images": rows,
    }
    write_mission_quality_report(report, output_json)
    return report


def write_mission_quality_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_mission_quality_csv(report, output_json.with_suffix(".csv"))
    return output_json


def _inspect_image(image_path: Path, thresholds: MissionQualityThresholds) -> dict[str, Any]:
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            gray = ImageOps.grayscale(image)
            stats = _image_stats(gray)
    except Exception as exc:
        return {
            "image": str(image_path),
            "width": 0,
            "height": 0,
            "mean_luma": 0.0,
            "dark_fraction": 1.0,
            "bright_fraction": 0.0,
            "focus_score": 0.0,
            "georeferenced": False,
            "georeference_type": None,
            "exif_latitude": None,
            "exif_longitude": None,
            "exif_altitude_m": None,
            "status": "fail",
            "issues": [f"unreadable: {exc}"],
        }

    georef = describe_image_georeference(image_path)
    geotag = (georef or {}).get("geotag") or {}
    georeferenced = georef is not None
    issues: list[str] = []
    warnings: list[str] = []
    if width < thresholds.min_width or height < thresholds.min_height:
        issues.append(f"resolution below {thresholds.min_width}x{thresholds.min_height}")
    if stats["focus_score"] < thresholds.min_focus_score:
        issues.append(f"focus_score below {thresholds.min_focus_score:.2f}")
    if stats["mean_luma"] < thresholds.min_mean_luma:
        issues.append(f"mean_luma below {thresholds.min_mean_luma:.2f}")
    if stats["mean_luma"] > thresholds.max_mean_luma:
        issues.append(f"mean_luma above {thresholds.max_mean_luma:.2f}")
    if stats["dark_fraction"] > thresholds.max_dark_fraction:
        issues.append(f"dark_fraction above {thresholds.max_dark_fraction:.2f}")
    if stats["bright_fraction"] > thresholds.max_bright_fraction:
        issues.append(f"bright_fraction above {thresholds.max_bright_fraction:.2f}")
    if not georeferenced:
        if thresholds.require_georef:
            issues.append("georeference missing")
        else:
            warnings.append("georeference missing")

    status = "fail" if issues else "warn" if warnings else "pass"
    return {
        "image": str(image_path),
        "width": width,
        "height": height,
        "mean_luma": stats["mean_luma"],
        "dark_fraction": stats["dark_fraction"],
        "bright_fraction": stats["bright_fraction"],
        "focus_score": stats["focus_score"],
        "georeferenced": georeferenced,
        "georeference_type": None if georef is None else georef.get("type"),
        "exif_latitude": geotag.get("latitude"),
        "exif_longitude": geotag.get("longitude"),
        "exif_altitude_m": geotag.get("altitude_m"),
        "status": status,
        "issues": issues + warnings,
    }


def _image_stats(gray: Image.Image) -> dict[str, float]:
    array = np.asarray(gray, dtype=np.float32)
    if array.size == 0:
        return {"mean_luma": 0.0, "dark_fraction": 1.0, "bright_fraction": 0.0, "focus_score": 0.0}
    dx = np.diff(array, axis=1)
    dy = np.diff(array, axis=0)
    focus_score = 0.0
    if dx.size or dy.size:
        focus_score = float((np.mean(dx * dx) if dx.size else 0.0) + (np.mean(dy * dy) if dy.size else 0.0))
    return {
        "mean_luma": round(float(np.mean(array)), 4),
        "dark_fraction": round(float(np.mean(array <= 10.0)), 6),
        "bright_fraction": round(float(np.mean(array >= 245.0)), 6),
        "focus_score": round(focus_score, 4),
    }


def _write_mission_quality_csv(report: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "width",
        "height",
        "mean_luma",
        "dark_fraction",
        "bright_fraction",
        "focus_score",
        "georeferenced",
        "georeference_type",
        "exif_latitude",
        "exif_longitude",
        "exif_altitude_m",
        "status",
        "issues",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("images", []):
            csv_row = {key: row.get(key, "") for key in fieldnames}
            csv_row["issues"] = " | ".join(row.get("issues", []))
            writer.writerow(csv_row)
