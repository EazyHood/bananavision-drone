from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

from .runtime import utc_now_iso


@dataclass(frozen=True)
class CaptureCoverageThresholds:
    min_images: int = 1
    max_position_gap_m: float | None = 35.0
    max_time_gap_s: float | None = None
    min_position_delta_m: float = 0.25
    require_positions: bool = True
    require_timestamps: bool = False
    require_image_files: bool = False


def audit_capture_coverage(
    capture_log_csv: str | Path,
    output_json: str | Path,
    image_dir: str | Path | None = None,
    thresholds: CaptureCoverageThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or CaptureCoverageThresholds()
    _validate_thresholds(thresholds)
    rows = _read_capture_rows(capture_log_csv)
    image_root = None if image_dir is None else Path(image_dir)
    row_reports = _row_reports(rows, image_root)
    _add_step_metrics(row_reports)
    _add_threshold_issues(row_reports, thresholds)
    checks = _checks(row_reports, image_root, thresholds)
    report = {
        "created_at": utc_now_iso(),
        "capture_log_csv": str(capture_log_csv),
        "image_dir": None if image_root is None else str(image_root),
        "status": _overall_status(checks),
        "thresholds": asdict(thresholds),
        "summary": _summary(row_reports, thresholds.min_position_delta_m),
        "checks": checks,
        "rows": row_reports,
    }
    write_capture_coverage_report(report, output_json)
    return report


def write_capture_coverage_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_capture_coverage_csv(report, output_json.with_suffix(".csv"))
    return output_json


def _read_capture_rows(capture_log_csv: str | Path) -> list[dict[str, str]]:
    path = Path(capture_log_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{_normalize_key(key): (value or "").strip() for key, value in row.items() if key} for row in reader]
    if not rows:
        raise ValueError(f"No capture rows found in CSV: {path}")
    return rows


def _row_reports(rows: list[dict[str, str]], image_root: Path | None) -> list[dict[str, Any]]:
    reports = []
    for index, row in enumerate(rows, start=1):
        image_ref = _first_text(row, "image", "filename", "file", "path", "image_path")
        position = _position(row)
        timestamp = _timestamp(row)
        image_exists = _image_exists(image_ref, image_root)
        issues = []
        if not image_ref:
            issues.append("image_reference_missing")
        if image_root is not None and image_ref and image_exists is False:
            issues.append("image_file_missing")
        if position is None:
            issues.append("position_missing")
        if timestamp is None:
            issues.append("timestamp_missing")
        reports.append(
            {
                "row": index,
                "image": image_ref,
                "image_exists": image_exists,
                "position_type": None if position is None else position["type"],
                "x": None if position is None else position["x"],
                "y": None if position is None else position["y"],
                "timestamp": None if timestamp is None else timestamp["raw"],
                "timestamp_seconds": None if timestamp is None else timestamp["seconds"],
                "step_distance_m": None,
                "step_time_s": None,
                "status": "warn" if issues else "pass",
                "issues": issues,
            }
        )
    return reports


def _add_step_metrics(rows: list[dict[str, Any]]) -> None:
    previous_position: dict[str, Any] | None = None
    previous_time: float | None = None
    for row in rows:
        position = _row_position(row)
        if previous_position is not None and position is not None:
            row["step_distance_m"] = _distance_m(previous_position, position)
        if previous_time is not None and row["timestamp_seconds"] is not None:
            row["step_time_s"] = max(0.0, float(row["timestamp_seconds"]) - previous_time)
        if position is not None:
            previous_position = position
        if row["timestamp_seconds"] is not None:
            previous_time = float(row["timestamp_seconds"])


def _checks(
    rows: list[dict[str, Any]],
    image_root: Path | None,
    thresholds: CaptureCoverageThresholds,
) -> list[dict[str, str]]:
    summary = _summary(rows, thresholds.min_position_delta_m)
    checks = [
        _threshold_check("min_images", summary["row_count"] >= thresholds.min_images, f"{summary['row_count']} >= {thresholds.min_images}"),
        _threshold_check("image_references", summary["image_ref_count"] == summary["row_count"], f"{summary['image_ref_count']} of {summary['row_count']} rows"),
    ]
    if thresholds.require_image_files:
        if image_root is None:
            checks.append(_check("image_files_present", "fail", "--images is required when image files are required"))
        else:
            checks.append(
                _threshold_check(
                    "image_files_present",
                    summary["missing_image_count"] == 0,
                    f"missing_image_count={summary['missing_image_count']}",
                )
            )
    elif image_root is not None:
        checks.append(
            _threshold_check(
                "image_files_present",
                summary["missing_image_count"] == 0,
                f"missing_image_count={summary['missing_image_count']}",
            )
        )

    if thresholds.require_positions:
        checks.append(
            _threshold_check(
                "positions_present",
                summary["position_count"] == summary["row_count"],
                f"{summary['position_count']} of {summary['row_count']} rows",
            )
        )
    elif summary["position_count"] == 0:
        checks.append(_check("positions_present", "warn", "no capture positions available"))

    if thresholds.max_position_gap_m is not None and summary["position_gap_count"] > 0:
        checks.append(
            _threshold_check(
                "position_gaps",
                summary["max_step_distance_m"] <= thresholds.max_position_gap_m,
                f"max_step_distance_m={summary['max_step_distance_m']:.3f} <= {thresholds.max_position_gap_m:.3f}",
            )
        )
    elif thresholds.max_position_gap_m is not None:
        checks.append(_check("position_gaps", "pass", "no sequential position gaps"))

    checks.append(
        _threshold_check(
            "duplicate_positions",
            summary["duplicate_position_count"] == 0,
            f"duplicate_position_count={summary['duplicate_position_count']}",
        )
    )

    timestamps_required = thresholds.require_timestamps or thresholds.max_time_gap_s is not None
    if timestamps_required:
        checks.append(
            _threshold_check(
                "timestamps_present",
                summary["timestamp_count"] == summary["row_count"],
                f"{summary['timestamp_count']} of {summary['row_count']} rows",
            )
        )
    elif summary["timestamp_count"] == 0:
        checks.append(_check("timestamps_present", "warn", "no capture timestamps available"))

    if thresholds.max_time_gap_s is not None and summary["time_gap_count"] > 0:
        checks.append(
            _threshold_check(
                "time_gaps",
                summary["max_step_time_s"] <= thresholds.max_time_gap_s,
                f"max_step_time_s={summary['max_step_time_s']:.3f} <= {thresholds.max_time_gap_s:.3f}",
            )
        )
    elif thresholds.max_time_gap_s is not None:
        checks.append(_check("time_gaps", "pass", "no sequential time gaps"))
    return checks


def _summary(rows: list[dict[str, Any]], min_position_delta_m: float) -> dict[str, Any]:
    distances = [float(row["step_distance_m"]) for row in rows if row.get("step_distance_m") is not None]
    times = [float(row["step_time_s"]) for row in rows if row.get("step_time_s") is not None]
    duplicate_distances = [distance for distance in distances if distance < min_position_delta_m]
    return {
        "row_count": len(rows),
        "image_ref_count": sum(1 for row in rows if row.get("image")),
        "missing_image_count": sum(1 for row in rows if row.get("image_exists") is False),
        "position_count": sum(1 for row in rows if row.get("position_type") is not None),
        "timestamp_count": sum(1 for row in rows if row.get("timestamp_seconds") is not None),
        "position_gap_count": len(distances),
        "time_gap_count": len(times),
        "max_step_distance_m": max(distances) if distances else 0.0,
        "mean_step_distance_m": sum(distances) / len(distances) if distances else 0.0,
        "max_step_time_s": max(times) if times else 0.0,
        "mean_step_time_s": sum(times) / len(times) if times else 0.0,
        "duplicate_position_count": len(duplicate_distances),
    }


def _add_threshold_issues(rows: list[dict[str, Any]], thresholds: CaptureCoverageThresholds) -> None:
    for row in rows:
        distance = row.get("step_distance_m")
        if distance is not None and distance < thresholds.min_position_delta_m:
            row["issues"].append("duplicate_position")
        if thresholds.max_position_gap_m is not None and distance is not None and distance > thresholds.max_position_gap_m:
            row["issues"].append("position_gap")
        step_time = row.get("step_time_s")
        if thresholds.max_time_gap_s is not None and step_time is not None and step_time > thresholds.max_time_gap_s:
            row["issues"].append("time_gap")
        row["status"] = "warn" if row["issues"] else "pass"


def _position(row: dict[str, str]) -> dict[str, Any] | None:
    lat = _float_field(row, "latitude", "lat", "gps_latitude")
    lon = _float_field(row, "longitude", "lon", "lng", "gps_longitude")
    if lat is not None and lon is not None:
        return {"type": "lonlat", "x": lon, "y": lat}
    x = _float_field(row, "x", "easting", "easting_m", "map_x")
    y = _float_field(row, "y", "northing", "northing_m", "map_y")
    if x is not None and y is not None:
        return {"type": "projected", "x": x, "y": y}
    return None


def _timestamp(row: dict[str, str]) -> dict[str, Any] | None:
    raw = _first_text(row, "timestamp", "time", "capture_time", "datetime", "created_at")
    if raw is None:
        return None
    numeric = _present_float(raw)
    if numeric is not None:
        return {"raw": raw, "seconds": numeric}
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return {"raw": raw, "seconds": value.timestamp()}


def _image_exists(image_ref: str | None, image_root: Path | None) -> bool | None:
    if image_root is None or not image_ref:
        return None
    path = Path(image_ref)
    candidate = path if path.is_absolute() else image_root / image_ref
    return candidate.exists()


def _row_position(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("position_type") is None or row.get("x") is None or row.get("y") is None:
        return None
    return {"type": row["position_type"], "x": float(row["x"]), "y": float(row["y"])}


def _distance_m(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["type"] == "lonlat" and right["type"] == "lonlat":
        return _haversine_m(left["x"], left["y"], right["x"], right["y"])
    return sqrt((left["x"] - right["x"]) ** 2 + (left["y"] - right["y"]) ** 2)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_000.0
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * asin(sqrt(a))


def _write_capture_coverage_csv(report: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row",
        "image",
        "image_exists",
        "position_type",
        "x",
        "y",
        "timestamp",
        "step_distance_m",
        "step_time_s",
        "status",
        "issues",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("rows", []):
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fieldnames if key != "issues"},
                    "issues": " | ".join(row.get("issues", [])),
                }
            )


def _validate_thresholds(thresholds: CaptureCoverageThresholds) -> None:
    if thresholds.min_images < 1:
        raise ValueError("min_images must be >= 1")
    if thresholds.max_position_gap_m is not None and thresholds.max_position_gap_m <= 0:
        raise ValueError("max_position_gap_m must be positive")
    if thresholds.max_time_gap_s is not None and thresholds.max_time_gap_s <= 0:
        raise ValueError("max_time_gap_s must be positive")
    if thresholds.min_position_delta_m < 0:
        raise ValueError("min_position_delta_m must be non-negative")


def _float_field(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = _present_float(row.get(_normalize_key(key)))
        if value is not None:
            return value
    return None


def _first_text(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(_normalize_key(key))
        if value:
            return value
    return None


def _present_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _threshold_check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return _check(name, "pass" if passed else "fail", detail)


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _overall_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"
