from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .models import InferenceConfig
from .runtime import utc_now_iso


@dataclass(frozen=True)
class FlightProfile:
    gsd_cm: float | None = None
    altitude_m: float | None = None
    sensor_width_mm: float | None = None
    focal_length_mm: float | None = None
    image_width_px: int | None = None
    front_overlap: float | None = None
    side_overlap: float | None = None
    speed_mps: float | None = None
    exposure_ms: float | None = None

    def normalized(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["front_overlap"] = _normalize_percent(self.front_overlap)
        payload["side_overlap"] = _normalize_percent(self.side_overlap)
        return payload


@dataclass(frozen=True)
class FlightEnvelope:
    target_gsd_cm: float
    max_gsd_drift_ratio: float = 0.20
    min_front_overlap: float = 70.0
    min_side_overlap: float = 70.0
    max_motion_blur_px: float = 1.5

    @classmethod
    def from_config(
        cls,
        config: InferenceConfig,
        max_gsd_drift_ratio: float = 0.20,
        min_front_overlap: float = 70.0,
        min_side_overlap: float = 70.0,
        max_motion_blur_px: float = 1.5,
    ) -> FlightEnvelope:
        return cls(
            target_gsd_cm=config.gsd_cm,
            max_gsd_drift_ratio=max_gsd_drift_ratio,
            min_front_overlap=min_front_overlap,
            min_side_overlap=min_side_overlap,
            max_motion_blur_px=max_motion_blur_px,
        )


def estimate_gsd_cm(
    altitude_m: float,
    sensor_width_mm: float,
    focal_length_mm: float,
    image_width_px: int,
) -> float:
    _require_positive("altitude_m", altitude_m)
    _require_positive("sensor_width_mm", sensor_width_mm)
    _require_positive("focal_length_mm", focal_length_mm)
    _require_positive("image_width_px", image_width_px)
    ground_width_m = altitude_m * (sensor_width_mm / focal_length_mm)
    return (ground_width_m / image_width_px) * 100.0


def audit_flight_profile(
    profile: FlightProfile,
    envelope: FlightEnvelope,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    observed_gsd_cm, gsd_source = _observed_gsd(profile)
    normalized = profile.normalized()
    normalized["observed_gsd_cm"] = observed_gsd_cm
    normalized["gsd_source"] = gsd_source

    if observed_gsd_cm is None:
        checks.append(_check("gsd_available", "fail", "No measured GSD or complete camera geometry was provided."))
    else:
        checks.append(_check("gsd_available", "pass", f"{observed_gsd_cm:.3f} cm/px from {gsd_source}"))
        lower = envelope.target_gsd_cm * (1.0 - envelope.max_gsd_drift_ratio)
        upper = envelope.target_gsd_cm * (1.0 + envelope.max_gsd_drift_ratio)
        checks.append(
            _check(
                "gsd_within_validated_range",
                "pass" if lower <= observed_gsd_cm <= upper else "fail",
                f"{observed_gsd_cm:.3f} cm/px within [{lower:.3f}, {upper:.3f}] cm/px",
                observed=observed_gsd_cm,
                lower=lower,
                upper=upper,
            )
        )

    _overlap_check(checks, "front_overlap", normalized["front_overlap"], envelope.min_front_overlap)
    _overlap_check(checks, "side_overlap", normalized["side_overlap"], envelope.min_side_overlap)
    _motion_blur_check(checks, profile, observed_gsd_cm, envelope.max_motion_blur_px)

    report = {
        "created_at": utc_now_iso(),
        "status": _overall_status(checks),
        "profile": normalized,
        "envelope": asdict(envelope),
        "checks": checks,
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit_flight_log(
    log_csv: str | Path,
    envelope: FlightEnvelope,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    records = _read_flight_log(log_csv)
    row_reports = []
    for index, record in enumerate(records, start=1):
        profile = FlightProfile(
            gsd_cm=_float_field(record, "gsd_cm", "gsd_cm_px", "gsd"),
            altitude_m=_float_field(record, "altitude_m", "relative_altitude_m", "agl_m"),
            sensor_width_mm=_float_field(record, "sensor_width_mm"),
            focal_length_mm=_float_field(record, "focal_length_mm"),
            image_width_px=_int_field(record, "image_width_px", "width_px"),
            front_overlap=_float_field(record, "front_overlap", "front_overlap_percent"),
            side_overlap=_float_field(record, "side_overlap", "side_overlap_percent"),
            speed_mps=_float_field(record, "speed_mps", "groundspeed_mps", "ground_speed_mps"),
            exposure_ms=_float_field(record, "exposure_ms", "shutter_ms"),
        )
        try:
            row_report = audit_flight_profile(profile, envelope)
            failed = [check["name"] for check in row_report["checks"] if check["status"] == "fail"]
            warned = [check["name"] for check in row_report["checks"] if check["status"] == "warn"]
            status = row_report["status"]
            normalized = row_report["profile"]
        except ValueError as exc:
            failed = ["invalid_flight_log_row"]
            warned = []
            status = "fail"
            normalized = profile.normalized()
            normalized["error"] = str(exc)
        row_reports.append(
            {
                "row": index,
                "id": record.get("image") or record.get("filename") or record.get("timestamp") or f"row-{index}",
                "status": status,
                "profile": normalized,
                "failed_checks": failed,
                "warn_checks": warned,
            }
        )

    status = _overall_status(row_reports)
    report = {
        "created_at": utc_now_iso(),
        "status": status,
        "log_csv": str(log_csv),
        "envelope": asdict(envelope),
        "summary": _flight_log_summary(row_reports),
        "rows": row_reports,
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_flight_log_csv(report, output_path.with_suffix(".csv"))
    return report


def _observed_gsd(profile: FlightProfile) -> tuple[float | None, str]:
    if profile.gsd_cm is not None:
        _require_positive("gsd_cm", profile.gsd_cm)
        return profile.gsd_cm, "measured"
    if (
        profile.altitude_m is not None
        and profile.sensor_width_mm is not None
        and profile.focal_length_mm is not None
        and profile.image_width_px is not None
    ):
        return (
            estimate_gsd_cm(
                profile.altitude_m,
                profile.sensor_width_mm,
                profile.focal_length_mm,
                profile.image_width_px,
            ),
            "estimated_from_camera_geometry",
        )
    return None, "missing"


def _overlap_check(checks: list[dict[str, Any]], name: str, observed: float | None, minimum: float) -> None:
    if observed is None:
        checks.append(_check(name, "fail", f"{name} was not provided; required >= {minimum:.1f}%"))
        return
    _require_non_negative(name, observed)
    checks.append(
        _check(
            name,
            "pass" if observed >= minimum else "fail",
            f"{observed:.1f}% >= {minimum:.1f}%",
            observed=observed,
            minimum=minimum,
        )
    )


def _motion_blur_check(
    checks: list[dict[str, Any]],
    profile: FlightProfile,
    observed_gsd_cm: float | None,
    max_motion_blur_px: float,
) -> None:
    if profile.speed_mps is None or profile.exposure_ms is None:
        checks.append(_check("motion_blur", "warn", "speed_mps or exposure_ms missing; blur not computed."))
        return
    _require_non_negative("speed_mps", profile.speed_mps)
    _require_positive("exposure_ms", profile.exposure_ms)
    if observed_gsd_cm is None:
        checks.append(_check("motion_blur", "fail", "Cannot compute motion blur without GSD."))
        return
    gsd_m = observed_gsd_cm / 100.0
    blur_px = (profile.speed_mps * (profile.exposure_ms / 1000.0)) / gsd_m
    checks.append(
        _check(
            "motion_blur",
            "pass" if blur_px <= max_motion_blur_px else "fail",
            f"{blur_px:.3f}px <= {max_motion_blur_px:.3f}px",
            observed=blur_px,
            maximum=max_motion_blur_px,
        )
    )


def _normalize_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def _check(name: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    payload = {"name": name, "status": status, "detail": detail}
    payload.update(extra)
    return payload


def _overall_status(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"


def _read_flight_log(log_csv: str | Path) -> list[dict[str, str]]:
    path = Path(log_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{_normalize_key(key): (value or "").strip() for key, value in row.items() if key} for row in reader]
    if not rows:
        raise ValueError(f"No flight log rows found in CSV: {path}")
    return rows


def _flight_log_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gsd_values = [_present_float((row.get("profile") or {}).get("observed_gsd_cm")) for row in rows]
    blur_values = [
        _motion_blur_from_profile(row.get("profile") or {})
        for row in rows
        if _motion_blur_from_profile(row.get("profile") or {}) is not None
    ]
    gsd_values = [value for value in gsd_values if value is not None]
    return {
        "row_count": len(rows),
        "pass_count": sum(1 for row in rows if row["status"] == "pass"),
        "warn_count": sum(1 for row in rows if row["status"] == "warn"),
        "fail_count": sum(1 for row in rows if row["status"] == "fail"),
        "mean_gsd_cm": mean(gsd_values) if gsd_values else None,
        "min_gsd_cm": min(gsd_values) if gsd_values else None,
        "max_gsd_cm": max(gsd_values) if gsd_values else None,
        "max_motion_blur_px": max(blur_values) if blur_values else None,
    }


def _write_flight_log_csv(report: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "id", "status", "observed_gsd_cm", "failed_checks", "warn_checks"])
        writer.writeheader()
        for row in report.get("rows", []):
            profile = row.get("profile", {}) or {}
            writer.writerow(
                {
                    "row": row.get("row"),
                    "id": row.get("id"),
                    "status": row.get("status"),
                    "observed_gsd_cm": profile.get("observed_gsd_cm"),
                    "failed_checks": ",".join(row.get("failed_checks", [])),
                    "warn_checks": ",".join(row.get("warn_checks", [])),
                }
            )


def _motion_blur_from_profile(profile: dict[str, Any]) -> float | None:
    gsd_cm = _present_float(profile.get("observed_gsd_cm"))
    speed = _present_float(profile.get("speed_mps"))
    exposure = _present_float(profile.get("exposure_ms"))
    if gsd_cm is None or speed is None or exposure is None:
        return None
    return (speed * (exposure / 1000.0)) / (gsd_cm / 100.0)


def _float_field(record: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = _present_float(record.get(_normalize_key(key)))
        if value is not None:
            return value
    return None


def _int_field(record: dict[str, str], *keys: str) -> int | None:
    value = _float_field(record, *keys)
    return None if value is None else int(value)


def _present_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _require_positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
