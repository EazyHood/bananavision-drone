from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime import utc_now_iso


def audit_mission_delivery(
    output_json: str | Path,
    run_manifest: str | Path,
    mission_quality_report: str | Path,
    prediction_quality_report: str | Path,
    flight_check_report: str | Path | None = None,
    domain_check_report: str | Path | None = None,
    flight_log_report: str | Path | None = None,
    capture_coverage_report: str | Path | None = None,
    geo_accuracy_report: str | Path | None = None,
    preflight_report: str | Path | None = None,
    deployment_audit_report: str | Path | None = None,
    field_report: str | Path | None = None,
    min_detections: int = 1,
    allow_warn_quality: bool = False,
    require_flight_check: bool = True,
    require_domain_check: bool = True,
    require_capture_coverage: bool = False,
    require_geo_accuracy: bool = False,
    require_preflight: bool = False,
    require_deployment_audit: bool = False,
) -> dict[str, Any]:
    gates: list[dict[str, str]] = []
    run = _load_json_gate("run_manifest_present", run_manifest, gates, required=True)
    mission_quality = _load_json_gate("mission_quality_report_present", mission_quality_report, gates, required=True)
    prediction_quality = _load_json_gate("prediction_quality_report_present", prediction_quality_report, gates, required=True)
    flight_check = _load_json_gate("flight_check_report_present", flight_check_report, gates, required=require_flight_check)
    domain_check = _load_json_gate("domain_check_report_present", domain_check_report, gates, required=require_domain_check)
    flight_log = _load_json_gate("flight_log_report_present", flight_log_report, gates, required=False)
    capture_coverage = _load_json_gate(
        "capture_coverage_report_present",
        capture_coverage_report,
        gates,
        required=require_capture_coverage,
    )
    geo_accuracy = _load_json_gate("geo_accuracy_report_present", geo_accuracy_report, gates, required=require_geo_accuracy)
    preflight = _load_json_gate("preflight_report_present", preflight_report, gates, required=require_preflight)
    deployment = _load_json_gate("deployment_audit_report_present", deployment_audit_report, gates, required=require_deployment_audit)

    if run is not None:
        _check_run_manifest(run, gates, min_detections=min_detections)
    if mission_quality is not None:
        _check_quality("mission_quality", mission_quality, gates, allow_warn=allow_warn_quality)
    if prediction_quality is not None:
        _check_quality("prediction_quality", prediction_quality, gates, allow_warn=allow_warn_quality)
    if flight_check is not None:
        _check_status_report("flight_check", flight_check, gates)
    if domain_check is not None:
        _check_status_report("domain_check", domain_check, gates)
    if flight_log is not None:
        _check_flight_log(flight_log, gates)
    if capture_coverage is not None:
        _check_capture_coverage(capture_coverage, gates)
    if geo_accuracy is not None:
        _check_status_report("geo_accuracy", geo_accuracy, gates)
    if preflight is not None:
        _check_status_report("preflight", preflight, gates)
    if deployment is not None:
        _check_status_report("deployment_audit", deployment, gates)
    _check_file("field_report", field_report, gates)

    report = {
        "created_at": utc_now_iso(),
        "status": _overall_status(gates),
        "thresholds": {
            "min_detections": min_detections,
            "allow_warn_quality": allow_warn_quality,
            "require_flight_check": require_flight_check,
            "require_domain_check": require_domain_check,
            "require_capture_coverage": require_capture_coverage,
            "require_geo_accuracy": require_geo_accuracy,
            "require_preflight": require_preflight,
            "require_deployment_audit": require_deployment_audit,
        },
        "summary": {
            "image_count": None if run is None else run.get("image_count"),
            "total_detections": None if run is None else run.get("total_detections"),
            "mission_quality_status": None if mission_quality is None else mission_quality.get("status"),
            "prediction_quality_status": None if prediction_quality is None else prediction_quality.get("status"),
            "capture_coverage_status": None if capture_coverage is None else capture_coverage.get("status"),
        },
        "gates": gates,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _check_run_manifest(run: dict[str, Any], gates: list[dict[str, str]], min_detections: int) -> None:
    image_count = _int_value(run.get("image_count"), 0)
    total_detections = _int_value(run.get("total_detections"), 0)
    gates.append(_threshold_gate("mission_images_present", image_count > 0, f"image_count={image_count}"))
    gates.append(_threshold_gate("mission_detection_floor", total_detections >= min_detections, f"{total_detections} >= {min_detections}"))
    runtime = run.get("runtime", {}) or {}
    gates.append(
        _gate(
            "mission_runtime_fingerprint",
            "pass" if runtime.get("config_sha256") else "fail",
            "config hash present" if runtime.get("config_sha256") else "config hash missing",
        )
    )


def _check_quality(name: str, report: dict[str, Any], gates: list[dict[str, str]], allow_warn: bool) -> None:
    status = str(report.get("status", "missing"))
    if status == "pass":
        gates.append(_gate(f"{name}_passed", "pass", f"{name} passed"))
    elif status == "warn" and allow_warn:
        gates.append(_gate(f"{name}_passed", "warn", f"{name} warned; allowed"))
    else:
        gates.append(_gate(f"{name}_passed", "fail", f"{name} status={status}"))
    fail_count = _int_value(report.get("fail_count"), 0)
    gates.append(_threshold_gate(f"{name}_failures", fail_count == 0, f"fail_count={fail_count}"))


def _check_status_report(name: str, report: dict[str, Any], gates: list[dict[str, str]]) -> None:
    status = str(report.get("status", "missing"))
    gates.append(
        _gate(
            f"{name}_passed",
            "pass" if status == "pass" else "fail",
            f"{name} passed" if status == "pass" else f"{name} status={status}",
        )
    )


def _check_flight_log(report: dict[str, Any], gates: list[dict[str, str]]) -> None:
    _check_status_report("flight_log", report, gates)
    summary = report.get("summary", {}) or {}
    fail_count = _int_value(summary.get("fail_count"), 0)
    gates.append(_threshold_gate("flight_log_failures", fail_count == 0, f"fail_count={fail_count}"))


def _check_capture_coverage(report: dict[str, Any], gates: list[dict[str, str]]) -> None:
    _check_status_report("capture_coverage", report, gates)
    summary = report.get("summary", {}) or {}
    row_count = _int_value(summary.get("row_count"), 0)
    missing_images = _int_value(summary.get("missing_image_count"), 0)
    position_count = _int_value(summary.get("position_count"), 0)
    duplicate_positions = _int_value(summary.get("duplicate_position_count"), 0)
    gates.append(_threshold_gate("capture_rows", row_count > 0, f"row_count={row_count}"))
    gates.append(_threshold_gate("capture_missing_images", missing_images == 0, f"missing_image_count={missing_images}"))
    gates.append(_threshold_gate("capture_positions", position_count > 0, f"position_count={position_count}"))
    gates.append(
        _threshold_gate(
            "capture_duplicate_positions",
            duplicate_positions == 0,
            f"duplicate_position_count={duplicate_positions}",
        )
    )


def _load_json_gate(
    name: str,
    path: str | Path | None,
    gates: list[dict[str, str]],
    required: bool,
) -> dict[str, Any] | None:
    if path is None:
        if required:
            gates.append(_gate(name, "fail", "artifact path not provided"))
        return None
    path = Path(path)
    if not path.exists():
        gates.append(_gate(name, "fail" if required else "warn", f"artifact does not exist: {path}"))
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gates.append(_gate(name, "fail", f"artifact is not readable JSON: {exc}"))
        return None
    gates.append(_gate(name, "pass", str(path)))
    return payload


def _check_file(name: str, path: str | Path | None, gates: list[dict[str, str]]) -> None:
    if path is None:
        return
    path = Path(path)
    gates.append(
        _gate(
            f"{name}_present",
            "pass" if path.exists() and path.is_file() else "warn",
            str(path) if path.exists() and path.is_file() else f"artifact does not exist: {path}",
        )
    )


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _threshold_gate(name: str, passed: bool, detail: str) -> dict[str, str]:
    return _gate(name, "pass" if passed else "fail", detail)


def _gate(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _overall_status(gates: list[dict[str, str]]) -> str:
    if any(gate["status"] == "fail" for gate in gates):
        return "fail"
    if any(gate["status"] == "warn" for gate in gates):
        return "warn"
    return "pass"
