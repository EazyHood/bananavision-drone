from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__
from .runtime import file_sha256, utc_now_iso

DEFAULT_EVIDENCE_LABELS = (
    "run_manifest",
    "preflight_report",
    "mission_audit_report",
    "mission_quality_report",
    "prediction_quality_report",
    "flight_check_report",
    "flight_log_report",
    "capture_coverage_report",
    "domain_check_report",
    "geo_accuracy_report",
    "validation_plan_report",
    "truth_quality_report",
    "truth_coverage_report",
    "stratified_truth_coverage_report",
    "acceptance_report",
    "stratified_acceptance_report",
    "benchmark_report",
    "tuning_report",
    "cluster_review_report",
    "model_manifest",
    "model_card",
    "field_report",
    "release_audit_report",
    "release_package_manifest",
    "deployment_manifest",
    "deployment_smoke_report",
    "deployment_audit_report",
    "drone_ready_report",
    "model",
    "config",
)


def build_evidence_manifest(
    output_json: str | Path,
    artifacts: dict[str, str | Path | None],
    required_labels: list[str] | None = None,
) -> dict[str, Any]:
    required = set(required_labels or [])
    entries = [_artifact_entry(label, artifacts.get(label), label in required) for label in DEFAULT_EVIDENCE_LABELS]
    checks = _checks(entries, required)
    report = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "bananavision_version": __version__,
        "status": _overall_status(checks),
        "required_labels": sorted(required),
        "artifact_count": len(entries),
        "present_count": sum(1 for entry in entries if entry["exists"]),
        "missing_required_count": sum(1 for entry in entries if entry["required"] and not entry["exists"]),
        "failed_artifact_count": sum(1 for entry in entries if entry.get("reported_status") == "fail"),
        "warn_artifact_count": sum(1 for entry in entries if entry.get("reported_status") == "warn"),
        "checks": checks,
        "artifacts": entries,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _artifact_entry(label: str, path: str | Path | None, required: bool) -> dict[str, Any]:
    path_value = None if path is None else Path(path)
    exists = path_value is not None and path_value.exists() and path_value.is_file()
    payload = _load_json(path_value) if exists and path_value is not None and path_value.suffix.lower() == ".json" else None
    return {
        "label": label,
        "path": None if path_value is None else str(path_value),
        "required": required,
        "exists": exists,
        "kind": _artifact_kind(path_value),
        "sha256": file_sha256(path_value) if exists else None,
        "size_bytes": path_value.stat().st_size if exists and path_value is not None else None,
        "reported_status": _reported_status(payload),
        "summary": _summary(payload),
    }


def _checks(entries: list[dict[str, Any]], required: set[str]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    labels = {entry["label"] for entry in entries}
    for label in sorted(required):
        if label not in labels:
            checks.append(_check(f"required_label:{label}", "fail", "unknown evidence label"))
    for entry in entries:
        label = str(entry["label"])
        if entry["required"]:
            checks.append(
                _check(
                    f"required_artifact:{label}",
                    "pass" if entry["exists"] else "fail",
                    str(entry.get("path") or "path not provided"),
                )
            )
        status = entry.get("reported_status")
        if status == "fail":
            checks.append(_check(f"artifact_status:{label}", "fail", "reported status=fail"))
        elif status == "warn":
            checks.append(_check(f"artifact_status:{label}", "warn", "reported status=warn"))
        elif status == "pass":
            checks.append(_check(f"artifact_status:{label}", "pass", "reported status=pass"))
    return checks


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _reported_status(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    if "status" in payload:
        value = str(payload.get("status")).lower()
        if value in {"pass", "warn", "fail"}:
            return value
    if "passed" in payload:
        return "pass" if payload.get("passed") is True else "fail"
    if payload.get("package_status") == "release" and payload.get("release_audit_status") == "pass":
        return "pass"
    if payload.get("package_status") == "exploratory":
        return "warn"
    return None


def _summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    summary: dict[str, Any] = {}
    for key in [
        "status",
        "passed",
        "image_count",
        "truth_count",
        "cluster_count",
        "cluster_truth_count",
        "issue_count",
        "artifact_count",
        "verified_artifact_count",
        "package_status",
        "release_audit_status",
    ]:
        if key in payload:
            summary[key] = payload[key]
    metrics = payload.get("metrics", {}) or {}
    for key in ["count_error_rate", "precision", "recall", "f1", "cluster_recall", "fully_detected_cluster_rate"]:
        if key in metrics:
            summary[key] = metrics[key]
    return summary


def _artifact_kind(path: Path | None) -> str:
    if path is None:
        return "missing"
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".pt", ".onnx", ".engine"}:
        return "model"
    if suffix in {".yaml", ".yml"}:
        return "config"
    if suffix == ".md":
        return "markdown"
    if suffix == ".zip":
        return "zip"
    return suffix.lstrip(".") or "file"


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _overall_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"
