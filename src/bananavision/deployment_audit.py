from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .release_package import verify_release_package
from .runtime import utc_now_iso


def audit_deployment(
    output_json: str | Path,
    release_package: str | Path,
    preflight_report: str | Path,
    deployment_manifest: str | Path,
    deployment_smoke_report: str | Path | None = None,
    allow_warn_preflight: bool = False,
    allow_exploratory_package: bool = False,
    require_deployment_artifacts: bool = True,
    require_smoke_test: bool = True,
) -> dict[str, Any]:
    gates: list[dict[str, str]] = []
    package_report = verify_release_package(
        release_package,
        allow_exploratory=allow_exploratory_package,
        require_deployment_artifacts=require_deployment_artifacts,
    )
    _check_package_report(package_report, gates, allow_exploratory_package=allow_exploratory_package)

    preflight = _load_json_gate("preflight_report_present", preflight_report, gates, required=True)
    if preflight is not None:
        _check_preflight(preflight, gates, allow_warn=allow_warn_preflight)

    manifest = _load_json_gate("deployment_manifest_present", deployment_manifest, gates, required=True)
    if manifest is not None:
        _check_deployment_manifest(manifest, gates)

    smoke = _load_json_gate(
        "deployment_smoke_report_present",
        deployment_smoke_report,
        gates,
        required=require_smoke_test,
    )
    if smoke is not None:
        _check_deployment_smoke(smoke, gates)

    report = {
        "created_at": utc_now_iso(),
        "status": _overall_status(gates),
        "release_package": str(release_package),
        "preflight_report": str(preflight_report),
        "deployment_manifest": str(deployment_manifest),
        "deployment_smoke_report": None if deployment_smoke_report is None else str(deployment_smoke_report),
        "thresholds": {
            "allow_warn_preflight": allow_warn_preflight,
            "allow_exploratory_package": allow_exploratory_package,
            "require_deployment_artifacts": require_deployment_artifacts,
            "require_smoke_test": require_smoke_test,
        },
        "package_summary": {
            "status": package_report.get("status"),
            "package_status": package_report.get("package_status"),
            "release_audit_status": package_report.get("release_audit_status"),
            "artifact_count": package_report.get("artifact_count"),
            "verified_artifact_count": package_report.get("verified_artifact_count"),
        },
        "smoke_summary": _smoke_summary(smoke),
        "gates": gates,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _check_package_report(
    package_report: dict[str, Any],
    gates: list[dict[str, str]],
    allow_exploratory_package: bool,
) -> None:
    status = str(package_report.get("status", "missing"))
    if status == "pass":
        gates.append(_gate("release_package_verified", "pass", "release package verification passed"))
    elif status == "warn" and allow_exploratory_package:
        gates.append(_gate("release_package_verified", "warn", "exploratory package verification warned; allowed"))
    else:
        gates.append(_gate("release_package_verified", "fail", f"release package verify status={status}"))
    artifact_count = int(package_report.get("artifact_count", 0) or 0)
    verified_count = int(package_report.get("verified_artifact_count", 0) or 0)
    gates.append(
        _threshold_gate(
            "release_package_artifacts_verified",
            artifact_count > 0 and verified_count == artifact_count,
            f"{verified_count}/{artifact_count} artifact(s) verified",
        )
    )
    release_audit_status = str(package_report.get("release_audit_status", "missing"))
    gates.append(
        _gate(
            "release_audit_passed",
            "pass" if release_audit_status == "pass" else "fail",
            f"release_audit_status={release_audit_status}",
        )
    )


def _check_preflight(preflight: dict[str, Any], gates: list[dict[str, str]], allow_warn: bool) -> None:
    status = str(preflight.get("status", "missing"))
    if status == "pass":
        gates.append(_gate("preflight_passed", "pass", "preflight passed"))
    elif status == "warn" and allow_warn:
        gates.append(_gate("preflight_passed", "warn", "preflight warned; allowed by operator"))
    else:
        gates.append(_gate("preflight_passed", "fail", f"preflight status={status}"))

    checks = {str(check.get("name")): str(check.get("status")) for check in preflight.get("checks", []) if isinstance(check, dict)}
    gates.append(_gate("preflight_config_valid", "pass" if checks.get("config") == "pass" else "fail", f"config={checks.get('config', 'missing')}"))
    gates.append(_gate("preflight_output_writable", "pass" if checks.get("output_dir") == "pass" else "fail", f"output_dir={checks.get('output_dir', 'missing')}"))
    if checks.get("model_file") is not None:
        gates.append(_gate("preflight_model_file", "pass" if checks.get("model_file") == "pass" else "fail", f"model_file={checks.get('model_file')}"))
    runtime = preflight.get("runtime", {}) or {}
    gates.append(_gate("preflight_runtime_fingerprint", "pass" if runtime.get("config_sha256") else "fail", "runtime config hash present" if runtime.get("config_sha256") else "runtime config hash missing"))


def _check_deployment_manifest(manifest: dict[str, Any], gates: list[dict[str, str]]) -> None:
    gates.append(
        _gate(
            "deployment_manifest_schema",
            "pass" if manifest.get("schema_version") == 1 else "fail",
            f"schema_version={manifest.get('schema_version')}",
        )
    )
    services = manifest.get("services", {}) or {}
    mission_watch = services.get("mission_watch", {}) or {}
    api = services.get("api", {}) or {}
    preflight_command = str(mission_watch.get("preflight_command", ""))
    start_command = str(mission_watch.get("start_command", ""))
    api_start_command = str(api.get("start_command", ""))
    health_commands = [str(item) for item in api.get("health_commands", []) or []]
    gates.append(_gate("deployment_preflight_command", "pass" if " preflight " in f" {preflight_command} " else "fail", preflight_command or "missing"))
    gates.append(_gate("deployment_mission_watch_command", "pass" if " mission-watch " in f" {start_command} " else "fail", start_command or "missing"))
    gates.append(
        _gate(
            "deployment_api_key_env",
            "pass" if "--api-key-env" in api_start_command else "fail",
            api_start_command or "missing",
        )
    )
    gates.append(
        _gate(
            "deployment_api_upload_limit",
            "pass" if "--max-upload-mb" in api_start_command else "fail",
            api_start_command or "missing",
        )
    )
    gates.append(
        _gate(
            "deployment_api_ready_check",
            "pass" if any("/ready" in command for command in health_commands) else "fail",
            "API /ready health command present" if any("/ready" in command for command in health_commands) else "API /ready health command missing",
        )
    )
    generated = {Path(str(path)).name for path in manifest.get("generated_files", []) or []}
    for name in ["bananavision-mission-watch.service", "bananavision-api.service", "deployment_manifest.json"]:
        gates.append(_gate(f"deployment_generated:{name}", "pass" if name in generated else "fail", "present" if name in generated else "missing"))


def _check_deployment_smoke(smoke: dict[str, Any], gates: list[dict[str, str]]) -> None:
    status = str(smoke.get("status", "missing"))
    gates.append(
        _gate(
            "deployment_smoke_passed",
            "pass" if status == "pass" else "fail",
            f"deployment smoke status={status}",
        )
    )
    gates.append(
        _threshold_gate(
            "deployment_smoke_images",
            int(smoke.get("image_count", 0) or 0) > 0,
            f"image_count={smoke.get('image_count', 0)}",
        )
    )
    runtime = smoke.get("runtime", {}) or {}
    gates.append(
        _gate(
            "deployment_smoke_runtime_fingerprint",
            "pass" if runtime.get("config_sha256") else "fail",
            "runtime config hash present" if runtime.get("config_sha256") else "runtime config hash missing",
        )
    )
    artifacts = smoke.get("artifacts", {}) or {}
    required_outputs = ["run_manifest", "mission_csv", "mission_geojson", "mission_kml"]
    missing = [
        name
        for name in required_outputs
        if not artifacts.get(name) or not Path(str(artifacts.get(name))).exists()
    ]
    gates.append(
        _gate(
            "deployment_smoke_outputs",
            "pass" if not missing else "fail",
            "output files present" if not missing else "missing " + ", ".join(missing),
        )
    )


def _load_json_gate(
    name: str,
    path: str | Path | None,
    gates: list[dict[str, str]],
    required: bool = False,
) -> dict[str, Any] | None:
    if path is None:
        gates.append(_gate(name, "fail" if required else "warn", "artifact path not provided"))
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


def _smoke_summary(smoke: dict[str, Any] | None) -> dict[str, Any] | None:
    if smoke is None:
        return None
    return {
        "status": smoke.get("status"),
        "image_count": smoke.get("image_count"),
        "total_detections": smoke.get("total_detections"),
        "latency_ms": smoke.get("latency_ms"),
    }


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
