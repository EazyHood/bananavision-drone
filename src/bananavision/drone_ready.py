from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .deployment_audit import audit_deployment
from .deployment_smoke import run_deployment_smoke_test, write_deployment_smoke_report
from .evidence_manifest import build_evidence_manifest
from .models import InferenceConfig
from .readiness import run_preflight, write_preflight_report
from .runtime import utc_now_iso


def run_drone_ready_check(
    output_dir: str | Path,
    release_package: str | Path,
    deployment_manifest: str | Path,
    smoke_image: str | Path,
    config: InferenceConfig,
    config_path: str | Path | None = None,
    min_detections: int = 1,
    max_image_latency_ms: float | None = None,
    min_free_gb: float = 1.0,
    allow_warn_preflight: bool = False,
    allow_exploratory_package: bool = False,
    require_deployment_artifacts: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight_path = output_dir / "preflight_report.json"
    smoke_path = output_dir / "deployment_smoke_report.json"
    deployment_audit_path = output_dir / "deployment_audit.json"
    evidence_manifest_path = output_dir / "evidence_manifest.json"
    report_path = output_dir / "drone_ready_report.json"

    preflight = run_preflight(config, input_path=smoke_image, output_dir=output_dir, min_free_gb=min_free_gb)
    write_preflight_report(preflight, preflight_path)

    smoke = run_deployment_smoke_test(
        smoke_image,
        output_dir / "smoke_artifacts",
        config,
        min_detections=min_detections,
        max_image_latency_ms=max_image_latency_ms,
        min_free_gb=min_free_gb,
    )
    write_deployment_smoke_report(smoke, smoke_path)

    deployment = audit_deployment(
        deployment_audit_path,
        release_package=release_package,
        preflight_report=preflight_path,
        deployment_manifest=deployment_manifest,
        deployment_smoke_report=smoke_path,
        allow_warn_preflight=allow_warn_preflight,
        allow_exploratory_package=allow_exploratory_package,
        require_deployment_artifacts=require_deployment_artifacts,
        require_smoke_test=True,
    )

    evidence = build_evidence_manifest(
        evidence_manifest_path,
        {
            "preflight_report": preflight_path,
            "deployment_manifest": deployment_manifest,
            "deployment_smoke_report": smoke_path,
            "deployment_audit_report": deployment_audit_path,
            "release_package_manifest": _release_package_manifest_path(release_package),
            "config": config_path,
        },
        required_labels=[
            "preflight_report",
            "deployment_manifest",
            "deployment_smoke_report",
            "deployment_audit_report",
            *([] if config_path is None else ["config"]),
        ],
    )

    report = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "status": _overall_status([preflight, smoke, deployment, evidence]),
        "release_package": str(release_package),
        "deployment_manifest": str(deployment_manifest),
        "smoke_image": str(smoke_image),
        "artifacts": {
            "preflight_report": str(preflight_path),
            "deployment_smoke_report": str(smoke_path),
            "deployment_audit_report": str(deployment_audit_path),
            "evidence_manifest": str(evidence_manifest_path),
            "drone_ready_report": str(report_path),
        },
        "summary": {
            "preflight_status": preflight.get("status"),
            "deployment_smoke_status": smoke.get("status"),
            "deployment_audit_status": deployment.get("status"),
            "evidence_manifest_status": evidence.get("status"),
            "smoke_image_count": smoke.get("image_count"),
            "smoke_total_detections": smoke.get("total_detections"),
            "smoke_latency_ms": smoke.get("latency_ms"),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _release_package_manifest_path(release_package: str | Path) -> Path | None:
    path = Path(release_package)
    if path.is_dir():
        manifest = path / "release_package_manifest.json"
        return manifest if manifest.exists() else None
    if path.is_file() and path.name == "release_package_manifest.json":
        return path
    return None


def _overall_status(reports: list[dict[str, Any]]) -> str:
    statuses = [str(report.get("status", "missing")) for report in reports]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "warn" for status in statuses):
        return "warn"
    return "pass"
