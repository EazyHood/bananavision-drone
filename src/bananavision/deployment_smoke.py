from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import InferenceConfig
from .pipeline import predict_path
from .readiness import run_preflight
from .runtime import runtime_fingerprint, utc_now_iso


def run_deployment_smoke_test(
    input_path: str | Path,
    artifacts_dir: str | Path,
    config: InferenceConfig,
    min_images: int = 1,
    min_detections: int = 1,
    max_image_latency_ms: float | None = None,
    min_free_gb: float = 1.0,
) -> dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    gates: list[dict[str, str]] = []
    preflight = run_preflight(config, input_path=input_path, output_dir=artifacts_dir, min_free_gb=min_free_gb)
    gates.append(
        _gate(
            "smoke_preflight_no_fail",
            "pass" if preflight.get("status") != "fail" else "fail",
            f"preflight status={preflight.get('status')}",
        )
    )

    results = []
    inference_error = None
    try:
        results = predict_path(input_path, artifacts_dir, config)
        gates.append(_gate("smoke_inference_completed", "pass", f"{len(results)} image(s) processed"))
    except Exception as exc:
        inference_error = str(exc)
        gates.append(_gate("smoke_inference_completed", "fail", inference_error))

    image_count = len(results)
    total_detections = sum(result.count for result in results)
    latencies = [float(result.meta.get("elapsed_ms", 0.0) or 0.0) for result in results]
    max_latency = max(latencies) if latencies else None
    mean_latency = sum(latencies) / len(latencies) if latencies else None

    gates.append(
        _threshold_gate(
            "smoke_min_images",
            image_count >= min_images,
            f"{image_count} >= {min_images}",
        )
    )
    gates.append(
        _threshold_gate(
            "smoke_min_detections",
            total_detections >= min_detections,
            f"{total_detections} >= {min_detections}",
        )
    )
    if max_image_latency_ms is not None:
        gates.append(
            _threshold_gate(
                "smoke_max_image_latency",
                max_latency is not None and max_latency <= max_image_latency_ms,
                "missing latency"
                if max_latency is None
                else f"{max_latency:.3f}ms <= {max_image_latency_ms:.3f}ms",
            )
        )

    artifacts = _artifacts(artifacts_dir, results)
    missing_outputs = [name for name, path in artifacts.items() if path is not None and not Path(path).exists()]
    gates.append(
        _threshold_gate(
            "smoke_outputs_written",
            not missing_outputs and bool(artifacts.get("run_manifest")),
            "all expected outputs written" if not missing_outputs else "missing " + ", ".join(missing_outputs),
        )
    )

    report = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "status": _overall_status(gates),
        "input_path": str(input_path),
        "artifacts_dir": str(artifacts_dir),
        "thresholds": {
            "min_images": min_images,
            "min_detections": min_detections,
            "max_image_latency_ms": max_image_latency_ms,
            "min_free_gb": min_free_gb,
        },
        "runtime": runtime_fingerprint(config),
        "preflight_status": preflight.get("status"),
        "preflight_checks": preflight.get("checks", []),
        "image_count": image_count,
        "total_detections": total_detections,
        "latency_ms": {
            "max": None if max_latency is None else round(max_latency, 3),
            "mean": None if mean_latency is None else round(mean_latency, 3),
        },
        "inference_error": inference_error,
        "artifacts": artifacts,
        "images": [
            {
                "image": str(result.image_path),
                "width": result.width,
                "height": result.height,
                "count": result.count,
                "elapsed_ms": result.meta.get("elapsed_ms"),
            }
            for result in results
        ],
        "gates": gates,
    }
    return report


def write_deployment_smoke_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_json


def _artifacts(artifacts_dir: Path, results: list[Any]) -> dict[str, str | None]:
    first_stem = results[0].image_path.stem if results else None
    return {
        "run_manifest": str(artifacts_dir / "run_manifest.json"),
        "mission_csv": str(artifacts_dir / "mission.detections.csv"),
        "mission_geojson": str(artifacts_dir / "mission.detections.geojson"),
        "mission_kml": str(artifacts_dir / "mission.detections.kml"),
        "first_detection_json": None if first_stem is None else str(artifacts_dir / f"{first_stem}.detections.json"),
        "first_detection_csv": None if first_stem is None else str(artifacts_dir / f"{first_stem}.detections.csv"),
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
