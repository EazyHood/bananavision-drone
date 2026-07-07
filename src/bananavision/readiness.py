from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .geo import describe_image_georeference
from .models import InferenceConfig
from .pipeline import iter_images
from .runtime import runtime_fingerprint, utc_now_iso


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "warn"}

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def run_preflight(
    config: InferenceConfig,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    min_free_gb: float = 2.0,
) -> dict[str, Any]:
    checks: list[ReadinessCheck] = []
    checks.append(_check_python())
    checks.append(_check_config(config))
    checks.extend(_check_detector_dependencies(config))
    checks.extend(_check_paths(input_path, output_dir))
    checks.append(_check_disk(output_dir or ".", min_free_gb=min_free_gb))
    checks.append(_check_gpu(config))
    checks.append(_check_georeference(input_path))
    status = "pass" if all(check.status == "pass" for check in checks) else "warn"
    if any(check.status == "fail" for check in checks):
        status = "fail"
    return {
        "created_at": utc_now_iso(),
        "status": status,
        "runtime": runtime_fingerprint(config),
        "checks": [check.to_dict() for check in checks],
    }


def write_preflight_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_json


def _check_python() -> ReadinessCheck:
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return ReadinessCheck("python", "pass", f"Python {version.major}.{version.minor}.{version.micro}")
    return ReadinessCheck("python", "fail", "Python 3.10 or newer is required")


def _check_config(config: InferenceConfig) -> ReadinessCheck:
    try:
        config.validate()
    except Exception as exc:
        return ReadinessCheck("config", "fail", str(exc))
    return ReadinessCheck("config", "pass", "Configuration is valid")


def _check_detector_dependencies(config: InferenceConfig) -> list[ReadinessCheck]:
    checks = [ReadinessCheck("pillow", "pass" if _module_exists("PIL") else "fail", "Pillow image backend")]
    if config.detector in {"yolo-seg", "yolo-ensemble"}:
        paths = (
            [config.model_path or ""]
            if config.detector == "yolo-seg"
            else list(config.ensemble_model_paths or [])
        )
        for model_path in (Path(p) for p in paths):
            checks.append(
                ReadinessCheck(
                    "model_file",
                    "pass" if model_path.exists() else "fail",
                    str(model_path) if model_path.exists() else f"Model not found: {model_path}",
                )
            )
        checks.append(
            ReadinessCheck(
                "ultralytics",
                "pass" if _module_exists("ultralytics") else "fail",
                "Ultralytics is required for YOLO inference",
            )
        )
        if config.detector == "yolo-ensemble":
            checks.append(
                ReadinessCheck(
                    "ensemble_boxes",
                    "pass" if _module_exists("ensemble_boxes") else "fail",
                    "ensemble-boxes is required for yolo-ensemble inference",
                )
            )
    return checks


def _check_paths(input_path: str | Path | None, output_dir: str | Path | None) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    if input_path is not None:
        try:
            images = iter_images(input_path)
            checks.append(ReadinessCheck("input_images", "pass", f"{len(images)} supported image(s) found"))
        except Exception as exc:
            checks.append(ReadinessCheck("input_images", "fail", str(exc)))
    if output_dir is not None:
        try:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            probe = output / ".bananavision_write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            checks.append(ReadinessCheck("output_dir", "pass", f"Writable: {output}"))
        except Exception as exc:
            checks.append(ReadinessCheck("output_dir", "fail", str(exc)))
    return checks


def _check_disk(path: str | Path, min_free_gb: float) -> ReadinessCheck:
    usage = shutil.disk_usage(Path(path).resolve())
    free_gb = usage.free / (1024**3)
    if free_gb >= min_free_gb:
        return ReadinessCheck("disk_free", "pass", f"{free_gb:.2f} GB free")
    return ReadinessCheck("disk_free", "warn", f"{free_gb:.2f} GB free; recommended >= {min_free_gb:.2f} GB")


def _check_gpu(config: InferenceConfig) -> ReadinessCheck:
    if config.detector not in {"yolo-seg", "yolo-ensemble"}:
        return ReadinessCheck("gpu", "warn", "RGB baseline can run on CPU; GPU not required")
    if not _module_exists("torch"):
        return ReadinessCheck("gpu", "warn", "Torch not installed; cannot verify CUDA")
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return ReadinessCheck("gpu", "pass", f"CUDA available: {torch.cuda.get_device_name(0)}")
        return ReadinessCheck("gpu", "warn", "CUDA not available; yolo-seg may be slow on CPU")
    except Exception as exc:
        return ReadinessCheck("gpu", "warn", f"Could not verify GPU: {exc}")


def _check_georeference(input_path: str | Path | None) -> ReadinessCheck:
    if input_path is None:
        return ReadinessCheck("georeference", "warn", "No input path supplied; georeference not checked")
    try:
        images = iter_images(input_path)
    except Exception:
        return ReadinessCheck("georeference", "warn", "No readable image for georeference check")
    if not images:
        return ReadinessCheck("georeference", "warn", "No image for georeference check")
    georef = describe_image_georeference(images[0])
    if georef is None:
        return ReadinessCheck(
            "georeference",
            "warn",
            "No world file, GeoTIFF transform, or EXIF GPS geotag found for first image",
        )
    if georef["type"] == "exif_gps":
        return ReadinessCheck(
            "georeference",
            "pass",
            f"EXIF GPS geotag available for {images[0].name}; use orthomosaic/world file for plant-level coordinates",
        )
    return ReadinessCheck("georeference", "pass", f"Pixel geotransform available for {images[0].name}")


def _module_exists(module: str) -> bool:
    return importlib.util.find_spec(module) is not None
