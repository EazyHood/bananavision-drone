from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from . import __version__
from .models import InferenceConfig, PredictionResult


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def monotonic_seconds() -> float:
    return perf_counter()


def file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_dict(config: InferenceConfig) -> dict[str, Any]:
    return asdict(config)


def model_fingerprint(config: InferenceConfig) -> dict[str, Any]:
    model_path = Path(config.model_path).resolve() if config.model_path else None
    return {
        "detector": config.detector,
        "model_path": None if model_path is None else str(model_path),
        "model_sha256": file_sha256(model_path),
    }


def runtime_fingerprint(config: InferenceConfig) -> dict[str, Any]:
    cfg = config_dict(config)
    return {
        "bananavision_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": cfg,
        "config_sha256": stable_json_hash(cfg),
        "model": model_fingerprint(config),
    }


def annotate_result(result: PredictionResult, elapsed_ms: float, config: InferenceConfig) -> PredictionResult:
    result.meta.update(
        {
            "elapsed_ms": round(elapsed_ms, 3),
            "runtime": runtime_fingerprint(config),
        }
    )
    return result


def build_run_manifest(
    input_path: str | Path,
    output_dir: str | Path,
    results: list[PredictionResult],
    config: InferenceConfig,
    started_at: str,
    elapsed_ms: float,
) -> dict[str, Any]:
    image_summaries = [
        {
            "image": str(result.image_path),
            "width": result.width,
            "height": result.height,
            "count": result.count,
            "elapsed_ms": result.meta.get("elapsed_ms"),
        }
        for result in results
    ]
    return {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "elapsed_ms": round(elapsed_ms, 3),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "image_count": len(results),
        "total_detections": sum(result.count for result in results),
        "runtime": runtime_fingerprint(config),
        "images": image_summaries,
    }


def write_run_manifest(manifest: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "run_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
