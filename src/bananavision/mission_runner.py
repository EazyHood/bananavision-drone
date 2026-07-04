from __future__ import annotations

import json
from pathlib import Path
from time import sleep, time
from typing import Any

from .io import write_bundle
from .mission import write_mission_outputs
from .models import InferenceConfig, PredictionResult
from .pipeline import IMAGE_EXTENSIONS, make_detector, predict_image
from .runtime import build_run_manifest, monotonic_seconds, utc_now_iso, write_run_manifest

STATE_FILE = "mission_watch_state.json"


def watch_mission(
    watch_dir: str | Path,
    output_dir: str | Path,
    config: InferenceConfig,
    poll_interval: float = 1.0,
    settle_seconds: float = 0.5,
    max_cycles: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    watch_dir = Path(watch_dir)
    if not watch_dir.exists() or not watch_dir.is_dir():
        raise FileNotFoundError(f"Watch directory does not exist: {watch_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now_iso()
    started = monotonic_seconds()
    detector = make_detector(config)
    state = _load_watch_state(output_dir) if resume else {}
    processed: set[Path] = {Path(path) for path in state.get("processed_images_resolved", [])}
    failures: list[dict[str, Any]] = list(state.get("failures", []))
    results: list[PredictionResult] = []
    cycle = 0

    while True:
        cycle += 1
        for image_path in _ready_images(watch_dir, processed, settle_seconds):
            resolved = image_path.resolve()
            try:
                result = predict_image(image_path, config, detector=detector)
                write_bundle(result, output_dir)
                processed.add(resolved)
                results.append(result)
            except Exception as exc:
                processed.add(resolved)
                failures.append({"image": str(image_path), "error": str(exc), "failed_at": utc_now_iso()})
            _write_watch_state(output_dir, processed, results, failures)
            _write_mission_summary(watch_dir, output_dir, results, config, started_at, started, failures)
        if max_cycles is not None and cycle >= max_cycles:
            break
        sleep(poll_interval)

    manifest = _write_mission_summary(watch_dir, output_dir, results, config, started_at, started, failures)
    return manifest


def _ready_images(watch_dir: Path, processed: set[Path], settle_seconds: float) -> list[Path]:
    now = time()
    images: list[Path] = []
    for path in sorted(watch_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS or path.resolve() in processed:
            continue
        age = now - path.stat().st_mtime
        if age >= settle_seconds:
            images.append(path)
    return images


def _write_mission_summary(
    watch_dir: Path,
    output_dir: Path,
    results: list[PredictionResult],
    config: InferenceConfig,
    started_at: str,
    started: float,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    mission = write_mission_outputs(
        results,
        output_dir,
        geo_distance_m=config.mission_geo_dedupe_distance_m,
        pixel_distance_px=config.mission_pixel_dedupe_distance_px,
    )
    manifest = build_run_manifest(
        input_path=watch_dir,
        output_dir=output_dir,
        results=results,
        config=config,
        started_at=started_at,
        elapsed_ms=(monotonic_seconds() - started) * 1000.0,
    )
    manifest["mode"] = "watch"
    manifest["mission"] = mission
    manifest["failures"] = failures
    manifest["failure_count"] = len(failures)
    write_run_manifest(manifest, output_dir)
    return manifest


def _write_watch_state(
    output_dir: Path,
    processed: set[Path],
    results: list[PredictionResult],
    failures: list[dict[str, Any]],
) -> Path:
    state = {
        "updated_at": utc_now_iso(),
        "processed_images_resolved": [str(path) for path in sorted(processed)],
        "processed_count": len(processed),
        "total_detections": sum(result.count for result in results),
        "failure_count": len(failures),
        "failures": failures,
    }
    path = output_dir / STATE_FILE
    _atomic_write_json(path, state)
    return path


def _load_watch_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / STATE_FILE
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if "processed_images_resolved" not in state and "processed_images" in state:
        state["processed_images_resolved"] = [str(Path(value).resolve()) for value in state.get("processed_images", [])]
    return state


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)
