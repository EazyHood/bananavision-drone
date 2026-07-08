from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from PIL import Image

from .detectors.rgb_canopy import RgbCanopyDetector
from .detectors.yolo_ensemble import YoloEnsembleDetector
from .detectors.yolo_seg import YoloSegDetector
from .crown_centers import attach_crown_centers
from .geo import load_geotransform
from .io import assign_ids, attach_geo, write_bundle
from .mission import write_mission_outputs
from .models import Detection, InferenceConfig, PredictionResult
from .postprocess import nms
from .runtime import (
    annotate_result,
    build_run_manifest,
    monotonic_seconds,
    utc_now_iso,
    write_run_manifest,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> InferenceConfig:
    values: dict[str, Any] = {}
    if path:
        values.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    return InferenceConfig.from_mapping(values)


def make_detector(config: InferenceConfig):
    if config.detector == "rgb-canopy":
        return RgbCanopyDetector()
    if config.detector == "yolo-seg":
        if not config.model_path:
            raise ValueError("model_path is required when detector='yolo-seg'")
        return YoloSegDetector(config.model_path)
    if config.detector == "yolo-ensemble":
        return YoloEnsembleDetector(config.ensemble_model_paths or [], config.ensemble_weights)
    raise ValueError(f"Unknown detector: {config.detector}")


def predict_image(image_path: str | Path, config: InferenceConfig, detector=None) -> PredictionResult:
    image_path = Path(image_path)
    detector = detector or make_detector(config)
    started = perf_counter()
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        detections = _predict_image_or_tiles(image, detector, config)
        if config.mark_crown_centers:
            attach_crown_centers(image, detections, config)
        transform = load_geotransform(image_path)
        attach_geo(assign_ids(detections), transform)
        result = PredictionResult(
            image_path=image_path,
            width=image.width,
            height=image.height,
            detections=detections,
            config=config,
        )
        return annotate_result(result, (perf_counter() - started) * 1000.0, config)


def _predict_image_or_tiles(image: Image.Image, detector, config: InferenceConfig) -> list[Detection]:
    if max(image.size) <= config.tile_size:
        return detector.predict(image, config)
    stride = max(1, config.tile_size - config.tile_overlap)
    detections: list[Detection] = []
    width, height = image.size
    for y in _tile_starts(height, config.tile_size, stride):
        for x in _tile_starts(width, config.tile_size, stride):
            box = (x, y, min(width, x + config.tile_size), min(height, y + config.tile_size))
            tile = image.crop(box)
            for detection in detector.predict(tile, config):
                detections.append(_shift_detection(detection, x, y))
    return nms(detections, config.iou_threshold)


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _shift_detection(detection: Detection, offset_x: int, offset_y: int) -> Detection:
    x1, y1, x2, y2 = detection.bbox
    cx, cy = detection.center
    detection.bbox = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
    detection.center = (cx + offset_x, cy + offset_y)
    detection.meta["tile_offset"] = [offset_x, offset_y]
    return detection


def iter_images(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    images = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_path}")
    return images


def predict_path(input_path: str | Path, output_dir: str | Path, config: InferenceConfig) -> list[PredictionResult]:
    started_at = utc_now_iso()
    started = monotonic_seconds()
    detector = make_detector(config)
    results: list[PredictionResult] = []
    for image_path in iter_images(input_path):
        result = predict_image(image_path, config, detector=detector)
        write_bundle(result, output_dir)
        results.append(result)
    mission = write_mission_outputs(
        results,
        output_dir,
        geo_distance_m=config.mission_geo_dedupe_distance_m,
        pixel_distance_px=config.mission_pixel_dedupe_distance_px,
    )
    manifest = build_run_manifest(
        input_path=input_path,
        output_dir=output_dir,
        results=results,
        config=config,
        started_at=started_at,
        elapsed_ms=(monotonic_seconds() - started) * 1000.0,
    )
    manifest["mission"] = mission
    write_run_manifest(manifest, output_dir)
    return results
