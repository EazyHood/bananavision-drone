from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from .pipeline import iter_images, make_detector, predict_image
from .runtime import runtime_fingerprint, utc_now_iso


def benchmark_images(
    input_path: str | Path,
    config,
    runs: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    images = iter_images(input_path)
    detector = make_detector(config)
    for image_path in images[: max(0, warmup)]:
        predict_image(image_path, config, detector=detector)

    samples: list[dict[str, Any]] = []
    for _ in range(runs):
        for image_path in images:
            started = perf_counter()
            result = predict_image(image_path, config, detector=detector)
            elapsed_ms = (perf_counter() - started) * 1000.0
            samples.append(
                {
                    "image": str(image_path),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "count": result.count,
                    "width": result.width,
                    "height": result.height,
                }
            )

    latencies = [sample["elapsed_ms"] for sample in samples]
    return {
        "created_at": utc_now_iso(),
        "input_path": str(input_path),
        "image_count": len(images),
        "runs": runs,
        "warmup": warmup,
        "runtime": runtime_fingerprint(config),
        "latency_ms": {
            "mean": round(mean(latencies), 3) if latencies else 0.0,
            "median": round(median(latencies), 3) if latencies else 0.0,
            "p95": round(_percentile(latencies, 95), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "samples": samples,
    }


def write_benchmark_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
