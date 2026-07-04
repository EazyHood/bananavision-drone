from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from .evaluation import evaluate_result
from .models import InferenceConfig
from .pipeline import make_detector, predict_image
from .runtime import utc_now_iso


def tune_config(
    image_path: str | Path,
    truth_path: str | Path,
    base_config: InferenceConfig,
    tolerance_px: float,
    crown_diameters_m: list[float],
    min_center_distance_ratios: list[float],
    center_distance_weights: list[float] | None,
    canopy_fill_ratios: list[float],
    rgb_threshold_quantiles: list[float],
    max_split_instances: list[int] | None = None,
) -> dict[str, Any]:
    detector = make_detector(base_config)
    rows: list[dict[str, Any]] = []
    split_limits = max_split_instances or [base_config.max_split_instances]
    distance_weights = center_distance_weights or [base_config.center_distance_weight]
    for crown_m in crown_diameters_m:
        for distance_ratio in min_center_distance_ratios:
            for center_weight in distance_weights:
                for fill_ratio in canopy_fill_ratios:
                    for rgb_quantile in rgb_threshold_quantiles:
                        for split_limit in split_limits:
                            config = replace(
                                base_config,
                                expected_crown_diameter_m=crown_m,
                                min_center_distance_ratio=distance_ratio,
                                center_distance_weight=center_weight,
                                canopy_fill_ratio=fill_ratio,
                                rgb_threshold_quantile=rgb_quantile,
                                max_split_instances=split_limit,
                                output_overlay=False,
                            )
                            config.validate()
                            result = predict_image(image_path, config, detector=detector)
                            metrics = evaluate_result(result, truth_path, tolerance_px=tolerance_px)
                            rows.append(
                                {
                                    "config": _tuned_fields(config),
                                    "count": result.count,
                                    "metrics": asdict(metrics),
                                }
                            )
    best = min(rows, key=_ranking_key) if rows else None
    best_config = None if best is None else {**asdict(base_config), **best["config"]}
    return {
        "created_at": utc_now_iso(),
        "image": str(image_path),
        "truth": str(truth_path),
        "tolerance_px": tolerance_px,
        "search_space": {
            "expected_crown_diameter_m": crown_diameters_m,
            "min_center_distance_ratio": min_center_distance_ratios,
            "center_distance_weight": distance_weights,
            "canopy_fill_ratio": canopy_fill_ratios,
            "rgb_threshold_quantile": rgb_threshold_quantiles,
            "max_split_instances": split_limits,
        },
        "best": best,
        "best_config": best_config,
        "rows": rows,
    }


def write_tuning_report(report: dict[str, Any], output_json: str | Path) -> Path:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_json


def write_tuned_config(report: dict[str, Any], output_yaml: str | Path) -> Path:
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    config = report.get("best_config")
    if not config:
        raise ValueError("Tuning report does not contain best_config")
    output_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_yaml


def _tuned_fields(config: InferenceConfig) -> dict[str, float | int]:
    return {
        "expected_crown_diameter_m": config.expected_crown_diameter_m,
        "min_center_distance_ratio": config.min_center_distance_ratio,
        "center_distance_weight": config.center_distance_weight,
        "canopy_fill_ratio": config.canopy_fill_ratio,
        "rgb_threshold_quantile": config.rgb_threshold_quantile,
        "max_split_instances": config.max_split_instances,
    }


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    metrics = row["metrics"]
    return (
        float(metrics["count_error_rate"]),
        -float(metrics.get("cluster_recall", 0.0)),
        -float(metrics["f1"]),
        -float(metrics["recall"]),
        abs(int(metrics["count_error"])),
    )
