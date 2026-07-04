from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evaluation import evaluate_path, write_batch_evaluation_report
from .models import InferenceConfig
from .runtime import utc_now_iso
from .synthetic import generate_scene
from .truth_coverage import audit_truth_coverage, write_truth_coverage_report


def run_cluster_benchmark(
    output_dir: str | Path,
    config: InferenceConfig,
    scenes: int = 3,
    width: int = 320,
    height: int = 240,
    plants_per_scene: int = 12,
    clustered_mats_per_scene: int = 3,
    min_plants_per_mat: int = 3,
    max_plants_per_mat: int = 3,
    cluster_spread_px: float = 24.0,
    seed: int = 21,
    tolerance_px: float = 45.0,
    max_count_error_rate: float = 0.5,
    max_mean_image_count_error_rate: float | None = 0.6,
    min_precision: float = 0.4,
    min_recall: float = 0.4,
    min_f1: float = 0.4,
    min_cluster_recall: float = 0.5,
    min_cluster_full_detection_rate: float = 0.25,
) -> dict[str, Any]:
    if scenes < 1:
        raise ValueError("scenes must be at least 1")
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    truth_dir = output_dir / "truth"
    inference_dir = output_dir / "inference"
    report_path = output_dir / "cluster_benchmark_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for index in range(scenes):
        image = image_dir / f"cluster_scene_{index + 1:03d}.jpg"
        truth = truth_dir / f"cluster_scene_{index + 1:03d}.truth.json"
        generate_scene(
            image,
            truth,
            width=width,
            height=height,
            plant_count=plants_per_scene,
            seed=seed + index,
            clustered_mats=clustered_mats_per_scene,
            min_plants_per_mat=min_plants_per_mat,
            max_plants_per_mat=max_plants_per_mat,
            cluster_spread_px=cluster_spread_px,
        )
        generated.append({"image": str(image), "truth": str(truth), "seed": seed + index})

    min_cluster_count = scenes * clustered_mats_per_scene
    min_cluster_truth_count = min_cluster_count * min_plants_per_mat
    coverage = audit_truth_coverage(
        truth_dir,
        min_truth_count=scenes * plants_per_scene,
        min_cluster_count=min_cluster_count,
        min_cluster_truth_count=min_cluster_truth_count,
        min_cluster_images=scenes if clustered_mats_per_scene > 0 else 0,
    )
    coverage_path = write_truth_coverage_report(coverage, output_dir / "truth_coverage_report.json")

    acceptance = evaluate_path(image_dir, truth_dir, config, tolerance_px=tolerance_px, output_dir=inference_dir)
    thresholds: dict[str, float | int] = {
        "tolerance_px": tolerance_px,
        "max_count_error_rate": max_count_error_rate,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "min_f1": min_f1,
        "min_cluster_count": min_cluster_count,
        "min_cluster_recall": min_cluster_recall,
        "min_cluster_full_detection_rate": min_cluster_full_detection_rate,
    }
    if max_mean_image_count_error_rate is not None:
        thresholds["max_mean_image_count_error_rate"] = max_mean_image_count_error_rate
    acceptance_path = write_batch_evaluation_report(output_dir / "cluster_acceptance_report.json", acceptance, thresholds)
    status = "pass" if coverage["status"] == "pass" and acceptance["passed"] is True else "fail"
    report = {
        "created_at": utc_now_iso(),
        "status": status,
        "output_dir": str(output_dir),
        "config": asdict(config),
        "scenario": {
            "scenes": scenes,
            "width": width,
            "height": height,
            "plants_per_scene": plants_per_scene,
            "clustered_mats_per_scene": clustered_mats_per_scene,
            "min_plants_per_mat": min_plants_per_mat,
            "max_plants_per_mat": max_plants_per_mat,
            "cluster_spread_px": cluster_spread_px,
            "seed": seed,
        },
        "thresholds": thresholds,
        "truth_coverage_report": str(coverage_path),
        "acceptance_report": str(acceptance_path),
        "truth_coverage": {
            "status": coverage["status"],
            "truth_count": coverage["truth_count"],
            "cluster_count": coverage["cluster_count"],
            "cluster_truth_count": coverage["cluster_truth_count"],
            "cluster_image_count": coverage["cluster_image_count"],
        },
        "metrics": acceptance["metrics"],
        "passed": status == "pass",
        "generated": generated,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
