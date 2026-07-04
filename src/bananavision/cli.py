from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console

from .active_learning import build_review_queue, export_review_crops
from .annotations import convert_coco_to_yolo_seg, convert_labelme_to_yolo_seg
from .benchmark import benchmark_images, write_benchmark_report
from .calibration import calibrate_thresholds, write_calibration_report
from .capture_coverage import CaptureCoverageThresholds, audit_capture_coverage
from .cluster_benchmark import run_cluster_benchmark
from .cluster_review import build_cluster_review
from .dataset import audit_yolo_dataset
from .deploy import write_systemd_units
from .deployment_audit import audit_deployment
from .deployment_smoke import run_deployment_smoke_test, write_deployment_smoke_report
from .domain import (
    DomainProfileSettings,
    DomainShiftThresholds,
    audit_domain_shift,
    build_domain_profile,
)
from .drone_ready import run_drone_ready_check
from .evaluation import (
    acceptance_passed,
    evaluate_image,
    evaluate_path,
    write_batch_evaluation_report,
    write_evaluation_report,
)
from .evidence_manifest import build_evidence_manifest
from .flight_profile import FlightEnvelope, FlightProfile, audit_flight_log, audit_flight_profile
from .geo_accuracy import audit_geo_accuracy
from .holdout import lock_holdout, verify_holdout_lock
from .inventory import diff_inventories, update_inventory
from .io import write_bundle
from .metrics import BatchMetrics
from .mission_audit import audit_mission_delivery
from .mission_process import process_mission
from .mission_quality import MissionQualityThresholds, audit_mission_images
from .mission_runner import watch_mission
from .model_card import build_model_card
from .pipeline import load_config, predict_path
from .prediction_quality import PredictionQualityThresholds, audit_prediction_outputs
from .publication import audit_publication
from .quality import audit_dataset_quality, write_quality_report
from .readiness import run_preflight, write_preflight_report
from .registry import promote_model, register_model
from .release_audit import audit_release
from .release_package import (
    PackageArtifact,
    build_release_package,
    parse_artifact_specs,
    verify_release_package,
)
from .reporting import build_field_report
from .splitting import split_yolo_dataset
from .stratified_acceptance import build_stratified_acceptance_report
from .stratified_truth_coverage import build_stratified_truth_coverage_report
from .synthetic import generate_scene
from .tiling import tile_yolo_dataset
from .train import export_yolo, train_yolo, validate_yolo
from .truth_coverage import audit_truth_coverage, write_truth_coverage_report
from .truth_quality import TruthQualityThresholds, audit_truth_quality
from .tuning import tune_config, write_tuned_config, write_tuning_report
from .validation_plan import build_validation_plan, write_validation_plan

app = typer.Typer(help="BananaVision Drone: banana plant detection and counting for UAV imagery.")
annotations_app = typer.Typer(help="Annotation conversion utilities.")
app.add_typer(annotations_app, name="annotations")
console = Console()


@app.command()
def infer(
    input_path: Path = typer.Argument(..., help="Image file or directory."),
    output_dir: Path = typer.Option(Path("runs/infer"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    gsd_cm: float | None = typer.Option(None, "--gsd-cm"),
    expected_crown_diameter_m: float | None = typer.Option(None, "--crown-m"),
    min_component_area_px: int | None = typer.Option(None, "--min-component-area-px"),
    max_split_instances: int | None = typer.Option(None, "--max-split-instances"),
    center_distance_weight: float | None = typer.Option(None, "--center-distance-weight"),
    rgb_threshold_quantile: float | None = typer.Option(None, "--rgb-threshold-quantile"),
) -> None:
    cfg = load_config(
        config,
        {
            "detector": detector,
            "model_path": model_path,
            "gsd_cm": gsd_cm,
            "expected_crown_diameter_m": expected_crown_diameter_m,
            "min_component_area_px": min_component_area_px,
            "max_split_instances": max_split_instances,
            "center_distance_weight": center_distance_weight,
            "rgb_threshold_quantile": rgb_threshold_quantile,
        },
    )
    results = predict_path(input_path, output_dir, cfg)
    total = sum(result.count for result in results)
    console.print(f"[green]Processed {len(results)} image(s). Detected {total} banana plant candidate(s).[/green]")
    console.print(f"Outputs: {output_dir}")


@app.command("inventory-update")
def inventory_update(
    detections_geojson: Path = typer.Argument(..., help="mission.detections.geojson or per-image detections GeoJSON."),
    inventory_dir: Path = typer.Argument(..., help="Persistent inventory output directory."),
    distance_threshold: float = typer.Option(1.2, "--distance-threshold"),
    id_prefix: str = typer.Option("banana-plant", "--id-prefix"),
    observed_at: str | None = typer.Option(None, "--observed-at"),
) -> None:
    summary = update_inventory(
        detections_geojson,
        inventory_dir,
        distance_threshold=distance_threshold,
        id_prefix=id_prefix,
        observed_at=observed_at,
    )
    console.print(
        f"[green]Inventory updated:[/green] {summary['plant_count']} plants "
        f"({summary['created']} created, {summary['updated']} updated)"
    )
    console.print(f"JSON: {summary['inventory_json']}")
    console.print(f"Snapshot: {summary['inventory_snapshot']}")
    console.print(f"GeoJSON: {summary['inventory_geojson']}")


@app.command("inventory-diff")
def inventory_diff(
    before_inventory: Path = typer.Argument(..., help="Earlier inventory.json or snapshot."),
    after_inventory: Path = typer.Argument(..., help="Later inventory.json or snapshot."),
    output_dir: Path = typer.Option(Path("runs/inventory_diff"), "--output", "-o"),
) -> None:
    report = diff_inventories(before_inventory, after_inventory, output_dir)
    console.print(
        f"[green]Inventory diff written:[/green] {report['new_count']} new, "
        f"{report['missing_count']} missing, {report['persistent_count']} persistent"
    )
    console.print(f"Report: {output_dir / 'inventory_diff.json'}")


@app.command()
def preflight(
    input_path: Path | None = typer.Option(None, "--input"),
    output_json: Path = typer.Option(Path("runs/preflight/preflight_report.json"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    min_free_gb: float = typer.Option(2.0, "--min-free-gb"),
) -> None:
    cfg = load_config(config, {"detector": detector, "model_path": model_path})
    report = run_preflight(cfg, input_path=input_path, output_dir=output_json.parent, min_free_gb=min_free_gb)
    path = write_preflight_report(report, output_json)
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(f"[{color}]Preflight {str(report['status']).upper()}[/{color}]")
    for check in report["checks"]:
        console.print(f"- {check['name']}: {check['status']} - {check['detail']}")
    console.print(f"Report: {path}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("deployment-smoke-test")
def deployment_smoke_test(
    input_path: Path = typer.Argument(..., help="Known-good smoke image or folder available on the drone computer."),
    output_json: Path = typer.Option(Path("runs/deployment_smoke/deployment_smoke_report.json"), "--output", "-o"),
    artifacts_dir: Path | None = typer.Option(None, "--artifacts-dir"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    gsd_cm: float | None = typer.Option(None, "--gsd-cm"),
    expected_crown_diameter_m: float | None = typer.Option(None, "--crown-m"),
    min_component_area_px: int | None = typer.Option(None, "--min-component-area-px"),
    max_split_instances: int | None = typer.Option(None, "--max-split-instances"),
    center_distance_weight: float | None = typer.Option(None, "--center-distance-weight"),
    rgb_threshold_quantile: float | None = typer.Option(None, "--rgb-threshold-quantile"),
    min_images: int = typer.Option(1, "--min-images"),
    min_detections: int = typer.Option(1, "--min-detections"),
    max_image_latency_ms: float | None = typer.Option(None, "--max-image-latency-ms"),
    min_free_gb: float = typer.Option(1.0, "--min-free-gb"),
) -> None:
    cfg = load_config(
        config,
        {
            "detector": detector,
            "model_path": model_path,
            "gsd_cm": gsd_cm,
            "expected_crown_diameter_m": expected_crown_diameter_m,
            "min_component_area_px": min_component_area_px,
            "max_split_instances": max_split_instances,
            "center_distance_weight": center_distance_weight,
            "rgb_threshold_quantile": rgb_threshold_quantile,
        },
    )
    artifacts = artifacts_dir or (output_json.parent / "artifacts")
    report = run_deployment_smoke_test(
        input_path,
        artifacts,
        cfg,
        min_images=min_images,
        min_detections=min_detections,
        max_image_latency_ms=max_image_latency_ms,
        min_free_gb=min_free_gb,
    )
    path = write_deployment_smoke_report(report, output_json)
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Deployment smoke {str(report['status']).upper()}:[/{color}] "
        f"{report['image_count']} image(s), {report['total_detections']} detection(s)"
    )
    console.print(f"Artifacts: {artifacts}")
    console.print(f"Report: {path}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("drone-ready")
def drone_ready(
    release_package: Path = typer.Argument(..., help="Installed release package folder, manifest JSON, or ZIP."),
    deployment_manifest: Path = typer.Argument(..., help="Deployment manifest installed with the package."),
    smoke_image: Path = typer.Argument(..., help="Known-good smoke image available on the drone computer."),
    output_dir: Path = typer.Option(Path("runs/drone_ready"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    gsd_cm: float | None = typer.Option(None, "--gsd-cm"),
    expected_crown_diameter_m: float | None = typer.Option(None, "--crown-m"),
    min_component_area_px: int | None = typer.Option(None, "--min-component-area-px"),
    max_split_instances: int | None = typer.Option(None, "--max-split-instances"),
    center_distance_weight: float | None = typer.Option(None, "--center-distance-weight"),
    rgb_threshold_quantile: float | None = typer.Option(None, "--rgb-threshold-quantile"),
    min_detections: int = typer.Option(1, "--min-detections"),
    max_image_latency_ms: float | None = typer.Option(None, "--max-image-latency-ms"),
    min_free_gb: float = typer.Option(1.0, "--min-free-gb"),
    allow_warn_preflight: bool = typer.Option(False, "--allow-warn-preflight"),
    allow_exploratory_package: bool = typer.Option(False, "--allow-exploratory-package"),
    no_require_deployment_artifacts: bool = typer.Option(False, "--no-require-deployment-artifacts"),
) -> None:
    cfg = load_config(
        config,
        {
            "detector": detector,
            "model_path": model_path,
            "gsd_cm": gsd_cm,
            "expected_crown_diameter_m": expected_crown_diameter_m,
            "min_component_area_px": min_component_area_px,
            "max_split_instances": max_split_instances,
            "center_distance_weight": center_distance_weight,
            "rgb_threshold_quantile": rgb_threshold_quantile,
        },
    )
    report = run_drone_ready_check(
        output_dir,
        release_package=release_package,
        deployment_manifest=deployment_manifest,
        smoke_image=smoke_image,
        config=cfg,
        config_path=config,
        min_detections=min_detections,
        max_image_latency_ms=max_image_latency_ms,
        min_free_gb=min_free_gb,
        allow_warn_preflight=allow_warn_preflight,
        allow_exploratory_package=allow_exploratory_package,
        require_deployment_artifacts=not no_require_deployment_artifacts,
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(f"[{color}]Drone ready {str(report['status']).upper()}[/{color}]")
    for label, path in report["artifacts"].items():
        console.print(f"{label}: {path}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("flight-check")
def flight_check(
    output_json: Path = typer.Option(Path("runs/flight_check/flight_check_report.json"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    gsd_cm: float | None = typer.Option(None, "--gsd-cm"),
    altitude_m: float | None = typer.Option(None, "--altitude-m"),
    sensor_width_mm: float | None = typer.Option(None, "--sensor-width-mm"),
    focal_length_mm: float | None = typer.Option(None, "--focal-length-mm"),
    image_width_px: int | None = typer.Option(None, "--image-width-px"),
    front_overlap: float | None = typer.Option(None, "--front-overlap"),
    side_overlap: float | None = typer.Option(None, "--side-overlap"),
    speed_mps: float | None = typer.Option(None, "--speed-mps"),
    exposure_ms: float | None = typer.Option(None, "--exposure-ms"),
    max_gsd_drift_ratio: float = typer.Option(0.20, "--max-gsd-drift-ratio"),
    min_front_overlap: float = typer.Option(70.0, "--min-front-overlap"),
    min_side_overlap: float = typer.Option(70.0, "--min-side-overlap"),
    max_motion_blur_px: float = typer.Option(1.5, "--max-motion-blur-px"),
) -> None:
    cfg = load_config(config)
    report = audit_flight_profile(
        FlightProfile(
            gsd_cm=gsd_cm,
            altitude_m=altitude_m,
            sensor_width_mm=sensor_width_mm,
            focal_length_mm=focal_length_mm,
            image_width_px=image_width_px,
            front_overlap=front_overlap,
            side_overlap=side_overlap,
            speed_mps=speed_mps,
            exposure_ms=exposure_ms,
        ),
        FlightEnvelope.from_config(
            cfg,
            max_gsd_drift_ratio=max_gsd_drift_ratio,
            min_front_overlap=min_front_overlap,
            min_side_overlap=min_side_overlap,
            max_motion_blur_px=max_motion_blur_px,
        ),
        output_json,
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(f"[{color}]Flight check {str(report['status']).upper()}[/{color}]")
    for check in report["checks"]:
        console.print(f"- {check['name']}: {check['status']} - {check['detail']}")
    console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("flight-log-audit")
def flight_log_audit(
    log_csv: Path = typer.Argument(..., help="CSV telemetry/capture log with GSD, overlap, speed, exposure, or camera geometry columns."),
    output_json: Path = typer.Option(Path("runs/flight_log/flight_log_audit.json"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    max_gsd_drift_ratio: float = typer.Option(0.20, "--max-gsd-drift-ratio"),
    min_front_overlap: float = typer.Option(70.0, "--min-front-overlap"),
    min_side_overlap: float = typer.Option(70.0, "--min-side-overlap"),
    max_motion_blur_px: float = typer.Option(1.5, "--max-motion-blur-px"),
) -> None:
    cfg = load_config(config)
    report = audit_flight_log(
        log_csv,
        FlightEnvelope.from_config(
            cfg,
            max_gsd_drift_ratio=max_gsd_drift_ratio,
            min_front_overlap=min_front_overlap,
            min_side_overlap=min_side_overlap,
            max_motion_blur_px=max_motion_blur_px,
        ),
        output_json,
    )
    summary = report["summary"]
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Flight log audit {str(report['status']).upper()}:[/{color}] "
        f"{summary['pass_count']} pass, {summary['warn_count']} warn, {summary['fail_count']} fail"
    )
    console.print(f"Report: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("capture-coverage")
def capture_coverage(
    capture_log_csv: Path = typer.Argument(..., help="CSV capture log with image plus lat/lon or x/y columns."),
    output_json: Path = typer.Option(Path("runs/capture_coverage/capture_coverage_report.json"), "--output", "-o"),
    image_dir: Path | None = typer.Option(None, "--images"),
    min_images: int = typer.Option(1, "--min-images"),
    max_position_gap_m: float | None = typer.Option(35.0, "--max-position-gap-m"),
    max_time_gap_s: float | None = typer.Option(None, "--max-time-gap-s"),
    min_position_delta_m: float = typer.Option(0.25, "--min-position-delta-m"),
    require_positions: bool = typer.Option(True, "--require-positions/--no-require-positions"),
    require_timestamps: bool = typer.Option(False, "--require-timestamps/--no-require-timestamps"),
    require_image_files: bool = typer.Option(False, "--require-image-files/--no-require-image-files"),
) -> None:
    thresholds = CaptureCoverageThresholds(
        min_images=min_images,
        max_position_gap_m=max_position_gap_m,
        max_time_gap_s=max_time_gap_s,
        min_position_delta_m=min_position_delta_m,
        require_positions=require_positions,
        require_timestamps=require_timestamps,
        require_image_files=require_image_files,
    )
    try:
        report = audit_capture_coverage(capture_log_csv, output_json, image_dir=image_dir, thresholds=thresholds)
    except ValueError as exc:
        console.print(f"[red]Capture coverage failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    summary = report["summary"]
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Capture coverage {str(report['status']).upper()}:[/{color}] "
        f"rows={summary['row_count']}, missing_images={summary['missing_image_count']}, "
        f"max_step_m={summary['max_step_distance_m']:.3f}, max_gap_s={summary['max_step_time_s']:.3f}"
    )
    console.print(f"Report: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("domain-profile")
def domain_profile(
    input_path: Path = typer.Argument(..., help="Validated holdout image file or directory."),
    output_json: Path = typer.Option(Path("runs/domain/domain_profile.json"), "--output", "-o"),
    bins: int = typer.Option(16, "--bins"),
    max_side: int = typer.Option(512, "--max-side"),
) -> None:
    profile = build_domain_profile(
        input_path,
        output_json,
        DomainProfileSettings(bins=bins, max_side=max_side),
    )
    console.print(
        f"[green]Domain profile written:[/green] {profile['image_count']} reference image(s), "
        f"bins={profile['settings']['bins']}"
    )
    console.print(f"Report: {output_json}")


@app.command("domain-check")
def domain_check(
    input_path: Path = typer.Argument(..., help="Mission image file or directory to compare."),
    profile_json: Path = typer.Argument(..., help="domain_profile.json built from validated holdout imagery."),
    output_json: Path = typer.Option(Path("runs/domain/domain_check_report.json"), "--output", "-o"),
    max_histogram_distance: float = typer.Option(0.20, "--max-histogram-distance"),
    max_feature_z: float = typer.Option(4.0, "--max-feature-z"),
    max_outlier_fraction: float = typer.Option(0.0, "--max-outlier-fraction"),
    min_reference_images: int = typer.Option(1, "--min-reference-images"),
) -> None:
    report = audit_domain_shift(
        input_path,
        profile_json,
        output_json,
        DomainShiftThresholds(
            max_histogram_distance=max_histogram_distance,
            max_feature_z=max_feature_z,
            max_outlier_fraction=max_outlier_fraction,
            min_reference_images=min_reference_images,
        ),
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Domain check {str(report['status']).upper()}:[/{color}] "
        f"{report['outlier_count']} of {report['image_count']} image(s) out-of-domain "
        f"({report['outlier_fraction']:.2%})"
    )
    console.print(f"Report: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("deploy-systemd")
def deploy_systemd(
    output_dir: Path = typer.Option(Path("edge/systemd"), "--output", "-o"),
    install_dir: str = typer.Option("/opt/bananavision-drone", "--install-dir"),
    user: str = typer.Option("bananavision", "--user"),
    bananavision_bin: str = typer.Option("/opt/bananavision-drone/.venv/bin/bananavision", "--bin"),
    config: str = typer.Option("/opt/bananavision-drone/configs/banana_uav.yaml", "--config", "-c"),
    detector: str = typer.Option("yolo-seg", "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    watch_dir: str = typer.Option("/data/mission/incoming", "--watch-dir"),
    mission_output_dir: str = typer.Option("/data/mission/output", "--mission-output"),
    api_host: str = typer.Option("0.0.0.0", "--api-host"),
    api_port: int = typer.Option(8080, "--api-port"),
    environment_file: str | None = typer.Option(None, "--environment-file"),
) -> None:
    artifacts = write_systemd_units(
        output_dir=output_dir,
        install_dir=install_dir,
        user=user,
        bananavision_bin=bananavision_bin,
        config_path=config,
        detector=detector,
        model_path=model_path,
        watch_dir=watch_dir,
        mission_output_dir=mission_output_dir,
        api_host=api_host,
        api_port=api_port,
        environment_file=environment_file,
    )
    console.print(f"[green]systemd artifacts written:[/green] {artifacts.output_dir}")
    for file in artifacts.files:
        console.print(f"- {file}")


@app.command("mission-quality")
def mission_quality(
    input_path: Path = typer.Argument(..., help="Mission image file or directory."),
    output_json: Path = typer.Option(Path("runs/mission_quality/mission_quality_report.json"), "--output", "-o"),
    min_width: int = typer.Option(1024, "--min-width"),
    min_height: int = typer.Option(768, "--min-height"),
    min_focus_score: float = typer.Option(12.0, "--min-focus-score"),
    min_mean_luma: float = typer.Option(25.0, "--min-mean-luma"),
    max_mean_luma: float = typer.Option(235.0, "--max-mean-luma"),
    max_dark_fraction: float = typer.Option(0.35, "--max-dark-fraction"),
    max_bright_fraction: float = typer.Option(0.35, "--max-bright-fraction"),
    require_georef: bool = typer.Option(False, "--require-georef"),
) -> None:
    thresholds = MissionQualityThresholds(
        min_width=min_width,
        min_height=min_height,
        min_focus_score=min_focus_score,
        min_mean_luma=min_mean_luma,
        max_mean_luma=max_mean_luma,
        max_dark_fraction=max_dark_fraction,
        max_bright_fraction=max_bright_fraction,
        require_georef=require_georef,
    )
    report = audit_mission_images(input_path, output_json, thresholds)
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Mission quality {str(report['status']).upper()}:[/{color}] "
        f"{report['pass_count']} pass, {report['warn_count']} warn, {report['fail_count']} fail"
    )
    console.print(f"Report: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("prediction-quality")
def prediction_quality(
    predictions_dir: Path = typer.Argument(..., help="Directory containing *.detections.json files."),
    output_json: Path = typer.Option(Path("runs/prediction_quality/prediction_quality_report.json"), "--output", "-o"),
    low_confidence: float = typer.Option(0.45, "--low-confidence"),
    high_split_count: int = typer.Option(3, "--high-split-count"),
    edge_margin_px: float = typer.Option(8.0, "--edge-margin-px"),
    min_center_distance_px: float = typer.Option(24.0, "--min-center-distance-px"),
    duplicate_iou: float = typer.Option(0.65, "--duplicate-iou"),
    max_review_fraction: float = typer.Option(0.2, "--max-review-fraction"),
    fail_on_zero_detections: bool = typer.Option(False, "--fail-on-zero-detections"),
) -> None:
    thresholds = PredictionQualityThresholds(
        low_confidence=low_confidence,
        high_split_count=high_split_count,
        edge_margin_px=edge_margin_px,
        min_center_distance_px=min_center_distance_px,
        duplicate_iou=duplicate_iou,
        max_review_fraction=max_review_fraction,
        fail_on_zero_detections=fail_on_zero_detections,
    )
    report = audit_prediction_outputs(predictions_dir, output_json, thresholds)
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Prediction quality {str(report['status']).upper()}:[/{color}] "
        f"{report['review_detection_count']} of {report['detection_count']} detections require review "
        f"({report['review_fraction']:.2%})"
    )
    console.print(f"Report: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("geo-accuracy")
def geo_accuracy(
    predictions_geojson: Path = typer.Argument(..., help="Predicted detections GeoJSON."),
    truth_geojson: Path = typer.Argument(..., help="Field truth GeoJSON with Point features."),
    output_json: Path = typer.Option(Path("runs/geo_accuracy/geo_accuracy_report.json"), "--output", "-o"),
    tolerance_m: float = typer.Option(1.0, "--tolerance-m"),
    max_rmse_m: float = typer.Option(1.0, "--max-rmse-m"),
    max_p95_m: float | None = typer.Option(None, "--max-p95-m"),
    min_recall: float | None = typer.Option(0.99, "--min-recall"),
) -> None:
    report = audit_geo_accuracy(
        predictions_geojson,
        truth_geojson,
        output_json,
        tolerance_m=tolerance_m,
        max_rmse_m=max_rmse_m,
        max_p95_m=max_p95_m,
        min_recall=min_recall,
    )
    metrics = report["metrics"]
    color = "green" if report["status"] == "pass" else "red"
    console.print(
        f"[{color}]Geo accuracy {str(report['status']).upper()}:[/{color}] "
        f"matched={metrics['matched_count']}/{metrics['truth_count']}, "
        f"rmse_m={metrics['rmse_m']:.3f}, p95_m={metrics['p95_m']:.3f}"
    )
    console.print(f"Report: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("mission-process")
def mission_process(
    input_path: Path = typer.Argument(..., help="Mission image file or directory."),
    output_dir: Path = typer.Option(Path("runs/mission_process"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    gsd_cm: float | None = typer.Option(None, "--gsd-cm"),
    expected_crown_diameter_m: float | None = typer.Option(None, "--crown-m"),
    require_georef: bool = typer.Option(False, "--require-georef"),
    min_width: int = typer.Option(1024, "--min-width"),
    min_height: int = typer.Option(768, "--min-height"),
    min_focus_score: float = typer.Option(12.0, "--min-focus-score"),
    max_review_fraction: float = typer.Option(0.2, "--max-review-fraction"),
    low_confidence: float = typer.Option(0.45, "--low-confidence"),
    high_split_count: int = typer.Option(3, "--high-split-count"),
    inventory_dir: Path | None = typer.Option(None, "--inventory-dir"),
    inventory_distance_threshold: float = typer.Option(1.2, "--inventory-distance-threshold"),
    id_prefix: str = typer.Option("banana-plant", "--id-prefix"),
    observed_at: str | None = typer.Option(None, "--observed-at"),
    title: str = typer.Option("BananaVision Mission Report", "--title"),
    no_fail: bool = typer.Option(False, "--no-fail"),
) -> None:
    cfg = load_config(
        config,
        {
            "detector": detector,
            "model_path": model_path,
            "gsd_cm": gsd_cm,
            "expected_crown_diameter_m": expected_crown_diameter_m,
        },
    )
    manifest = process_mission(
        input_path,
        output_dir,
        cfg,
        mission_quality_thresholds=MissionQualityThresholds(
            min_width=min_width,
            min_height=min_height,
            min_focus_score=min_focus_score,
            require_georef=require_georef,
        ),
        prediction_quality_thresholds=PredictionQualityThresholds(
            low_confidence=low_confidence,
            high_split_count=high_split_count,
            max_review_fraction=max_review_fraction,
        ),
        inventory_dir=inventory_dir,
        inventory_distance_threshold=inventory_distance_threshold,
        id_prefix=id_prefix,
        observed_at=observed_at,
        report_title=title,
    )
    color = "green" if manifest["status"] == "pass" else "yellow"
    if manifest["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Mission process {str(manifest['status']).upper()}:[/{color}] "
        f"{manifest['image_count']} image(s), {manifest['total_detections']} detection(s)"
    )
    console.print(f"Manifest: {manifest['artifacts']['mission_process_manifest']}")
    console.print(f"Field report: {manifest['artifacts']['field_report']}")
    if manifest["status"] == "fail" and not no_fail:
        raise typer.Exit(code=2)


@app.command("mission-watch")
def mission_watch(
    watch_dir: Path = typer.Argument(..., help="Directory receiving drone images."),
    output_dir: Path = typer.Option(Path("runs/mission_watch"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    poll_interval: float = typer.Option(1.0, "--poll-interval"),
    settle_seconds: float = typer.Option(0.5, "--settle-seconds"),
    once: bool = typer.Option(False, "--once"),
    no_resume: bool = typer.Option(False, "--no-resume"),
) -> None:
    cfg = load_config(config, {"detector": detector, "model_path": model_path})
    manifest = watch_mission(
        watch_dir,
        output_dir,
        cfg,
        poll_interval=poll_interval,
        settle_seconds=settle_seconds,
        max_cycles=1 if once else None,
        resume=not no_resume,
    )
    console.print(
        f"[green]Mission watch processed {manifest['image_count']} image(s), "
        f"{manifest['total_detections']} detection(s).[/green]"
    )
    console.print(f"Outputs: {output_dir}")


@app.command()
def acceptance(
    image: Path = typer.Argument(..., help="Image to validate."),
    truth: Path = typer.Argument(..., help="Truth JSON with plant centers."),
    output_dir: Path = typer.Option(Path("runs/acceptance"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    tolerance_px: float = typer.Option(24.0, "--tolerance-px"),
    max_count_error_rate: float = typer.Option(0.01, "--max-count-error-rate"),
    min_precision: float = typer.Option(0.99, "--min-precision"),
    min_recall: float = typer.Option(0.99, "--min-recall"),
    min_f1: float = typer.Option(0.99, "--min-f1"),
    min_cluster_recall: float | None = typer.Option(None, "--min-cluster-recall"),
    min_cluster_full_detection_rate: float | None = typer.Option(None, "--min-cluster-full-detection-rate"),
    min_cluster_count: int | None = typer.Option(None, "--min-cluster-count"),
) -> None:
    cfg = load_config(config, {"detector": detector, "model_path": model_path})
    result, metrics = evaluate_image(image, truth, cfg, tolerance_px=tolerance_px)
    write_bundle(result, output_dir)
    thresholds = {
        "tolerance_px": tolerance_px,
        "max_count_error_rate": max_count_error_rate,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "min_f1": min_f1,
    }
    if min_cluster_recall is not None:
        thresholds["min_cluster_recall"] = min_cluster_recall
    if min_cluster_full_detection_rate is not None:
        thresholds["min_cluster_full_detection_rate"] = min_cluster_full_detection_rate
    if min_cluster_count is not None:
        thresholds["min_cluster_count"] = min_cluster_count
    report_path = write_evaluation_report(output_dir / "acceptance_report.json", result, metrics, thresholds)
    passed = acceptance_passed(
        metrics,
        max_count_error_rate,
        min_precision,
        min_recall,
        min_f1,
        min_cluster_recall=min_cluster_recall,
        min_cluster_full_detection_rate=min_cluster_full_detection_rate,
        min_cluster_count=min_cluster_count,
    )
    color = "green" if passed else "red"
    console.print(
        f"[{color}]Acceptance {'PASSED' if passed else 'FAILED'}: "
        f"count_error_rate={metrics.count_error_rate:.4f}, "
        f"precision={metrics.precision:.4f}, recall={metrics.recall:.4f}, f1={metrics.f1:.4f}, "
        f"clusters={metrics.cluster_count}, cluster_recall={metrics.cluster_recall:.4f}, "
        f"cluster_full_rate={metrics.fully_detected_cluster_rate:.4f}[/{color}]"
    )
    console.print(f"Report: {report_path}")
    if not passed:
        raise typer.Exit(code=2)


@app.command("acceptance-batch")
def acceptance_batch(
    input_path: Path = typer.Argument(..., help="Image file or directory to validate."),
    truth_path: Path = typer.Argument(..., help="Truth manifest JSON or directory of per-image truth JSON files."),
    output_dir: Path = typer.Option(Path("runs/acceptance_batch"), "--output", "-o"),
    holdout_lock: Path | None = typer.Option(None, "--holdout-lock"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    tolerance_px: float = typer.Option(24.0, "--tolerance-px"),
    max_count_error_rate: float = typer.Option(0.01, "--max-count-error-rate"),
    max_mean_image_count_error_rate: float | None = typer.Option(None, "--max-mean-image-count-error-rate"),
    min_precision: float = typer.Option(0.99, "--min-precision"),
    min_recall: float = typer.Option(0.99, "--min-recall"),
    min_f1: float = typer.Option(0.99, "--min-f1"),
    confidence_level: float = typer.Option(0.95, "--confidence-level"),
    min_truth_count: int | None = typer.Option(None, "--min-truth-count"),
    min_precision_ci_lower: float | None = typer.Option(None, "--min-precision-ci-lower"),
    min_recall_ci_lower: float | None = typer.Option(None, "--min-recall-ci-lower"),
    max_mean_image_count_error_rate_ci_upper: float | None = typer.Option(
        None,
        "--max-mean-image-count-error-rate-ci-upper",
    ),
    min_cluster_recall: float | None = typer.Option(None, "--min-cluster-recall"),
    min_cluster_full_detection_rate: float | None = typer.Option(None, "--min-cluster-full-detection-rate"),
    min_cluster_count: int | None = typer.Option(None, "--min-cluster-count"),
) -> None:
    holdout_report = None
    if holdout_lock is not None:
        holdout_report = verify_holdout_lock(
            holdout_lock,
            output_dir / "holdout_verify.json",
            expected_image_path=input_path,
            expected_truth_path=truth_path,
        )
        if holdout_report["status"] != "pass":
            console.print(
                f"[red]Holdout verification FAILED:[/red] {holdout_report['issue_count']} issue(s). "
                f"Report: {output_dir / 'holdout_verify.json'}"
            )
            raise typer.Exit(code=2)
    cfg = load_config(config, {"detector": detector, "model_path": model_path})
    report = evaluate_path(input_path, truth_path, cfg, tolerance_px=tolerance_px, output_dir=output_dir)
    if holdout_report is not None:
        report["holdout_verification"] = holdout_report
    thresholds = {
        "tolerance_px": tolerance_px,
        "max_count_error_rate": max_count_error_rate,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "min_f1": min_f1,
        "confidence_level": confidence_level,
    }
    if holdout_lock is not None:
        thresholds["holdout_lock"] = str(holdout_lock)
    if max_mean_image_count_error_rate is not None:
        thresholds["max_mean_image_count_error_rate"] = max_mean_image_count_error_rate
    if min_truth_count is not None:
        thresholds["min_truth_count"] = min_truth_count
    if min_precision_ci_lower is not None:
        thresholds["min_precision_ci_lower"] = min_precision_ci_lower
    if min_recall_ci_lower is not None:
        thresholds["min_recall_ci_lower"] = min_recall_ci_lower
    if max_mean_image_count_error_rate_ci_upper is not None:
        thresholds["max_mean_image_count_error_rate_ci_upper"] = max_mean_image_count_error_rate_ci_upper
    if min_cluster_recall is not None:
        thresholds["min_cluster_recall"] = min_cluster_recall
    if min_cluster_full_detection_rate is not None:
        thresholds["min_cluster_full_detection_rate"] = min_cluster_full_detection_rate
    if min_cluster_count is not None:
        thresholds["min_cluster_count"] = min_cluster_count
    report_path = write_batch_evaluation_report(output_dir / "acceptance_batch_report.json", report, thresholds)
    metrics = BatchMetrics(**report["metrics"])
    passed = bool(report["passed"])
    statistics = report.get("statistics", {})
    precision_ci = statistics.get("precision_wilson_ci", {})
    recall_ci = statistics.get("recall_wilson_ci", {})
    color = "green" if passed else "red"
    console.print(
        f"[{color}]Batch acceptance {'PASSED' if passed else 'FAILED'}: "
        f"images={metrics.images}, count_error_rate={metrics.count_error_rate:.4f}, "
        f"mean_image_count_error_rate={metrics.mean_abs_image_count_error_rate:.4f}, "
        f"precision={metrics.precision:.4f}, recall={metrics.recall:.4f}, f1={metrics.f1:.4f}, "
        f"clusters={metrics.cluster_count}, cluster_recall={metrics.cluster_recall:.4f}, "
        f"cluster_full_rate={metrics.fully_detected_cluster_rate:.4f}, "
        f"precision_ci_lower={float(precision_ci.get('lower', 0.0)):.4f}, "
        f"recall_ci_lower={float(recall_ci.get('lower', 0.0)):.4f}[/{color}]"
    )
    console.print(f"Report: {report_path}")
    if not passed:
        raise typer.Exit(code=2)


@app.command("stratified-acceptance")
def stratified_acceptance(
    acceptance_report: Path = typer.Argument(..., help="acceptance_batch_report.json with per-image metrics."),
    metadata_csv: Path = typer.Argument(..., help="CSV with image plus strata columns such as farm,date,gsd_band."),
    output_json: Path = typer.Option(Path("runs/stratified_acceptance/stratified_acceptance_report.json"), "--output", "-o"),
    strata: list[str] = typer.Option(["farm", "flight_date", "gsd_band", "cultivar"], "--strata"),
    max_count_error_rate: float = typer.Option(0.01, "--max-count-error-rate"),
    max_mean_image_count_error_rate: float | None = typer.Option(None, "--max-mean-image-count-error-rate"),
    min_precision: float = typer.Option(0.99, "--min-precision"),
    min_recall: float = typer.Option(0.99, "--min-recall"),
    min_f1: float = typer.Option(0.99, "--min-f1"),
    confidence_level: float = typer.Option(0.95, "--confidence-level"),
    min_truth_count: int | None = typer.Option(None, "--min-truth-count"),
    min_precision_ci_lower: float | None = typer.Option(None, "--min-precision-ci-lower"),
    min_recall_ci_lower: float | None = typer.Option(None, "--min-recall-ci-lower"),
    max_mean_image_count_error_rate_ci_upper: float | None = typer.Option(
        None,
        "--max-mean-image-count-error-rate-ci-upper",
    ),
    min_cluster_recall: float | None = typer.Option(None, "--min-cluster-recall"),
    min_cluster_full_detection_rate: float | None = typer.Option(None, "--min-cluster-full-detection-rate"),
    min_cluster_count: int | None = typer.Option(None, "--min-cluster-count"),
) -> None:
    report = build_stratified_acceptance_report(
        acceptance_report,
        metadata_csv,
        output_json,
        strata_keys=strata,
        max_count_error_rate=max_count_error_rate,
        max_mean_image_count_error_rate=max_mean_image_count_error_rate,
        min_precision=min_precision,
        min_recall=min_recall,
        min_f1=min_f1,
        confidence_level=confidence_level,
        min_truth_count=min_truth_count,
        min_precision_ci_lower=min_precision_ci_lower,
        min_recall_ci_lower=min_recall_ci_lower,
        max_mean_image_count_error_rate_ci_upper=max_mean_image_count_error_rate_ci_upper,
        min_cluster_recall=min_cluster_recall,
        min_cluster_full_detection_rate=min_cluster_full_detection_rate,
        min_cluster_count=min_cluster_count,
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Stratified acceptance {str(report['status']).upper()}:[/{color}] "
        f"{report['failed_stratum_count']}/{report['stratum_count']} stratum(s) failed"
    )
    console.print(f"JSON: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("holdout-lock")
def holdout_lock(
    image_path: Path = typer.Argument(..., help="Locked holdout image file or directory."),
    truth_path: Path = typer.Argument(..., help="Truth manifest JSON or directory of truth JSON files."),
    output_json: Path = typer.Option(Path("runs/holdout/holdout_lock.json"), "--output", "-o"),
    name: str = typer.Option("BananaVision locked holdout", "--name"),
    target_count_error_rate: float = typer.Option(0.01, "--target-count-error-rate"),
) -> None:
    lock = lock_holdout(
        image_path,
        truth_path,
        output_json,
        name=name,
        target_count_error_rate=target_count_error_rate,
    )
    console.print(
        f"[green]Holdout locked:[/green] {lock['image_count']} image(s), "
        f"{lock['truth_count']} annotated plant(s)"
    )
    console.print(f"Minimum detectable count error: {lock['min_detectable_count_error_rate']:.6f}")
    console.print(f"Lock: {output_json}")


@app.command("holdout-verify")
def holdout_verify(
    lock_json: Path = typer.Argument(..., help="holdout_lock.json generated by holdout-lock."),
    output_json: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    report = verify_holdout_lock(lock_json, output_json)
    color = "green" if report["status"] == "pass" else "red"
    console.print(
        f"[{color}]Holdout verify {str(report['status']).upper()}:[/{color}] "
        f"{report['verified_entries']} entries, {report['issue_count']} issue(s)"
    )
    if output_json is not None:
        console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("validation-plan")
def validation_plan(
    output_json: Path = typer.Option(Path("runs/validation/validation_plan.json"), "--output", "-o"),
    target_count_error_rate: float = typer.Option(0.01, "--target-count-error-rate"),
    target_cluster_recall_loss: float = typer.Option(0.01, "--target-cluster-recall-loss"),
    target_cluster_full_detection_loss: float = typer.Option(0.01, "--target-cluster-full-detection-loss"),
    farms: int = typer.Option(3, "--farms"),
    flight_dates: int = typer.Option(3, "--flight-dates"),
    gsd_bands: int = typer.Option(2, "--gsd-bands"),
    cultivars: int = typer.Option(1, "--cultivars"),
    min_plants_per_condition: int = typer.Option(50, "--min-plants-per-condition"),
    min_cluster_mats_per_condition: int = typer.Option(10, "--min-cluster-mats-per-condition"),
    min_cluster_truth_fraction: float = typer.Option(0.20, "--min-cluster-truth-fraction"),
) -> None:
    try:
        report = build_validation_plan(
            target_count_error_rate=target_count_error_rate,
            target_cluster_recall_loss=target_cluster_recall_loss,
            target_cluster_full_detection_loss=target_cluster_full_detection_loss,
            farms=farms,
            flight_dates=flight_dates,
            gsd_bands=gsd_bands,
            cultivars=cultivars,
            min_plants_per_condition=min_plants_per_condition,
            min_cluster_mats_per_condition=min_cluster_mats_per_condition,
            min_cluster_truth_fraction=min_cluster_truth_fraction,
        )
    except ValueError as exc:
        console.print(f"[red]Validation plan failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    write_validation_plan(report, output_json)
    support = report["minimum_support"]
    resolution = report["resolution"]
    console.print(
        "[green]Validation plan written:[/green] "
        f"truth>={support['truth_count']}, clusters>={support['cluster_count']}, "
        f"cluster_truth>={support['cluster_truth_count']}, "
        f"min_detectable_count_error={resolution['min_detectable_count_error_rate']:.6f}"
    )
    console.print(f"Report: {output_json}")


@app.command("truth-coverage")
def truth_coverage(
    truth_path: Path = typer.Argument(..., help="Truth manifest JSON, truth JSON, or directory of truth JSON files."),
    image_path: Path | None = typer.Option(None, "--images", help="Optional image file or directory to bind truth lookup."),
    output_json: Path = typer.Option(Path("runs/truth_coverage/truth_coverage_report.json"), "--output", "-o"),
    min_truth_count: int = typer.Option(0, "--min-truth-count"),
    min_cluster_count: int = typer.Option(0, "--min-cluster-count"),
    min_cluster_truth_count: int = typer.Option(0, "--min-cluster-truth-count"),
    min_cluster_images: int = typer.Option(0, "--min-cluster-images"),
    min_cluster_truth_fraction: float = typer.Option(0.0, "--min-cluster-truth-fraction"),
) -> None:
    report = audit_truth_coverage(
        truth_path,
        image_path=image_path,
        min_truth_count=min_truth_count,
        min_cluster_count=min_cluster_count,
        min_cluster_truth_count=min_cluster_truth_count,
        min_cluster_images=min_cluster_images,
        min_cluster_truth_fraction=min_cluster_truth_fraction,
    )
    write_truth_coverage_report(report, output_json)
    color = "green" if report["status"] == "pass" else "red"
    console.print(
        f"[{color}]Truth coverage {str(report['status']).upper()}:[/{color}] "
        f"images={report['image_count']}, truth={report['truth_count']}, "
        f"clusters={report['cluster_count']}, cluster_truth={report['cluster_truth_count']}, "
        f"cluster_images={report['cluster_image_count']}"
    )
    console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("stratified-truth-coverage")
def stratified_truth_coverage(
    truth_path: Path = typer.Argument(..., help="Truth manifest JSON, truth JSON, or directory of truth JSON files."),
    metadata_csv: Path = typer.Argument(..., help="CSV with image plus strata columns such as farm,date,gsd_band."),
    image_path: Path | None = typer.Option(None, "--images", help="Optional image file or directory to bind truth lookup."),
    output_json: Path = typer.Option(
        Path("runs/stratified_truth_coverage/stratified_truth_coverage_report.json"),
        "--output",
        "-o",
    ),
    strata: list[str] = typer.Option(["farm", "flight_date", "gsd_band", "cultivar"], "--strata"),
    min_truth_count: int = typer.Option(0, "--min-truth-count"),
    min_cluster_count: int = typer.Option(0, "--min-cluster-count"),
    min_cluster_truth_count: int = typer.Option(0, "--min-cluster-truth-count"),
    min_cluster_images: int = typer.Option(0, "--min-cluster-images"),
    min_cluster_truth_fraction: float = typer.Option(0.0, "--min-cluster-truth-fraction"),
) -> None:
    try:
        report = build_stratified_truth_coverage_report(
            truth_path,
            metadata_csv,
            output_json,
            strata_keys=strata,
            image_path=image_path,
            min_truth_count=min_truth_count,
            min_cluster_count=min_cluster_count,
            min_cluster_truth_count=min_cluster_truth_count,
            min_cluster_images=min_cluster_images,
            min_cluster_truth_fraction=min_cluster_truth_fraction,
        )
    except ValueError as exc:
        console.print(f"[red]Stratified truth coverage failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    color = "green" if report["status"] == "pass" else "red"
    console.print(
        f"[{color}]Stratified truth coverage {str(report['status']).upper()}:[/{color}] "
        f"strata={report['stratum_count']}, failed={report['failed_stratum_count']}, "
        f"missing_metadata={report['missing_metadata_count']}"
    )
    console.print(f"Report: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("truth-quality")
def truth_quality(
    truth_path: Path = typer.Argument(..., help="Truth manifest JSON, truth JSON, or directory of truth JSON files."),
    image_path: Path | None = typer.Option(None, "--images", help="Optional image file or directory for bounds checks."),
    output_json: Path = typer.Option(Path("runs/truth_quality/truth_quality_report.json"), "--output", "-o"),
    min_center_distance_px: float = typer.Option(2.0, "--min-center-distance-px"),
    max_group_size: int = typer.Option(6, "--max-group-size"),
    allow_singleton_groups: bool = typer.Option(False, "--allow-singleton-groups"),
) -> None:
    thresholds = TruthQualityThresholds(
        min_center_distance_px=min_center_distance_px,
        max_group_size=max_group_size,
        allow_singleton_groups=allow_singleton_groups,
    )
    try:
        report = audit_truth_quality(truth_path, output_json, image_path=image_path, thresholds=thresholds)
    except ValueError as exc:
        console.print(f"[red]Truth quality failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    color = "green" if report["status"] == "pass" else "red"
    console.print(
        f"[{color}]Truth quality {str(report['status']).upper()}:[/{color}] "
        f"images={report['image_count']}, truth={report['truth_count']}, issues={report['issue_count']}"
    )
    console.print(f"Report: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command()
def benchmark(
    input_path: Path = typer.Argument(..., help="Image file or directory."),
    output: Path = typer.Option(Path("runs/benchmark/benchmark_report.json"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    runs: int = typer.Option(3, "--runs"),
    warmup: int = typer.Option(1, "--warmup"),
) -> None:
    cfg = load_config(config, {"detector": detector, "model_path": model_path})
    report = benchmark_images(input_path, cfg, runs=runs, warmup=warmup)
    path = write_benchmark_report(report, output)
    latency = report["latency_ms"]
    console.print(
        "[green]Benchmark complete[/green] "
        f"median={latency['median']:.3f}ms p95={latency['p95']:.3f}ms max={latency['max']:.3f}ms"
    )
    console.print(f"Report: {path}")


@app.command("cluster-benchmark")
def cluster_benchmark(
    output_dir: Path = typer.Option(Path("runs/cluster_benchmark"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    gsd_cm: float | None = typer.Option(None, "--gsd-cm"),
    expected_crown_diameter_m: float | None = typer.Option(None, "--crown-m"),
    min_component_area_px: int | None = typer.Option(None, "--min-component-area-px"),
    max_split_instances: int | None = typer.Option(None, "--max-split-instances"),
    center_distance_weight: float | None = typer.Option(None, "--center-distance-weight"),
    rgb_threshold_quantile: float | None = typer.Option(None, "--rgb-threshold-quantile"),
    scenes: int = typer.Option(3, "--scenes"),
    width: int = typer.Option(320, "--width"),
    height: int = typer.Option(240, "--height"),
    plants_per_scene: int = typer.Option(12, "--plants-per-scene"),
    clustered_mats_per_scene: int = typer.Option(3, "--clustered-mats-per-scene"),
    min_plants_per_mat: int = typer.Option(3, "--min-plants-per-mat"),
    max_plants_per_mat: int = typer.Option(3, "--max-plants-per-mat"),
    cluster_spread_px: float = typer.Option(24.0, "--cluster-spread-px"),
    seed: int = typer.Option(21, "--seed"),
    tolerance_px: float = typer.Option(45.0, "--tolerance-px"),
    max_count_error_rate: float = typer.Option(0.5, "--max-count-error-rate"),
    max_mean_image_count_error_rate: float | None = typer.Option(0.6, "--max-mean-image-count-error-rate"),
    min_precision: float = typer.Option(0.4, "--min-precision"),
    min_recall: float = typer.Option(0.4, "--min-recall"),
    min_f1: float = typer.Option(0.4, "--min-f1"),
    min_cluster_recall: float = typer.Option(0.5, "--min-cluster-recall"),
    min_cluster_full_detection_rate: float = typer.Option(0.25, "--min-cluster-full-detection-rate"),
) -> None:
    overrides: dict[str, object | None] = {
        "detector": detector,
        "model_path": model_path,
        "gsd_cm": gsd_cm,
        "expected_crown_diameter_m": expected_crown_diameter_m,
        "min_component_area_px": min_component_area_px,
        "max_split_instances": max_split_instances,
        "center_distance_weight": center_distance_weight,
        "rgb_threshold_quantile": rgb_threshold_quantile,
    }
    if config is None:
        overrides.update(
            {
                "gsd_cm": gsd_cm if gsd_cm is not None else 2.0,
                "expected_crown_diameter_m": (
                    expected_crown_diameter_m if expected_crown_diameter_m is not None else 0.55
                ),
                "min_component_area_px": min_component_area_px if min_component_area_px is not None else 20,
                "max_split_instances": max_split_instances if max_split_instances is not None else 12,
                "center_distance_weight": center_distance_weight if center_distance_weight is not None else 0.35,
                "rgb_threshold_quantile": rgb_threshold_quantile if rgb_threshold_quantile is not None else 0.78,
            }
        )
    cfg = load_config(config, overrides)
    report = run_cluster_benchmark(
        output_dir,
        cfg,
        scenes=scenes,
        width=width,
        height=height,
        plants_per_scene=plants_per_scene,
        clustered_mats_per_scene=clustered_mats_per_scene,
        min_plants_per_mat=min_plants_per_mat,
        max_plants_per_mat=max_plants_per_mat,
        cluster_spread_px=cluster_spread_px,
        seed=seed,
        tolerance_px=tolerance_px,
        max_count_error_rate=max_count_error_rate,
        max_mean_image_count_error_rate=max_mean_image_count_error_rate,
        min_precision=min_precision,
        min_recall=min_recall,
        min_f1=min_f1,
        min_cluster_recall=min_cluster_recall,
        min_cluster_full_detection_rate=min_cluster_full_detection_rate,
    )
    metrics = report["metrics"]
    color = "green" if report["status"] == "pass" else "red"
    console.print(
        f"[{color}]Cluster benchmark {str(report['status']).upper()}:[/{color}] "
        f"images={metrics['images']}, count_error_rate={metrics['count_error_rate']:.4f}, "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"cluster_recall={metrics['cluster_recall']:.4f}, "
        f"cluster_full_rate={metrics['fully_detected_cluster_rate']:.4f}"
    )
    console.print(f"Report: {output_dir / 'cluster_benchmark_report.json'}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command()
def calibrate(
    detections_json: Path = typer.Argument(..., help="*.detections.json file."),
    truth_json: Path = typer.Argument(..., help="Truth JSON with centers."),
    output_json: Path = typer.Option(Path("runs/calibration/calibration_report.json"), "--output", "-o"),
    tolerance_px: float = typer.Option(24.0, "--tolerance-px"),
) -> None:
    report = calibrate_thresholds(detections_json, truth_json, tolerance_px=tolerance_px)
    path = write_calibration_report(report, output_json)
    best = report.get("best") or {}
    metrics = best.get("metrics", {})
    console.print(
        "[green]Calibration complete[/green] "
        f"best_threshold={best.get('threshold')} "
        f"count_error_rate={metrics.get('count_error_rate')} "
        f"f1={metrics.get('f1')}"
    )
    console.print(f"Report: {path}")


@app.command("tune-config")
def tune_config_cmd(
    image: Path = typer.Argument(..., help="Calibration image."),
    truth: Path = typer.Argument(..., help="Truth JSON with plant centers."),
    output_json: Path = typer.Option(Path("runs/tuning/tuning_report.json"), "--output", "-o"),
    output_config: Path = typer.Option(Path("runs/tuning/tuned_config.yaml"), "--output-config"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    detector: str | None = typer.Option(None, "--detector"),
    model_path: str | None = typer.Option(None, "--model"),
    tolerance_px: float = typer.Option(24.0, "--tolerance-px"),
    crown_m: list[float] = typer.Option([1.8, 2.2, 2.6], "--crown-m"),
    min_distance_ratio: list[float] = typer.Option([0.35, 0.42, 0.5], "--min-distance-ratio"),
    center_distance_weight: list[float] = typer.Option([0.2, 0.35, 0.6], "--center-distance-weight"),
    canopy_fill_ratio: list[float] = typer.Option([0.48, 0.58, 0.68], "--canopy-fill-ratio"),
    rgb_quantile: list[float] = typer.Option([0.76, 0.82, 0.88], "--rgb-quantile"),
    max_split_instances: list[int] = typer.Option([8, 12, 16], "--max-split-instances"),
) -> None:
    cfg = load_config(config, {"detector": detector, "model_path": model_path})
    report = tune_config(
        image,
        truth,
        cfg,
        tolerance_px=tolerance_px,
        crown_diameters_m=crown_m,
        min_center_distance_ratios=min_distance_ratio,
        center_distance_weights=center_distance_weight,
        canopy_fill_ratios=canopy_fill_ratio,
        rgb_threshold_quantiles=rgb_quantile,
        max_split_instances=max_split_instances,
    )
    report_path = write_tuning_report(report, output_json)
    config_path = write_tuned_config(report, output_config)
    best = report.get("best") or {}
    metrics = best.get("metrics", {})
    console.print(
        "[green]Tuning complete[/green] "
        f"count_error_rate={metrics.get('count_error_rate')} "
        f"f1={metrics.get('f1')} "
        f"count={best.get('count')}"
    )
    console.print(f"Report: {report_path}")
    console.print(f"Tuned config: {config_path}")


@app.command()
def serve(
    config: Path | None = typer.Option(None, "--config", "-c"),
    host: str = "0.0.0.0",
    port: int = 8080,
    api_key_env: str = typer.Option("BANANAVISION_API_KEY", "--api-key-env"),
    max_upload_mb: float = typer.Option(25.0, "--max-upload-mb"),
) -> None:
    try:
        import uvicorn
    except Exception as exc:
        raise typer.BadParameter("Install API dependencies: pip install 'bananavision-drone[api]'") from exc
    from .api import create_app

    api_key = os.getenv(api_key_env) if api_key_env else None
    uvicorn.run(create_app(config, api_key=api_key, max_upload_mb=max_upload_mb), host=host, port=port, reload=False)


@app.command()
def train(
    data: Path = typer.Argument(..., help="Ultralytics data.yaml."),
    model: str = typer.Option("yolo26n-seg.pt", "--model"),
    epochs: int = 120,
    imgsz: int = 1024,
    batch: int = 8,
    device: str | None = None,
) -> None:
    result = train_yolo(data, model=model, epochs=epochs, imgsz=imgsz, batch=batch, device=device)
    console.print(result)


@app.command()
def validate(
    data: Path = typer.Argument(..., help="Ultralytics data.yaml."),
    model: Path = typer.Argument(..., help="Trained .pt/.onnx/.engine model."),
    imgsz: int = 1024,
    batch: int = 8,
    device: str | None = None,
) -> None:
    result = validate_yolo(data, model=model, imgsz=imgsz, batch=batch, device=device)
    console.print(result)


@app.command("export")
def export_model(
    model: Path = typer.Argument(..., help="Trained model path."),
    fmt: str = typer.Option("onnx", "--format"),
    imgsz: int = 1024,
    half: bool = False,
    int8: bool = False,
    device: str | None = None,
) -> None:
    result = export_yolo(model, fmt=fmt, imgsz=imgsz, half=half, int8=int8, device=device)
    console.print(result)


@app.command("audit-dataset")
def audit_dataset(data: Path = typer.Argument(..., help="Ultralytics data.yaml.")) -> None:
    audit = audit_yolo_dataset(data)
    console.print(json.dumps(audit.__dict__, indent=2))
    if not audit.ok:
        raise typer.Exit(code=2)


@app.command("quality-report")
def quality_report(
    data: Path = typer.Argument(..., help="Ultralytics data.yaml."),
    output_json: Path = typer.Option(Path("runs/quality/quality_report.json"), "--output", "-o"),
    require_test: bool = typer.Option(False, "--require-test"),
    no_fail: bool = typer.Option(False, "--no-fail"),
) -> None:
    report = audit_dataset_quality(data, require_test=require_test)
    path = write_quality_report(report, output_json)
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(f"[{color}]Dataset quality {str(report['status']).upper()}[/{color}]")
    console.print(f"Issues: {len(report['issues'])}")
    console.print(f"Report: {path}")
    if report["status"] == "fail" and not no_fail:
        raise typer.Exit(code=2)


@app.command("tile-dataset")
def tile_dataset(
    input_root: Path = typer.Argument(..., help="Dataset root with images/<split> and labels/<split>."),
    output_root: Path = typer.Argument(..., help="Output dataset root."),
    split: str = typer.Option("train", "--split"),
    tile_size: int = typer.Option(1024, "--tile-size"),
    overlap: int = typer.Option(128, "--overlap"),
    min_polygon_area_px: float = typer.Option(64.0, "--min-polygon-area-px"),
    include_empty: bool = typer.Option(False, "--include-empty"),
) -> None:
    summary = tile_yolo_dataset(
        input_root,
        output_root,
        split=split,
        tile_size=tile_size,
        overlap=overlap,
        min_polygon_area_px=min_polygon_area_px,
        include_empty=include_empty,
    )
    console.print(json.dumps(summary.to_dict(), indent=2))


@app.command("split-dataset")
def split_dataset(
    image_dir: Path = typer.Argument(..., help="Directory containing source images."),
    label_dir: Path = typer.Argument(..., help="Directory containing YOLO label files."),
    output_root: Path = typer.Argument(..., help="Output YOLO dataset root."),
    manifest_csv: Path | None = typer.Option(None, "--manifest-csv", help="CSV with image,group columns."),
    train_ratio: float = typer.Option(0.7, "--train-ratio"),
    val_ratio: float = typer.Option(0.2, "--val-ratio"),
    test_ratio: float = typer.Option(0.1, "--test-ratio"),
    seed: int = typer.Option(7, "--seed"),
    class_name: list[str] = typer.Option(["banana_plant"], "--class-name"),
    no_empty_labels: bool = typer.Option(False, "--no-empty-labels"),
) -> None:
    summary = split_yolo_dataset(
        image_dir,
        label_dir,
        output_root,
        manifest_csv=manifest_csv,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        class_names=class_name,
        write_empty_labels=not no_empty_labels,
    )
    console.print(json.dumps(summary.to_dict(), indent=2))


@annotations_app.command("coco")
def annotations_coco(
    coco_json: Path = typer.Argument(..., help="COCO instances JSON."),
    output_labels: Path = typer.Argument(..., help="YOLO labels output directory."),
    target_name: list[str] = typer.Option(["banana_plant"], "--target-name"),
) -> None:
    summary = convert_coco_to_yolo_seg(coco_json, output_labels, target_names=set(target_name))
    console.print(json.dumps(summary.to_dict(), indent=2))


@annotations_app.command("labelme")
def annotations_labelme(
    labelme_dir: Path = typer.Argument(..., help="Directory with LabelMe JSON files."),
    output_labels: Path = typer.Argument(..., help="YOLO labels output directory."),
    target_label: str = typer.Option("banana_plant", "--target-label"),
) -> None:
    summary = convert_labelme_to_yolo_seg(labelme_dir, output_labels, target_label=target_label)
    console.print(json.dumps(summary.to_dict(), indent=2))


@app.command("review-queue")
def review_queue(
    predictions_dir: Path = typer.Argument(..., help="Directory containing *.detections.json files."),
    output_json: Path = typer.Option(Path("runs/active_learning/review_queue.json"), "--output", "-o"),
    low_confidence: float = typer.Option(0.45, "--low-confidence"),
    high_split_count: int = typer.Option(2, "--high-split-count"),
) -> None:
    items = build_review_queue(
        predictions_dir,
        output_json,
        low_confidence=low_confidence,
        high_split_count=high_split_count,
    )
    console.print(f"[green]Review queue written with {len(items)} item(s).[/green]")
    console.print(f"JSON: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")


@app.command("review-crops")
def review_crops(
    queue_json: Path = typer.Argument(..., help="review_queue.json generated by review-queue."),
    output_dir: Path = typer.Option(Path("runs/active_learning/review_crops"), "--output", "-o"),
    margin_px: int = typer.Option(32, "--margin-px"),
    max_items: int | None = typer.Option(None, "--max-items"),
) -> None:
    manifest = export_review_crops(queue_json, output_dir, margin_px=margin_px, max_items=max_items)
    console.print(
        f"[green]Review crops exported:[/green] {manifest['exported_count']} "
        f"exported, {manifest['skipped_count']} skipped"
    )
    console.print(f"Manifest: {output_dir / 'review_crops_manifest.json'}")


@app.command("cluster-review")
def cluster_review(
    detections_json: Path = typer.Argument(..., help="*.detections.json from inference or acceptance."),
    truth_json: Path = typer.Argument(..., help="Truth JSON with grouped banana mats."),
    output_json: Path = typer.Option(Path("runs/cluster_review/cluster_review.json"), "--output", "-o"),
    tolerance_px: float = typer.Option(24.0, "--tolerance-px"),
    image: Path | None = typer.Option(None, "--image"),
    crops_dir: Path | None = typer.Option(None, "--crops-dir"),
    crop_margin_px: int = typer.Option(48, "--crop-margin-px"),
) -> None:
    report = build_cluster_review(
        detections_json,
        truth_json,
        output_json,
        tolerance_px=tolerance_px,
        image_path=image,
        crops_dir=crops_dir,
        crop_margin_px=crop_margin_px,
    )
    summary = report["summary"]
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Cluster review {str(report['status']).upper()}:[/{color}] "
        f"{summary['failed_cluster_count']}/{summary['cluster_count']} grouped mat(s) failed"
    )
    console.print(f"JSON: {output_json}")
    console.print(f"CSV: {output_json.with_suffix('.csv')}")
    if crops_dir is not None:
        console.print(f"Crops: {crops_dir}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("register-model")
def register_model_cmd(
    model: Path = typer.Argument(..., help="Model file to register."),
    version: str = typer.Argument(..., help="Human-readable model version."),
    registry_dir: Path = typer.Option(Path("models/registry"), "--registry-dir"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    acceptance_report: Path | None = typer.Option(None, "--acceptance-report"),
    benchmark_report: Path | None = typer.Option(None, "--benchmark-report"),
    notes: str = typer.Option("", "--notes"),
) -> None:
    cfg = load_config(config, {"detector": "yolo-seg", "model_path": str(model)})
    manifest_path = register_model(
        model,
        registry_dir,
        version=version,
        config=cfg,
        acceptance_report=acceptance_report,
        benchmark_report=benchmark_report,
        notes=notes,
    )
    console.print(f"[green]Model registered:[/green] {manifest_path}")
    console.print(f"Latest: {manifest_path.parent / 'latest.json'}")


@app.command("promote-model")
def promote_model_cmd(
    model: Path = typer.Argument(..., help="Model file to promote."),
    version: str = typer.Argument(..., help="Production model version."),
    acceptance_report: Path = typer.Argument(..., help="Passing acceptance or acceptance-batch report."),
    benchmark_report: Path = typer.Argument(..., help="Benchmark report."),
    registry_dir: Path = typer.Option(Path("models/registry"), "--registry-dir"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    max_p95_ms: float | None = typer.Option(None, "--max-p95-ms"),
    notes: str = typer.Option("", "--notes"),
) -> None:
    cfg = load_config(config, {"detector": "yolo-seg", "model_path": str(model)})
    manifest_path = promote_model(
        model,
        registry_dir,
        version=version,
        config=cfg,
        acceptance_report=acceptance_report,
        benchmark_report=benchmark_report,
        max_p95_ms=max_p95_ms,
        notes=notes,
    )
    console.print(f"[green]Model promoted:[/green] {manifest_path}")
    console.print(f"Latest: {manifest_path.parent / 'latest.json'}")


@app.command("model-card")
def model_card(
    output_md: Path = typer.Option(Path("docs/MODEL_CARD.generated.md"), "--output", "-o"),
    model_name: str = typer.Option("BananaVision model", "--model-name"),
    version: str = typer.Option("unversioned", "--version"),
    architecture: str = typer.Option("YOLO segmentation or BananaVision RGB baseline", "--architecture"),
    model_manifest: Path | None = typer.Option(None, "--model-manifest"),
    acceptance_report: Path | None = typer.Option(None, "--acceptance-report"),
    benchmark_report: Path | None = typer.Option(None, "--benchmark-report"),
    mission_quality_report: Path | None = typer.Option(None, "--mission-quality-report"),
    prediction_quality_report: Path | None = typer.Option(None, "--prediction-quality-report"),
    flight_log_report: Path | None = typer.Option(None, "--flight-log-report"),
    domain_check_report: Path | None = typer.Option(None, "--domain-check-report"),
    geo_accuracy_report: Path | None = typer.Option(None, "--geo-accuracy-report"),
    dataset_quality_report: Path | None = typer.Option(None, "--dataset-quality-report"),
    validation_plan_report: Path | None = typer.Option(None, "--validation-plan-report"),
    stratified_acceptance_report: Path | None = typer.Option(None, "--stratified-acceptance-report"),
    truth_quality_report: Path | None = typer.Option(None, "--truth-quality-report"),
    truth_coverage_report: Path | None = typer.Option(None, "--truth-coverage-report"),
    stratified_truth_coverage_report: Path | None = typer.Option(None, "--stratified-truth-coverage-report"),
    intended_use: str = typer.Option(
        "Banana plant instance detection and counting from UAV imagery inside the validated operating domain.",
        "--intended-use",
    ),
    notes: str = typer.Option("", "--notes"),
) -> None:
    path = build_model_card(
        output_md,
        model_name=model_name,
        version=version,
        architecture=architecture,
        model_manifest=model_manifest,
        acceptance_report=acceptance_report,
        benchmark_report=benchmark_report,
        mission_quality_report=mission_quality_report,
        prediction_quality_report=prediction_quality_report,
        flight_log_report=flight_log_report,
        domain_check_report=domain_check_report,
        geo_accuracy_report=geo_accuracy_report,
        dataset_quality_report=dataset_quality_report,
        validation_plan_report=validation_plan_report,
        stratified_acceptance_report=stratified_acceptance_report,
        truth_quality_report=truth_quality_report,
        truth_coverage_report=truth_coverage_report,
        stratified_truth_coverage_report=stratified_truth_coverage_report,
        intended_use=intended_use,
        notes=notes,
    )
    console.print(f"[green]Model card written:[/green] {path}")


@app.command("publication-audit")
def publication_audit(
    root: Path = typer.Argument(Path("."), help="Repository root to audit."),
    output_json: Path = typer.Option(Path("runs/publication/publication_audit.json"), "--output", "-o"),
) -> None:
    report = audit_publication(root, output_json)
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(f"[{color}]Publication audit {str(report['status']).upper()}[/{color}]")
    for check in report["checks"]:
        console.print(f"- {check['name']}: {check['status']} - {check['detail']}")
    console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("release-audit")
def release_audit(
    output_json: Path = typer.Option(Path("runs/release_audit/release_audit.json"), "--output", "-o"),
    acceptance_report: Path | None = typer.Option(None, "--acceptance-report"),
    stratified_acceptance_report: Path | None = typer.Option(None, "--stratified-acceptance-report"),
    benchmark_report: Path | None = typer.Option(None, "--benchmark-report"),
    mission_quality_report: Path | None = typer.Option(None, "--mission-quality-report"),
    prediction_quality_report: Path | None = typer.Option(None, "--prediction-quality-report"),
    holdout_verify_report: Path | None = typer.Option(None, "--holdout-verify-report"),
    validation_plan_report: Path | None = typer.Option(None, "--validation-plan-report"),
    truth_quality_report: Path | None = typer.Option(None, "--truth-quality-report"),
    truth_coverage_report: Path | None = typer.Option(None, "--truth-coverage-report"),
    stratified_truth_coverage_report: Path | None = typer.Option(None, "--stratified-truth-coverage-report"),
    flight_check_report: Path | None = typer.Option(None, "--flight-check-report"),
    flight_log_report: Path | None = typer.Option(None, "--flight-log-report"),
    domain_check_report: Path | None = typer.Option(None, "--domain-check-report"),
    geo_accuracy_report: Path | None = typer.Option(None, "--geo-accuracy-report"),
    model_manifest: Path | None = typer.Option(None, "--model-manifest"),
    model_card_path: Path | None = typer.Option(None, "--model-card"),
    field_report: Path | None = typer.Option(None, "--field-report"),
    max_count_error_rate: float = typer.Option(0.01, "--max-count-error-rate"),
    min_truth_count: int | None = typer.Option(None, "--min-truth-count"),
    min_precision_ci_lower: float | None = typer.Option(None, "--min-precision-ci-lower"),
    min_recall_ci_lower: float | None = typer.Option(None, "--min-recall-ci-lower"),
    min_cluster_recall: float | None = typer.Option(None, "--min-cluster-recall"),
    min_cluster_full_detection_rate: float | None = typer.Option(None, "--min-cluster-full-detection-rate"),
    min_cluster_count: int | None = typer.Option(None, "--min-cluster-count"),
    min_cluster_truth_count: int | None = typer.Option(None, "--min-cluster-truth-count"),
    min_cluster_images: int | None = typer.Option(None, "--min-cluster-images"),
    min_cluster_truth_fraction: float | None = typer.Option(None, "--min-cluster-truth-fraction"),
    max_p95_ms: float | None = typer.Option(None, "--max-p95-ms"),
    max_geo_rmse_m: float = typer.Option(1.0, "--max-geo-rmse-m"),
    max_geo_p95_m: float | None = typer.Option(None, "--max-geo-p95-m"),
    min_geo_recall: float = typer.Option(0.99, "--min-geo-recall"),
    allow_warn_quality: bool = typer.Option(False, "--allow-warn-quality"),
) -> None:
    report = audit_release(
        output_json,
        acceptance_report=acceptance_report,
        stratified_acceptance_report=stratified_acceptance_report,
        benchmark_report=benchmark_report,
        mission_quality_report=mission_quality_report,
        prediction_quality_report=prediction_quality_report,
        holdout_verify_report=holdout_verify_report,
        validation_plan_report=validation_plan_report,
        truth_quality_report=truth_quality_report,
        truth_coverage_report=truth_coverage_report,
        stratified_truth_coverage_report=stratified_truth_coverage_report,
        flight_check_report=flight_check_report,
        flight_log_report=flight_log_report,
        domain_check_report=domain_check_report,
        geo_accuracy_report=geo_accuracy_report,
        model_manifest=model_manifest,
        model_card=model_card_path,
        field_report=field_report,
        max_count_error_rate=max_count_error_rate,
        min_truth_count=min_truth_count,
        min_precision_ci_lower=min_precision_ci_lower,
        min_recall_ci_lower=min_recall_ci_lower,
        min_cluster_recall=min_cluster_recall,
        min_cluster_full_detection_rate=min_cluster_full_detection_rate,
        min_cluster_count=min_cluster_count,
        min_cluster_truth_count=min_cluster_truth_count,
        min_cluster_images=min_cluster_images,
        min_cluster_truth_fraction=min_cluster_truth_fraction,
        max_p95_ms=max_p95_ms,
        max_geo_rmse_m=max_geo_rmse_m,
        max_geo_p95_m=max_geo_p95_m,
        min_geo_recall=min_geo_recall,
        allow_warn_quality=allow_warn_quality,
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(f"[{color}]Release audit {str(report['status']).upper()}[/{color}]")
    for gate in report["gates"]:
        console.print(f"- {gate['name']}: {gate['status']} - {gate['detail']}")
    console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("release-package")
def release_package(
    release_audit_report: Path = typer.Argument(..., help="Passing release_audit.json."),
    output_dir: Path = typer.Option(Path("dist"), "--output", "-o"),
    package_name: str = typer.Option("bananavision-release", "--package-name"),
    model: Path | None = typer.Option(None, "--model"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    model_manifest: Path | None = typer.Option(None, "--model-manifest"),
    model_card_path: Path | None = typer.Option(None, "--model-card"),
    field_report: Path | None = typer.Option(None, "--field-report"),
    acceptance_report: Path | None = typer.Option(None, "--acceptance-report"),
    stratified_acceptance_report: Path | None = typer.Option(None, "--stratified-acceptance-report"),
    benchmark_report: Path | None = typer.Option(None, "--benchmark-report"),
    mission_quality_report: Path | None = typer.Option(None, "--mission-quality-report"),
    prediction_quality_report: Path | None = typer.Option(None, "--prediction-quality-report"),
    mission_audit_report: Path | None = typer.Option(None, "--mission-audit-report"),
    evidence_manifest: Path | None = typer.Option(None, "--evidence-manifest"),
    holdout_verify_report: Path | None = typer.Option(None, "--holdout-verify-report"),
    validation_plan_report: Path | None = typer.Option(None, "--validation-plan-report"),
    truth_quality_report: Path | None = typer.Option(None, "--truth-quality-report"),
    truth_coverage_report: Path | None = typer.Option(None, "--truth-coverage-report"),
    stratified_truth_coverage_report: Path | None = typer.Option(None, "--stratified-truth-coverage-report"),
    flight_check_report: Path | None = typer.Option(None, "--flight-check-report"),
    flight_log_report: Path | None = typer.Option(None, "--flight-log-report"),
    domain_check_report: Path | None = typer.Option(None, "--domain-check-report"),
    geo_accuracy_report: Path | None = typer.Option(None, "--geo-accuracy-report"),
    deployment_manifest: Path | None = typer.Option(None, "--deployment-manifest"),
    readme_file: Path | None = typer.Option(Path("README.md"), "--readme"),
    license_file: Path | None = typer.Option(Path("LICENSE"), "--license"),
    artifact: list[str] = typer.Option([], "--artifact", help="Additional artifact as LABEL=PATH."),
    allow_failed_audit: bool = typer.Option(False, "--allow-failed-audit"),
    no_zip: bool = typer.Option(False, "--no-zip"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    artifacts = _release_package_artifacts(
        model=model,
        config=config,
        model_manifest=model_manifest,
        model_card_path=model_card_path,
        field_report=field_report,
        acceptance_report=acceptance_report,
        stratified_acceptance_report=stratified_acceptance_report,
        benchmark_report=benchmark_report,
        mission_quality_report=mission_quality_report,
        prediction_quality_report=prediction_quality_report,
        mission_audit_report=mission_audit_report,
        evidence_manifest=evidence_manifest,
        holdout_verify_report=holdout_verify_report,
        validation_plan_report=validation_plan_report,
        truth_quality_report=truth_quality_report,
        truth_coverage_report=truth_coverage_report,
        stratified_truth_coverage_report=stratified_truth_coverage_report,
        flight_check_report=flight_check_report,
        flight_log_report=flight_log_report,
        domain_check_report=domain_check_report,
        geo_accuracy_report=geo_accuracy_report,
        deployment_manifest=deployment_manifest,
        readme_file=readme_file,
        license_file=license_file,
    )
    artifacts.extend(parse_artifact_specs(artifact))
    manifest = build_release_package(
        output_dir=output_dir,
        release_audit_report=release_audit_report,
        artifacts=artifacts,
        package_name=package_name,
        allow_failed_audit=allow_failed_audit,
        create_zip=not no_zip,
        overwrite=overwrite,
    )
    color = "green" if manifest["package_status"] == "release" else "yellow"
    console.print(
        f"[{color}]Release package {manifest['package_status'].upper()}:[/{color}] "
        f"{manifest['artifact_count']} artifact(s)"
    )
    console.print(f"Package: {manifest['package_root']}")
    console.print(f"Manifest: {Path(manifest['package_root']) / manifest['manifest_path']}")
    if manifest.get("zip_path"):
        console.print(f"ZIP: {output_dir / str(manifest['zip_path'])}")
        console.print(f"ZIP SHA256: {manifest.get('zip_sha256')}")


@app.command("release-package-verify")
def release_package_verify(
    package_path: Path = typer.Argument(..., help="Release package folder, manifest JSON, or ZIP."),
    output_json: Path | None = typer.Option(None, "--output", "-o"),
    allow_exploratory: bool = typer.Option(False, "--allow-exploratory"),
    require_deployment_artifacts: bool = typer.Option(False, "--require-deployment-artifacts"),
) -> None:
    report = verify_release_package(
        package_path,
        output_json=output_json,
        allow_exploratory=allow_exploratory,
        require_deployment_artifacts=require_deployment_artifacts,
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Release package verify {str(report['status']).upper()}:[/{color}] "
        f"{report['verified_artifact_count']}/{report['artifact_count']} artifact(s) verified"
    )
    for check in report["checks"]:
        console.print(f"- {check['name']}: {check['status']} - {check['detail']}")
    if output_json is not None:
        console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("deployment-audit")
def deployment_audit(
    release_package: Path = typer.Argument(..., help="Installed release package folder, manifest JSON, or ZIP."),
    preflight_report: Path = typer.Argument(..., help="Preflight report from the target drone computer."),
    deployment_manifest: Path = typer.Argument(..., help="Deployment manifest installed with the package."),
    output_json: Path = typer.Option(Path("runs/deployment/deployment_audit.json"), "--output", "-o"),
    deployment_smoke_report: Path | None = typer.Option(None, "--deployment-smoke-report"),
    allow_warn_preflight: bool = typer.Option(False, "--allow-warn-preflight"),
    allow_exploratory_package: bool = typer.Option(False, "--allow-exploratory-package"),
    no_require_deployment_artifacts: bool = typer.Option(False, "--no-require-deployment-artifacts"),
    no_require_smoke_test: bool = typer.Option(False, "--no-require-smoke-test"),
) -> None:
    report = audit_deployment(
        output_json,
        release_package=release_package,
        preflight_report=preflight_report,
        deployment_manifest=deployment_manifest,
        deployment_smoke_report=deployment_smoke_report,
        allow_warn_preflight=allow_warn_preflight,
        allow_exploratory_package=allow_exploratory_package,
        require_deployment_artifacts=not no_require_deployment_artifacts,
        require_smoke_test=not no_require_smoke_test,
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(f"[{color}]Deployment audit {str(report['status']).upper()}[/{color}]")
    for gate in report["gates"]:
        console.print(f"- {gate['name']}: {gate['status']} - {gate['detail']}")
    console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command("mission-audit")
def mission_audit(
    run_manifest: Path = typer.Argument(..., help="Mission run_manifest.json from inference or mission-watch."),
    mission_quality_report: Path = typer.Argument(..., help="Mission quality JSON report."),
    prediction_quality_report: Path = typer.Argument(..., help="Prediction quality JSON report."),
    output_json: Path = typer.Option(Path("runs/mission_audit/mission_audit.json"), "--output", "-o"),
    flight_check_report: Path | None = typer.Option(None, "--flight-check-report"),
    domain_check_report: Path | None = typer.Option(None, "--domain-check-report"),
    flight_log_report: Path | None = typer.Option(None, "--flight-log-report"),
    capture_coverage_report: Path | None = typer.Option(None, "--capture-coverage-report"),
    geo_accuracy_report: Path | None = typer.Option(None, "--geo-accuracy-report"),
    preflight_report: Path | None = typer.Option(None, "--preflight-report"),
    deployment_audit_report: Path | None = typer.Option(None, "--deployment-audit-report"),
    field_report: Path | None = typer.Option(None, "--field-report"),
    min_detections: int = typer.Option(1, "--min-detections"),
    allow_warn_quality: bool = typer.Option(False, "--allow-warn-quality"),
    no_require_flight_check: bool = typer.Option(False, "--no-require-flight-check"),
    no_require_domain_check: bool = typer.Option(False, "--no-require-domain-check"),
    require_capture_coverage: bool = typer.Option(False, "--require-capture-coverage"),
    require_geo_accuracy: bool = typer.Option(False, "--require-geo-accuracy"),
    require_preflight: bool = typer.Option(False, "--require-preflight"),
    require_deployment_audit: bool = typer.Option(False, "--require-deployment-audit"),
) -> None:
    report = audit_mission_delivery(
        output_json,
        run_manifest=run_manifest,
        mission_quality_report=mission_quality_report,
        prediction_quality_report=prediction_quality_report,
        flight_check_report=flight_check_report,
        domain_check_report=domain_check_report,
        flight_log_report=flight_log_report,
        capture_coverage_report=capture_coverage_report,
        geo_accuracy_report=geo_accuracy_report,
        preflight_report=preflight_report,
        deployment_audit_report=deployment_audit_report,
        field_report=field_report,
        min_detections=min_detections,
        allow_warn_quality=allow_warn_quality,
        require_flight_check=not no_require_flight_check,
        require_domain_check=not no_require_domain_check,
        require_capture_coverage=require_capture_coverage,
        require_geo_accuracy=require_geo_accuracy,
        require_preflight=require_preflight,
        require_deployment_audit=require_deployment_audit,
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(f"[{color}]Mission audit {str(report['status']).upper()}[/{color}]")
    for gate in report["gates"]:
        console.print(f"- {gate['name']}: {gate['status']} - {gate['detail']}")
    console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


def _release_package_artifacts(**paths: Path | None) -> list[PackageArtifact]:
    label_map = {
        "model_card_path": "model_card",
        "readme_file": "project_readme",
        "license_file": "license",
        "evidence_manifest": "evidence_manifest",
    }
    artifacts: list[PackageArtifact] = []
    for key, path in paths.items():
        if path is None:
            continue
        label = label_map.get(key, key)
        artifacts.append(PackageArtifact(label, path))
    return artifacts


@app.command("field-report")
def field_report(
    output_html: Path = typer.Option(Path("runs/reports/field_report.html"), "--output", "-o"),
    run_manifest: Path | None = typer.Option(None, "--run-manifest"),
    mission_audit_report: Path | None = typer.Option(None, "--mission-audit-report"),
    mission_quality_report: Path | None = typer.Option(None, "--mission-quality-report"),
    prediction_quality_report: Path | None = typer.Option(None, "--prediction-quality-report"),
    flight_check_report: Path | None = typer.Option(None, "--flight-check-report"),
    flight_log_report: Path | None = typer.Option(None, "--flight-log-report"),
    capture_coverage_report: Path | None = typer.Option(None, "--capture-coverage-report"),
    domain_check_report: Path | None = typer.Option(None, "--domain-check-report"),
    geo_accuracy_report: Path | None = typer.Option(None, "--geo-accuracy-report"),
    validation_plan_report: Path | None = typer.Option(None, "--validation-plan-report"),
    truth_quality_report: Path | None = typer.Option(None, "--truth-quality-report"),
    truth_coverage_report: Path | None = typer.Option(None, "--truth-coverage-report"),
    stratified_truth_coverage_report: Path | None = typer.Option(None, "--stratified-truth-coverage-report"),
    acceptance_report: Path | None = typer.Option(None, "--acceptance-report"),
    stratified_acceptance_report: Path | None = typer.Option(None, "--stratified-acceptance-report"),
    benchmark_report: Path | None = typer.Option(None, "--benchmark-report"),
    tuning_report: Path | None = typer.Option(None, "--tuning-report"),
    cluster_review_report: Path | None = typer.Option(None, "--cluster-review-report"),
    release_audit_report: Path | None = typer.Option(None, "--release-audit-report"),
    model_manifest: Path | None = typer.Option(None, "--model-manifest"),
    title: str = typer.Option("BananaVision Field Report", "--title"),
) -> None:
    path = build_field_report(
        output_html,
        run_manifest=run_manifest,
        mission_audit_report=mission_audit_report,
        mission_quality_report=mission_quality_report,
        prediction_quality_report=prediction_quality_report,
        flight_check_report=flight_check_report,
        flight_log_report=flight_log_report,
        capture_coverage_report=capture_coverage_report,
        domain_check_report=domain_check_report,
        geo_accuracy_report=geo_accuracy_report,
        validation_plan_report=validation_plan_report,
        truth_quality_report=truth_quality_report,
        truth_coverage_report=truth_coverage_report,
        stratified_truth_coverage_report=stratified_truth_coverage_report,
        acceptance_report=acceptance_report,
        stratified_acceptance_report=stratified_acceptance_report,
        benchmark_report=benchmark_report,
        tuning_report=tuning_report,
        cluster_review_report=cluster_review_report,
        release_audit_report=release_audit_report,
        model_manifest=model_manifest,
        title=title,
    )
    console.print(f"[green]Field report written:[/green] {path}")


@app.command("evidence-manifest")
def evidence_manifest(
    output_json: Path = typer.Option(Path("runs/evidence/evidence_manifest.json"), "--output", "-o"),
    run_manifest: Path | None = typer.Option(None, "--run-manifest"),
    preflight_report: Path | None = typer.Option(None, "--preflight-report"),
    mission_audit_report: Path | None = typer.Option(None, "--mission-audit-report"),
    mission_quality_report: Path | None = typer.Option(None, "--mission-quality-report"),
    prediction_quality_report: Path | None = typer.Option(None, "--prediction-quality-report"),
    flight_check_report: Path | None = typer.Option(None, "--flight-check-report"),
    flight_log_report: Path | None = typer.Option(None, "--flight-log-report"),
    capture_coverage_report: Path | None = typer.Option(None, "--capture-coverage-report"),
    domain_check_report: Path | None = typer.Option(None, "--domain-check-report"),
    geo_accuracy_report: Path | None = typer.Option(None, "--geo-accuracy-report"),
    validation_plan_report: Path | None = typer.Option(None, "--validation-plan-report"),
    truth_quality_report: Path | None = typer.Option(None, "--truth-quality-report"),
    truth_coverage_report: Path | None = typer.Option(None, "--truth-coverage-report"),
    stratified_truth_coverage_report: Path | None = typer.Option(None, "--stratified-truth-coverage-report"),
    acceptance_report: Path | None = typer.Option(None, "--acceptance-report"),
    stratified_acceptance_report: Path | None = typer.Option(None, "--stratified-acceptance-report"),
    benchmark_report: Path | None = typer.Option(None, "--benchmark-report"),
    tuning_report: Path | None = typer.Option(None, "--tuning-report"),
    cluster_review_report: Path | None = typer.Option(None, "--cluster-review-report"),
    model_manifest: Path | None = typer.Option(None, "--model-manifest"),
    model_card: Path | None = typer.Option(None, "--model-card"),
    field_report_path: Path | None = typer.Option(None, "--field-report"),
    release_audit_report: Path | None = typer.Option(None, "--release-audit-report"),
    release_package_manifest: Path | None = typer.Option(None, "--release-package-manifest"),
    deployment_manifest: Path | None = typer.Option(None, "--deployment-manifest"),
    deployment_smoke_report: Path | None = typer.Option(None, "--deployment-smoke-report"),
    deployment_audit_report: Path | None = typer.Option(None, "--deployment-audit-report"),
    drone_ready_report: Path | None = typer.Option(None, "--drone-ready-report"),
    model: Path | None = typer.Option(None, "--model"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    required_label: list[str] = typer.Option([], "--require", help="Evidence label that must exist and pass if it reports status."),
) -> None:
    report = build_evidence_manifest(
        output_json,
        {
            "run_manifest": run_manifest,
            "preflight_report": preflight_report,
            "mission_audit_report": mission_audit_report,
            "mission_quality_report": mission_quality_report,
            "prediction_quality_report": prediction_quality_report,
            "flight_check_report": flight_check_report,
            "flight_log_report": flight_log_report,
            "capture_coverage_report": capture_coverage_report,
            "domain_check_report": domain_check_report,
            "geo_accuracy_report": geo_accuracy_report,
            "validation_plan_report": validation_plan_report,
            "truth_quality_report": truth_quality_report,
            "truth_coverage_report": truth_coverage_report,
            "stratified_truth_coverage_report": stratified_truth_coverage_report,
            "acceptance_report": acceptance_report,
            "stratified_acceptance_report": stratified_acceptance_report,
            "benchmark_report": benchmark_report,
            "tuning_report": tuning_report,
            "cluster_review_report": cluster_review_report,
            "model_manifest": model_manifest,
            "model_card": model_card,
            "field_report": field_report_path,
            "release_audit_report": release_audit_report,
            "release_package_manifest": release_package_manifest,
            "deployment_manifest": deployment_manifest,
            "deployment_smoke_report": deployment_smoke_report,
            "deployment_audit_report": deployment_audit_report,
            "drone_ready_report": drone_ready_report,
            "model": model,
            "config": config,
        },
        required_labels=required_label,
    )
    color = "green" if report["status"] == "pass" else "yellow"
    if report["status"] == "fail":
        color = "red"
    console.print(
        f"[{color}]Evidence manifest {str(report['status']).upper()}:[/{color}] "
        f"{report['present_count']}/{report['artifact_count']} present, "
        f"{report['missing_required_count']} required missing"
    )
    console.print(f"Report: {output_json}")
    if report["status"] == "fail":
        raise typer.Exit(code=2)


@app.command()
def synthetic(
    output_image: Path = typer.Option(Path("examples/synthetic_banana_scene.jpg"), "--image"),
    output_truth: Path = typer.Option(Path("examples/synthetic_banana_scene.truth.json"), "--truth"),
    width: int = 960,
    height: int = 640,
    plants: int = 36,
    seed: int = 7,
    clustered_mats: int = typer.Option(0, "--clustered-mats"),
    min_plants_per_mat: int = typer.Option(2, "--min-plants-per-mat"),
    max_plants_per_mat: int = typer.Option(3, "--max-plants-per-mat"),
    cluster_spread_px: float = typer.Option(28.0, "--cluster-spread-px"),
) -> None:
    image_path, truth_path = generate_scene(
        output_image,
        output_truth,
        width,
        height,
        plants,
        seed,
        clustered_mats=clustered_mats,
        min_plants_per_mat=min_plants_per_mat,
        max_plants_per_mat=max_plants_per_mat,
        cluster_spread_px=cluster_spread_px,
    )
    console.print(f"Image: {image_path}")
    console.print(f"Truth: {truth_path}")
