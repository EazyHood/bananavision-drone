from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def build_field_report(
    output_html: str | Path,
    run_manifest: str | Path | None = None,
    mission_audit_report: str | Path | None = None,
    mission_quality_report: str | Path | None = None,
    prediction_quality_report: str | Path | None = None,
    flight_check_report: str | Path | None = None,
    flight_log_report: str | Path | None = None,
    capture_coverage_report: str | Path | None = None,
    domain_check_report: str | Path | None = None,
    geo_accuracy_report: str | Path | None = None,
    validation_plan_report: str | Path | None = None,
    truth_quality_report: str | Path | None = None,
    truth_coverage_report: str | Path | None = None,
    stratified_truth_coverage_report: str | Path | None = None,
    acceptance_report: str | Path | None = None,
    stratified_acceptance_report: str | Path | None = None,
    benchmark_report: str | Path | None = None,
    tuning_report: str | Path | None = None,
    cluster_review_report: str | Path | None = None,
    release_audit_report: str | Path | None = None,
    model_manifest: str | Path | None = None,
    title: str = "BananaVision Field Report",
) -> Path:
    sections = [
        _section("Run Manifest", _load_json(run_manifest)),
        _section("Mission Audit", _load_json(mission_audit_report)),
        _section("Mission Quality", _load_json(mission_quality_report)),
        _section("Prediction Quality", _load_json(prediction_quality_report)),
        _section("Flight Check", _load_json(flight_check_report)),
        _section("Flight Log Audit", _load_json(flight_log_report)),
        _section("Capture Coverage", _load_json(capture_coverage_report)),
        _section("Domain Check", _load_json(domain_check_report)),
        _section("Geo Accuracy", _load_json(geo_accuracy_report)),
        _section("Validation Plan", _load_json(validation_plan_report)),
        _section("Truth Quality", _load_json(truth_quality_report)),
        _section("Truth Coverage", _load_json(truth_coverage_report)),
        _section("Stratified Truth Coverage", _load_json(stratified_truth_coverage_report)),
        _section("Acceptance", _load_json(acceptance_report)),
        _section("Stratified Acceptance", _load_json(stratified_acceptance_report)),
        _section("Benchmark", _load_json(benchmark_report)),
        _section("Tuning", _load_json(tuning_report)),
        _section("Cluster Review", _load_json(cluster_review_report)),
        _section("Release Audit", _load_json(release_audit_report)),
        _section("Model Manifest", _load_json(model_manifest)),
    ]
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #17211b; background: #f7f9f6; }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ font-size: 32px; margin-bottom: 4px; }}
    h2 {{ font-size: 20px; margin-top: 28px; border-bottom: 1px solid #cfd8ce; padding-bottom: 8px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ background: #ffffff; border: 1px solid #d9e2d7; border-radius: 6px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 12px; color: #536255; text-transform: uppercase; }}
    .metric span {{ font-size: 24px; }}
    pre {{ background: #101811; color: #e8f3e6; padding: 16px; border-radius: 6px; overflow: auto; }}
    code {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p>Static audit report generated from BananaVision artifacts.</p>
  {''.join(sections)}
</main>
</body>
</html>
"""
    output_html.write_text(document, encoding="utf-8")
    return output_html


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Report artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _section(title: str, payload: dict[str, Any] | None) -> str:
    if payload is None:
        return ""
    metrics = _metrics_for(title, payload)
    metric_html = ""
    if metrics:
        metric_html = "<div class=\"summary\">" + "".join(
            f"<div class=\"metric\"><strong>{html.escape(label)}</strong><span>{html.escape(value)}</span></div>"
            for label, value in metrics
        ) + "</div>"
    return f"<section><h2>{html.escape(title)}</h2>{metric_html}<pre><code>{html.escape(json.dumps(payload, indent=2))}</code></pre></section>"


def _metrics_for(title: str, payload: dict[str, Any]) -> list[tuple[str, str]]:
    if title == "Run Manifest":
        return [
            ("Images", str(payload.get("image_count", ""))),
            ("Detections", str(payload.get("total_detections", ""))),
            ("Elapsed ms", str(payload.get("elapsed_ms", ""))),
        ]
    if title == "Mission Audit":
        summary = payload.get("summary", {}) or {}
        return [
            ("Status", str(payload.get("status", ""))),
            ("Images", str(summary.get("image_count", ""))),
            ("Detections", str(summary.get("total_detections", ""))),
            ("Mission QA", str(summary.get("mission_quality_status", ""))),
            ("Prediction QA", str(summary.get("prediction_quality_status", ""))),
            ("Capture coverage", str(summary.get("capture_coverage_status", ""))),
        ]
    if title == "Acceptance":
        metrics = payload.get("metrics", {})
        items = [
            ("Passed", str(payload.get("passed", ""))),
            ("Count error", f"{float(metrics.get('count_error_rate', 0.0)):.4f}"),
            ("Precision", f"{float(metrics.get('precision', 0.0)):.4f}"),
            ("Recall", f"{float(metrics.get('recall', 0.0)):.4f}"),
            ("Clusters", str(metrics.get("cluster_count", 0))),
            ("Cluster recall", f"{float(metrics.get('cluster_recall', 0.0)):.4f}"),
            ("Cluster full rate", f"{float(metrics.get('fully_detected_cluster_rate', 0.0)):.4f}"),
        ]
        statistics = payload.get("statistics", {}) or {}
        precision_ci = statistics.get("precision_wilson_ci", {}) or {}
        recall_ci = statistics.get("recall_wilson_ci", {}) or {}
        if precision_ci:
            items.append(("Precision CI low", f"{float(precision_ci.get('lower', 0.0)):.4f}"))
        if recall_ci:
            items.append(("Recall CI low", f"{float(recall_ci.get('lower', 0.0)):.4f}"))
        return items
    if title == "Mission Quality":
        return [
            ("Status", str(payload.get("status", ""))),
            ("Images", str(payload.get("image_count", ""))),
            ("Pass", str(payload.get("pass_count", ""))),
            ("Warn", str(payload.get("warn_count", ""))),
            ("Fail", str(payload.get("fail_count", ""))),
        ]
    if title == "Prediction Quality":
        return [
            ("Status", str(payload.get("status", ""))),
            ("Detections", str(payload.get("detection_count", ""))),
            ("Review", str(payload.get("review_detection_count", ""))),
            ("Review frac", f"{float(payload.get('review_fraction', 0.0)):.4f}"),
        ]
    if title == "Flight Check":
        profile = payload.get("profile", {}) or {}
        return [
            ("Status", str(payload.get("status", ""))),
            ("GSD cm/px", f"{float(profile.get('observed_gsd_cm', 0.0)):.3f}"),
            ("Front overlap", str(profile.get("front_overlap", ""))),
            ("Side overlap", str(profile.get("side_overlap", ""))),
        ]
    if title == "Flight Log Audit":
        summary = payload.get("summary", {}) or {}
        return [
            ("Status", str(payload.get("status", ""))),
            ("Rows", str(summary.get("row_count", ""))),
            ("Pass", str(summary.get("pass_count", ""))),
            ("Warn", str(summary.get("warn_count", ""))),
            ("Fail", str(summary.get("fail_count", ""))),
            ("Max GSD cm", f"{float(summary.get('max_gsd_cm', 0.0) or 0.0):.3f}"),
        ]
    if title == "Capture Coverage":
        summary = payload.get("summary", {}) or {}
        return [
            ("Status", str(payload.get("status", ""))),
            ("Rows", str(summary.get("row_count", ""))),
            ("Missing images", str(summary.get("missing_image_count", ""))),
            ("Positions", str(summary.get("position_count", ""))),
            ("Max step m", f"{float(summary.get('max_step_distance_m', 0.0) or 0.0):.3f}"),
            ("Duplicates", str(summary.get("duplicate_position_count", ""))),
        ]
    if title == "Domain Check":
        return [
            ("Status", str(payload.get("status", ""))),
            ("Images", str(payload.get("image_count", ""))),
            ("Outliers", str(payload.get("outlier_count", ""))),
            ("Outlier frac", f"{float(payload.get('outlier_fraction', 0.0)):.4f}"),
        ]
    if title == "Geo Accuracy":
        metrics = payload.get("metrics", {}) or {}
        return [
            ("Status", str(payload.get("status", ""))),
            ("Matched", f"{metrics.get('matched_count', '')}/{metrics.get('truth_count', '')}"),
            ("RMSE m", f"{float(metrics.get('rmse_m', 0.0)):.4f}"),
            ("P95 m", f"{float(metrics.get('p95_m', 0.0)):.4f}"),
            ("Recall", f"{float(metrics.get('recall', 0.0)):.4f}"),
        ]
    if title == "Validation Plan":
        support = payload.get("minimum_support", {}) or {}
        targets = payload.get("targets", {}) or {}
        return [
            ("Target count error", f"{float(targets.get('target_count_error_rate', 0.0) or 0.0):.4f}"),
            ("Truth minimum", str(support.get("truth_count", ""))),
            ("Cluster minimum", str(support.get("cluster_count", ""))),
            ("Cluster truth min", str(support.get("cluster_truth_count", ""))),
            ("Cluster images min", str(support.get("cluster_image_count", ""))),
        ]
    if title == "Truth Quality":
        return [
            ("Status", str(payload.get("status", ""))),
            ("Images", str(payload.get("image_count", ""))),
            ("Truth", str(payload.get("truth_count", ""))),
            ("Bounded images", str(payload.get("bounded_image_count", ""))),
            ("Issues", str(payload.get("issue_count", ""))),
        ]
    if title == "Truth Coverage":
        return [
            ("Status", str(payload.get("status", ""))),
            ("Truth", str(payload.get("truth_count", ""))),
            ("Clusters", str(payload.get("cluster_count", ""))),
            ("Cluster truth", str(payload.get("cluster_truth_count", ""))),
            ("Cluster images", str(payload.get("cluster_image_count", ""))),
            ("Cluster frac", f"{float(payload.get('cluster_truth_fraction', 0.0) or 0.0):.4f}"),
        ]
    if title == "Benchmark":
        latency = payload.get("latency_ms", {})
        return [
            ("Median ms", str(latency.get("median", ""))),
            ("P95 ms", str(latency.get("p95", ""))),
            ("Max ms", str(latency.get("max", ""))),
        ]
    if title == "Tuning":
        best = payload.get("best", {}) or {}
        metrics = best.get("metrics", {}) or {}
        return [
            ("Best count", str(best.get("count", ""))),
            ("Count error", f"{float(metrics.get('count_error_rate', 0.0)):.4f}"),
            ("F1", f"{float(metrics.get('f1', 0.0)):.4f}"),
        ]
    if title == "Release Audit":
        gates = payload.get("gates", []) or []
        return [
            ("Status", str(payload.get("status", ""))),
            ("Gates", str(len(gates))),
            ("Failed gates", str(sum(1 for gate in gates if gate.get("status") == "fail"))),
            ("Warn gates", str(sum(1 for gate in gates if gate.get("status") == "warn"))),
        ]
    if title == "Model Manifest":
        return [
            ("Version", str(payload.get("version", ""))),
            ("Model SHA", str(payload.get("model_sha256", ""))[:12]),
            ("Manifest SHA", str(payload.get("manifest_sha256", ""))[:12]),
        ]
    return []
