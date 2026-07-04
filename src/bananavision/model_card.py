from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime import utc_now_iso

DEFAULT_LIMITATIONS = [
    "Dense banana mats with severe leaf overlap can still need human review.",
    "Severe shadows, motion blur, low overlap, or off-domain flight altitude can reduce reliability.",
    "Off-domain camera, lighting, processing, season, or crop-state shifts require domain-check review.",
    "EXIF GPS geotags are capture-level evidence, not plant-level coordinates; plant coordinates require a passing geo-accuracy report.",
    "Do not claim a 1% error rate without a locked, representative, annotated holdout set.",
]


def build_model_card(
    output_md: str | Path,
    model_name: str,
    version: str,
    architecture: str = "YOLO segmentation or BananaVision RGB baseline",
    model_manifest: str | Path | None = None,
    acceptance_report: str | Path | None = None,
    benchmark_report: str | Path | None = None,
    mission_quality_report: str | Path | None = None,
    prediction_quality_report: str | Path | None = None,
    flight_log_report: str | Path | None = None,
    domain_check_report: str | Path | None = None,
    geo_accuracy_report: str | Path | None = None,
    dataset_quality_report: str | Path | None = None,
    validation_plan_report: str | Path | None = None,
    stratified_acceptance_report: str | Path | None = None,
    truth_quality_report: str | Path | None = None,
    truth_coverage_report: str | Path | None = None,
    stratified_truth_coverage_report: str | Path | None = None,
    intended_use: str = "Banana plant instance detection and counting from UAV imagery inside the validated operating domain.",
    notes: str = "",
) -> Path:
    model = _load_json(model_manifest)
    acceptance = _load_json(acceptance_report) or (model or {}).get("acceptance_report")
    benchmark = _load_json(benchmark_report) or (model or {}).get("benchmark_report")
    mission_quality = _load_json(mission_quality_report)
    prediction_quality = _load_json(prediction_quality_report)
    flight_log = _load_json(flight_log_report)
    domain_check = _load_json(domain_check_report)
    geo_accuracy = _load_json(geo_accuracy_report)
    dataset_quality = _load_json(dataset_quality_report)
    validation_plan = _load_json(validation_plan_report)
    stratified_acceptance = _load_json(stratified_acceptance_report)
    truth_quality = _load_json(truth_quality_report)
    truth_coverage = _load_json(truth_coverage_report)
    stratified_truth_coverage = _load_json(stratified_truth_coverage_report)
    claim = _claim_assessment(acceptance, validation_plan, stratified_acceptance, stratified_truth_coverage)
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"# Model Card: {model_name}",
            "",
            f"- Generated at: {utc_now_iso()}",
            f"- Version: {version}",
            f"- Architecture: {architecture}",
            f"- Model SHA256: {_model_sha(model)}",
            f"- Manifest SHA256: {_manifest_sha(model)}",
            "",
            "## Intended Use",
            "",
            intended_use,
            "",
            "## Claim Status",
            "",
            f"- Status: **{claim['status']}**",
            f"- Detail: {claim['detail']}",
            "",
            "## Validation Evidence",
            "",
            *_validation_plan_lines(validation_plan),
            *_acceptance_lines(acceptance),
            *_stratified_acceptance_lines(stratified_acceptance),
            "",
            "## Operational QA Evidence",
            "",
            *_quality_lines("Mission quality", mission_quality),
            *_quality_lines("Prediction quality", prediction_quality),
            *_flight_log_lines(flight_log),
            *_domain_lines(domain_check),
            *_geo_accuracy_lines(geo_accuracy),
            *_quality_lines("Dataset quality", dataset_quality),
            *_truth_quality_lines(truth_quality),
            *_truth_coverage_lines(truth_coverage),
            *_stratified_truth_coverage_lines(stratified_truth_coverage),
            "",
            "## Edge Performance",
            "",
            *_benchmark_lines(benchmark),
            "",
            "## Known Limitations",
            "",
            *[f"- {item}" for item in DEFAULT_LIMITATIONS],
            "",
            "## Notes",
            "",
            notes or "No additional notes.",
            "",
        ]
    )
    output_md.write_text(content, encoding="utf-8")
    return output_md


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model card artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_assessment(
    acceptance: dict[str, Any] | None,
    validation_plan: dict[str, Any] | None,
    stratified_acceptance: dict[str, Any] | None,
    stratified_truth_coverage: dict[str, Any] | None,
) -> dict[str, str]:
    if acceptance is None:
        return {"status": "unproven", "detail": "No acceptance report was provided."}
    if acceptance.get("passed") is not True:
        return {"status": "not accepted", "detail": "Acceptance report did not pass."}
    metrics = acceptance.get("metrics", {}) or {}
    statistics = acceptance.get("statistics", {}) or {}
    sample_support = statistics.get("sample_support", {}) or {}
    count_error_rate = float(metrics.get("count_error_rate", 1.0))
    min_detectable = float(sample_support.get("min_detectable_count_error_rate", 1.0))
    if count_error_rate <= 0.01 and min_detectable <= 0.01:
        if validation_plan is None:
            return {
                "status": "accepted but 1% validation plan not provided",
                "detail": "Acceptance metrics are inside 1%, but no validation-plan report was provided.",
            }
        if not _validation_plan_supports_acceptance(validation_plan, acceptance):
            return {
                "status": "accepted but validation plan support not met",
                "detail": "Acceptance metrics are inside 1%, but the provided evidence does not satisfy the validation plan.",
            }
        if stratified_truth_coverage is None:
            return {
                "status": "accepted but stratified truth coverage not provided",
                "detail": (
                    "Global acceptance metrics are inside 1%, but no stratified truth-coverage report was provided "
                    "for per-condition validation support."
                ),
            }
        if not _stratified_truth_coverage_supports_claim(stratified_truth_coverage):
            return {
                "status": "accepted but stratified truth coverage support not met",
                "detail": (
                    "Global acceptance metrics are inside 1%, but the field truth does not pass per-condition "
                    "coverage gates or has missing metadata."
                ),
            }
        if stratified_acceptance is None:
            return {
                "status": "accepted but stratified acceptance not provided",
                "detail": (
                    "Global acceptance metrics are inside 1%, but no stratified acceptance report was provided "
                    "for farm/date/GSD/cultivar coverage."
                ),
            }
        if not _stratified_acceptance_supports_claim(stratified_acceptance):
            return {
                "status": "accepted but stratified acceptance support not met",
                "detail": (
                    "Global acceptance metrics are inside 1%, but one or more field-condition strata failed, "
                    "are missing metadata, or were not checked at a 1% count-error threshold."
                ),
            }
        return {
            "status": "1% claim supported by provided reports",
            "detail": (
                "Count error, sample support, validation-plan support, and stratified field-condition "
                "acceptance are inside the 1% target; verify domain coverage before publication."
            ),
        }
    return {
        "status": "accepted but 1% claim not proven",
        "detail": (
            f"Acceptance passed, but count_error_rate={count_error_rate:.4f} "
            f"and min_detectable_count_error_rate={min_detectable:.4f} do not prove a 1% claim."
        ),
    }


def _acceptance_lines(acceptance: dict[str, Any] | None) -> list[str]:
    if acceptance is None:
        return ["- Acceptance report: not provided."]
    metrics = acceptance.get("metrics", {}) or {}
    statistics = acceptance.get("statistics", {}) or {}
    support = statistics.get("sample_support", {}) or {}
    precision_ci = statistics.get("precision_wilson_ci", {}) or {}
    recall_ci = statistics.get("recall_wilson_ci", {}) or {}
    return [
        f"- Passed: {acceptance.get('passed')}",
        f"- Images: {metrics.get('images', support.get('images', 'unknown'))}",
        f"- Truth count: {metrics.get('truth_count', support.get('truth_count', 'unknown'))}",
        f"- Prediction count: {metrics.get('prediction_count', support.get('prediction_count', 'unknown'))}",
        f"- Count error rate: {_fmt(metrics.get('count_error_rate'))}",
        f"- Precision: {_fmt(metrics.get('precision'))}",
        f"- Recall: {_fmt(metrics.get('recall'))}",
        f"- F1: {_fmt(metrics.get('f1'))}",
        f"- Annotated banana clusters: {metrics.get('cluster_count', support.get('cluster_count', 0))}",
        f"- Cluster truth plants: {metrics.get('cluster_truth_count', support.get('cluster_truth_count', 0))}",
        f"- Cluster recall: {_fmt(metrics.get('cluster_recall'))}",
        f"- Fully detected cluster rate: {_fmt(metrics.get('fully_detected_cluster_rate'))}",
        f"- Precision Wilson CI: {_ci(precision_ci)}",
        f"- Recall Wilson CI: {_ci(recall_ci)}",
        f"- Minimum detectable count error rate: {_fmt(support.get('min_detectable_count_error_rate'))}",
    ]


def _validation_plan_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["- Validation plan: not provided."]
    targets = report.get("targets", {}) or {}
    support = report.get("minimum_support", {}) or {}
    return [
        "- Validation plan: "
        f"status={report.get('status')}, "
        f"target_count_error_rate={_fmt(targets.get('target_count_error_rate'))}, "
        f"minimum_truth_count={support.get('truth_count')}, "
        f"minimum_cluster_count={support.get('cluster_count')}, "
        f"minimum_cluster_truth_count={support.get('cluster_truth_count')}, "
        f"minimum_cluster_image_count={support.get('cluster_image_count')}."
    ]


def _stratified_acceptance_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["- Stratified acceptance: not provided."]
    thresholds = report.get("thresholds", {}) or {}
    failed = _failed_strata_summary(report)
    return [
        "- Stratified acceptance: "
        f"status={report.get('status')}, "
        f"stratum_count={report.get('stratum_count')}, "
        f"failed_stratum_count={report.get('failed_stratum_count')}, "
        f"missing_metadata_count={report.get('missing_metadata_count')}, "
        f"max_count_error_rate={_fmt(thresholds.get('max_count_error_rate'))}, "
        f"failed_strata={failed}."
    ]


def _stratified_truth_coverage_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["- Stratified truth coverage: not provided."]
    thresholds = report.get("thresholds", {}) or {}
    failed = _failed_strata_summary(report)
    return [
        "- Stratified truth coverage: "
        f"status={report.get('status')}, "
        f"stratum_count={report.get('stratum_count')}, "
        f"failed_stratum_count={report.get('failed_stratum_count')}, "
        f"missing_metadata_count={report.get('missing_metadata_count')}, "
        f"min_truth_count={_fmt(thresholds.get('min_truth_count'))}, "
        f"failed_strata={failed}."
    ]


def _failed_strata_summary(report: dict[str, Any]) -> str:
    failed = [row.get("stratum", {}) for row in report.get("strata", []) if not _stratum_row_passed(row)]
    if not failed:
        return "none"
    labels = []
    for stratum in failed[:3]:
        labels.append("/".join(str(value) for value in stratum.values()))
    suffix = "" if len(failed) <= 3 else f" (+{len(failed) - 3} more)"
    return ", ".join(labels) + suffix


def _stratum_row_passed(row: dict[str, Any]) -> bool:
    return row.get("passed") is True or row.get("status") == "pass"


def _quality_lines(label: str, report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return [f"- {label}: not provided."]
    parts = [f"status={report.get('status')}"]
    for key in ["image_count", "pass_count", "warn_count", "fail_count", "detection_count", "review_detection_count"]:
        if key in report:
            parts.append(f"{key}={report[key]}")
    return [f"- {label}: " + ", ".join(parts) + "."]


def _truth_coverage_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["- Truth coverage: not provided."]
    return [
        "- Truth coverage: "
        f"status={report.get('status')}, "
        f"truth_count={report.get('truth_count')}, "
        f"cluster_count={report.get('cluster_count')}, "
        f"cluster_truth_count={report.get('cluster_truth_count')}, "
        f"cluster_image_count={report.get('cluster_image_count')}, "
        f"cluster_truth_fraction={_fmt(report.get('cluster_truth_fraction'))}."
    ]


def _truth_quality_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["- Truth quality: not provided."]
    return [
        "- Truth quality: "
        f"status={report.get('status')}, "
        f"image_count={report.get('image_count')}, "
        f"truth_count={report.get('truth_count')}, "
        f"bounded_image_count={report.get('bounded_image_count')}, "
        f"issue_count={report.get('issue_count')}."
    ]


def _domain_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["- Domain check: not provided."]
    return [
        "- Domain check: "
        f"status={report.get('status')}, "
        f"image_count={report.get('image_count')}, "
        f"reference_image_count={report.get('reference_image_count')}, "
        f"outlier_count={report.get('outlier_count')}, "
        f"outlier_fraction={_fmt(report.get('outlier_fraction'))}."
    ]


def _flight_log_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["- Flight log audit: not provided."]
    summary = report.get("summary", {}) or {}
    return [
        "- Flight log audit: "
        f"status={report.get('status')}, "
        f"row_count={summary.get('row_count')}, "
        f"pass_count={summary.get('pass_count')}, "
        f"warn_count={summary.get('warn_count')}, "
        f"fail_count={summary.get('fail_count')}, "
        f"max_gsd_cm={_fmt(summary.get('max_gsd_cm'))}."
    ]


def _geo_accuracy_lines(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["- Geo accuracy: not provided."]
    metrics = report.get("metrics", {}) or {}
    return [
        "- Geo accuracy: "
        f"status={report.get('status')}, "
        f"matched={metrics.get('matched_count')}/{metrics.get('truth_count')}, "
        f"rmse_m={_fmt(metrics.get('rmse_m'))}, "
        f"p95_m={_fmt(metrics.get('p95_m'))}, "
        f"recall={_fmt(metrics.get('recall'))}."
    ]


def _benchmark_lines(benchmark: dict[str, Any] | None) -> list[str]:
    if benchmark is None:
        return ["- Benchmark report: not provided."]
    latency = benchmark.get("latency_ms", {}) or {}
    return [
        f"- Median latency ms: {_fmt(latency.get('median'))}",
        f"- P95 latency ms: {_fmt(latency.get('p95'))}",
        f"- Max latency ms: {_fmt(latency.get('max'))}",
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return str(value)


def _ci(payload: dict[str, Any]) -> str:
    if not payload:
        return "unknown"
    return f"{_fmt(payload.get('lower'))} - {_fmt(payload.get('upper'))}"


def _validation_plan_supports_acceptance(validation_plan: dict[str, Any], acceptance: dict[str, Any]) -> bool:
    if validation_plan.get("status") != "plan":
        return False
    targets = validation_plan.get("targets", {}) or {}
    if float(targets.get("target_count_error_rate", 1.0)) > 0.01:
        return False
    support = validation_plan.get("minimum_support", {}) or {}
    metrics = acceptance.get("metrics", {}) or {}
    for key in ["truth_count", "cluster_count", "cluster_truth_count"]:
        required = support.get(key)
        if required is None:
            continue
        if int(metrics.get(key, 0)) < int(required):
            return False
    return True


def _stratified_acceptance_supports_claim(report: dict[str, Any]) -> bool:
    thresholds = report.get("thresholds", {}) or {}
    try:
        max_count_error_rate = float(thresholds.get("max_count_error_rate", 1.0))
    except (TypeError, ValueError):
        return False
    return (
        report.get("status") == "pass"
        and int(report.get("stratum_count", 0) or 0) > 0
        and int(report.get("failed_stratum_count", 1) or 0) == 0
        and int(report.get("missing_metadata_count", 1) or 0) == 0
        and max_count_error_rate <= 0.01
    )


def _stratified_truth_coverage_supports_claim(report: dict[str, Any]) -> bool:
    return (
        report.get("status") == "pass"
        and int(report.get("stratum_count", 0) or 0) > 0
        and int(report.get("failed_stratum_count", 1) or 0) == 0
        and int(report.get("missing_metadata_count", 1) or 0) == 0
    )


def _model_sha(model: dict[str, Any] | None) -> str:
    return str((model or {}).get("model_sha256") or "unknown")


def _manifest_sha(model: dict[str, Any] | None) -> str:
    return str((model or {}).get("manifest_sha256") or "unknown")
