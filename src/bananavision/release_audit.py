from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime import utc_now_iso


def audit_release(
    output_json: str | Path,
    acceptance_report: str | Path | None = None,
    stratified_acceptance_report: str | Path | None = None,
    benchmark_report: str | Path | None = None,
    mission_quality_report: str | Path | None = None,
    prediction_quality_report: str | Path | None = None,
    holdout_verify_report: str | Path | None = None,
    validation_plan_report: str | Path | None = None,
    truth_quality_report: str | Path | None = None,
    truth_coverage_report: str | Path | None = None,
    stratified_truth_coverage_report: str | Path | None = None,
    flight_check_report: str | Path | None = None,
    flight_log_report: str | Path | None = None,
    domain_check_report: str | Path | None = None,
    geo_accuracy_report: str | Path | None = None,
    model_manifest: str | Path | None = None,
    model_card: str | Path | None = None,
    field_report: str | Path | None = None,
    max_count_error_rate: float = 0.01,
    min_truth_count: int | None = None,
    min_precision_ci_lower: float | None = None,
    min_recall_ci_lower: float | None = None,
    min_cluster_recall: float | None = None,
    min_cluster_full_detection_rate: float | None = None,
    min_cluster_count: int | None = None,
    min_cluster_truth_count: int | None = None,
    min_cluster_images: int | None = None,
    min_cluster_truth_fraction: float | None = None,
    max_p95_ms: float | None = None,
    max_geo_rmse_m: float = 1.0,
    max_geo_p95_m: float | None = None,
    min_geo_recall: float = 0.99,
    allow_warn_quality: bool = False,
) -> dict[str, Any]:
    gates: list[dict[str, str]] = []
    acceptance = _load_json_gate("acceptance_report_present", acceptance_report, gates, required=True)
    stratified_acceptance = _load_json_gate(
        "stratified_acceptance_report_present",
        stratified_acceptance_report,
        gates,
        required=_requires_validation_plan(max_count_error_rate),
    )
    benchmark = _load_json_gate("benchmark_report_present", benchmark_report, gates, required=True)
    mission_quality = _load_json_gate("mission_quality_report_present", mission_quality_report, gates)
    prediction_quality = _load_json_gate("prediction_quality_report_present", prediction_quality_report, gates)
    holdout_verify = _load_json_gate("holdout_verify_report_present", holdout_verify_report, gates, required=True)
    validation_plan = _load_json_gate(
        "validation_plan_report_present",
        validation_plan_report,
        gates,
        required=_requires_validation_plan(max_count_error_rate),
    )
    truth_quality = _load_json_gate(
        "truth_quality_report_present",
        truth_quality_report,
        gates,
        required=_requires_validation_plan(max_count_error_rate),
    )
    truth_coverage = _load_json_gate(
        "truth_coverage_report_present",
        truth_coverage_report,
        gates,
        required=_requires_truth_coverage(
            min_cluster_count,
            min_cluster_truth_count,
            min_cluster_images,
            min_cluster_truth_fraction,
        )
        or _requires_validation_plan(max_count_error_rate),
    )
    stratified_truth_coverage = _load_json_gate(
        "stratified_truth_coverage_report_present",
        stratified_truth_coverage_report,
        gates,
        required=_requires_validation_plan(max_count_error_rate),
    )
    flight_check = _load_json_gate("flight_check_report_present", flight_check_report, gates, required=True)
    flight_log = None
    if flight_log_report is not None:
        flight_log = _load_json_gate("flight_log_report_present", flight_log_report, gates, required=True)
    domain_check = _load_json_gate("domain_check_report_present", domain_check_report, gates, required=True)
    geo_accuracy = _load_json_gate("geo_accuracy_report_present", geo_accuracy_report, gates, required=True)
    model = _load_json_gate("model_manifest_present", model_manifest, gates)

    if acceptance is not None:
        _check_acceptance(
            acceptance,
            gates,
            max_count_error_rate=max_count_error_rate,
            min_truth_count=min_truth_count,
            min_precision_ci_lower=min_precision_ci_lower,
            min_recall_ci_lower=min_recall_ci_lower,
            min_cluster_recall=min_cluster_recall,
            min_cluster_full_detection_rate=min_cluster_full_detection_rate,
            min_cluster_count=min_cluster_count,
        )
    if stratified_acceptance is not None:
        _check_stratified_acceptance(stratified_acceptance, gates)
    if benchmark is not None:
        _check_benchmark(benchmark, gates, max_p95_ms=max_p95_ms)
    if mission_quality is not None:
        _check_quality("mission_quality", mission_quality, gates, allow_warn=allow_warn_quality)
    if prediction_quality is not None:
        _check_quality("prediction_quality", prediction_quality, gates, allow_warn=allow_warn_quality)
    if holdout_verify is not None:
        _check_holdout_verify(
            holdout_verify,
            gates,
            max_count_error_rate=max_count_error_rate,
            min_cluster_count=min_cluster_count,
        )
    if validation_plan is not None:
        _check_validation_plan(
            validation_plan,
            acceptance,
            truth_coverage,
            gates,
            max_count_error_rate=max_count_error_rate,
        )
    if truth_quality is not None:
        _check_truth_quality(truth_quality, gates)
    if truth_coverage is not None:
        _check_truth_coverage(
            truth_coverage,
            gates,
            min_truth_count=min_truth_count,
            min_cluster_count=min_cluster_count,
            min_cluster_truth_count=min_cluster_truth_count,
            min_cluster_images=min_cluster_images,
            min_cluster_truth_fraction=min_cluster_truth_fraction,
        )
    if stratified_truth_coverage is not None:
        _check_stratified_truth_coverage(stratified_truth_coverage, gates, validation_plan)
    if flight_check is not None:
        _check_flight_check(flight_check, gates)
    if flight_log is not None:
        _check_flight_log(flight_log, gates)
    if domain_check is not None:
        _check_domain_check(domain_check, gates)
    if geo_accuracy is not None:
        _check_geo_accuracy(
            geo_accuracy,
            gates,
            max_geo_rmse_m=max_geo_rmse_m,
            max_geo_p95_m=max_geo_p95_m,
            min_geo_recall=min_geo_recall,
        )
    if model is not None:
        _check_model_manifest(model, gates)
    _check_file("model_card", model_card, gates)
    _check_file("field_report", field_report, gates)

    status = _overall_status(gates)
    report = {
        "created_at": utc_now_iso(),
        "status": status,
        "thresholds": {
            "max_count_error_rate": max_count_error_rate,
            "stratified_acceptance_required": _requires_validation_plan(max_count_error_rate),
            "stratified_truth_coverage_required": _requires_validation_plan(max_count_error_rate),
            "min_truth_count": min_truth_count,
            "min_precision_ci_lower": min_precision_ci_lower,
            "min_recall_ci_lower": min_recall_ci_lower,
            "min_cluster_recall": min_cluster_recall,
            "min_cluster_full_detection_rate": min_cluster_full_detection_rate,
            "min_cluster_count": min_cluster_count,
            "min_cluster_truth_count": min_cluster_truth_count,
            "min_cluster_images": min_cluster_images,
            "min_cluster_truth_fraction": min_cluster_truth_fraction,
            "max_p95_ms": max_p95_ms,
            "max_geo_rmse_m": max_geo_rmse_m,
            "max_geo_p95_m": max_geo_p95_m,
            "min_geo_recall": min_geo_recall,
            "allow_warn_quality": allow_warn_quality,
            "validation_plan_required": _requires_validation_plan(max_count_error_rate),
            "truth_quality_required": _requires_validation_plan(max_count_error_rate),
        },
        "gates": gates,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _requires_truth_coverage(
    min_cluster_count: int | None,
    min_cluster_truth_count: int | None,
    min_cluster_images: int | None,
    min_cluster_truth_fraction: float | None,
) -> bool:
    return any(
        value is not None
        for value in [min_cluster_count, min_cluster_truth_count, min_cluster_images, min_cluster_truth_fraction]
    )


def _requires_validation_plan(max_count_error_rate: float) -> bool:
    return max_count_error_rate <= 0.01


def _load_json_gate(
    name: str,
    path: str | Path | None,
    gates: list[dict[str, str]],
    required: bool = False,
) -> dict[str, Any] | None:
    if path is None:
        gates.append(_gate(name, "fail" if required else "warn", "artifact path not provided"))
        return None
    path = Path(path)
    if not path.exists():
        gates.append(_gate(name, "fail" if required else "warn", f"artifact does not exist: {path}"))
        return None
    gates.append(_gate(name, "pass", str(path)))
    return json.loads(path.read_text(encoding="utf-8"))


def _check_validation_plan(
    validation_plan: dict[str, Any],
    acceptance: dict[str, Any] | None,
    truth_coverage: dict[str, Any] | None,
    gates: list[dict[str, str]],
    max_count_error_rate: float,
) -> None:
    status = str(validation_plan.get("status", "missing"))
    gates.append(
        _gate(
            "validation_plan_status",
            "pass" if status == "plan" else "fail",
            "validation plan present" if status == "plan" else f"validation plan status={status}",
        )
    )
    targets = validation_plan.get("targets", {}) or {}
    planned_error = _float_value(targets.get("target_count_error_rate"), 1.0)
    gates.append(
        _threshold_gate(
            "validation_plan_target",
            planned_error <= max_count_error_rate,
            f"{planned_error:.4f} <= {max_count_error_rate:.4f}",
        )
    )
    support = validation_plan.get("minimum_support", {}) or {}
    _check_support_payload(
        "validation_plan_acceptance",
        support,
        acceptance,
        gates,
        keys=["truth_count", "cluster_count", "cluster_truth_count"],
        metric_source="metrics",
    )
    _check_support_payload(
        "validation_plan_truth_coverage",
        support,
        truth_coverage,
        gates,
        keys=["truth_count", "cluster_count", "cluster_truth_count", "cluster_image_count"],
        metric_source=None,
    )


def _check_support_payload(
    prefix: str,
    support: dict[str, Any],
    evidence: dict[str, Any] | None,
    gates: list[dict[str, str]],
    keys: list[str],
    metric_source: str | None,
) -> None:
    if evidence is None:
        gates.append(_gate(f"{prefix}_present", "fail", "required evidence missing"))
        return
    payload = evidence.get(metric_source, {}) if metric_source else evidence
    payload = payload or {}
    for key in keys:
        required = support.get(key)
        if required is None:
            continue
        observed = _int_value(payload.get(key), 0)
        gates.append(
            _threshold_gate(
                f"{prefix}_{key}",
                observed >= int(required),
                f"{observed} >= {int(required)}",
            )
        )


def _check_truth_quality(report: dict[str, Any], gates: list[dict[str, str]]) -> None:
    status = str(report.get("status", "missing"))
    gates.append(
        _gate(
            "truth_quality_passed",
            "pass" if status == "pass" else "fail",
            "truth quality report passed" if status == "pass" else f"truth quality status={status}",
        )
    )
    issue_count = _int_value(report.get("issue_count"), 0)
    image_count = _int_value(report.get("image_count"), 0)
    gates.append(_threshold_gate("truth_quality_issues", issue_count == 0, f"issue_count={issue_count}"))
    gates.append(_threshold_gate("truth_quality_images", image_count > 0, f"image_count={image_count}"))


def _check_acceptance(
    acceptance: dict[str, Any],
    gates: list[dict[str, str]],
    max_count_error_rate: float,
    min_truth_count: int | None,
    min_precision_ci_lower: float | None,
    min_recall_ci_lower: float | None,
    min_cluster_recall: float | None,
    min_cluster_full_detection_rate: float | None,
    min_cluster_count: int | None,
) -> None:
    gates.append(
        _gate(
            "acceptance_passed",
            "pass" if acceptance.get("passed") is True else "fail",
            "acceptance report passed" if acceptance.get("passed") is True else "acceptance report did not pass",
        )
    )
    metrics = acceptance.get("metrics", {}) or {}
    statistics = acceptance.get("statistics", {}) or {}
    sample = statistics.get("sample_support", {}) or {}
    count_error_rate = float(metrics.get("count_error_rate", 1.0))
    gates.append(
        _threshold_gate(
            "count_error_rate",
            count_error_rate <= max_count_error_rate,
            f"{count_error_rate:.4f} <= {max_count_error_rate:.4f}",
        )
    )
    min_detectable = sample.get("min_detectable_count_error_rate")
    if min_detectable is None:
        gates.append(_gate("sample_support_present", "fail", "statistics.sample_support is missing"))
    else:
        min_detectable = float(min_detectable)
        gates.append(
            _threshold_gate(
                "sample_support_resolution",
                min_detectable <= max_count_error_rate,
                f"min_detectable_count_error_rate={min_detectable:.4f} <= {max_count_error_rate:.4f}",
            )
        )
    if min_truth_count is not None:
        truth_count = int(metrics.get("truth_count", sample.get("truth_count", 0)))
        gates.append(_threshold_gate("min_truth_count", truth_count >= min_truth_count, f"{truth_count} >= {min_truth_count}"))
    if min_cluster_count is not None:
        cluster_count = int(metrics.get("cluster_count", sample.get("cluster_count", 0)))
        gates.append(
            _threshold_gate(
                "min_cluster_count",
                cluster_count >= min_cluster_count,
                f"{cluster_count} >= {min_cluster_count}",
            )
        )
    if min_cluster_recall is not None:
        cluster_truth_count = int(metrics.get("cluster_truth_count", sample.get("cluster_truth_count", 0)))
        cluster_recall = float(metrics.get("cluster_recall", 0.0))
        gates.append(
            _threshold_gate(
                "cluster_recall",
                cluster_truth_count > 0 and cluster_recall >= min_cluster_recall,
                f"{cluster_recall:.4f} >= {min_cluster_recall:.4f}; cluster_truth_count={cluster_truth_count}",
            )
        )
    if min_cluster_full_detection_rate is not None:
        cluster_count = int(metrics.get("cluster_count", sample.get("cluster_count", 0)))
        full_rate = float(metrics.get("fully_detected_cluster_rate", 0.0))
        gates.append(
            _threshold_gate(
                "cluster_full_detection_rate",
                cluster_count > 0 and full_rate >= min_cluster_full_detection_rate,
                f"{full_rate:.4f} >= {min_cluster_full_detection_rate:.4f}; cluster_count={cluster_count}",
            )
        )
    _check_ci_lower("precision_ci_lower", statistics.get("precision_wilson_ci", {}), min_precision_ci_lower, gates)
    _check_ci_lower("recall_ci_lower", statistics.get("recall_wilson_ci", {}), min_recall_ci_lower, gates)


def _check_stratified_acceptance(report: dict[str, Any], gates: list[dict[str, str]]) -> None:
    status = str(report.get("status", "missing"))
    gates.append(
        _gate(
            "stratified_acceptance_passed",
            "pass" if status == "pass" else "fail",
            "stratified acceptance passed" if status == "pass" else f"stratified acceptance status={status}",
        )
    )
    stratum_count = _int_value(report.get("stratum_count"), 0)
    failed_count = _int_value(report.get("failed_stratum_count"), 0)
    missing_count = _int_value(report.get("missing_metadata_count"), 0)
    gates.append(_threshold_gate("stratified_acceptance_strata", stratum_count > 0, f"stratum_count={stratum_count}"))
    gates.append(
        _threshold_gate(
            "stratified_acceptance_failed_strata",
            failed_count == 0,
            f"failed_stratum_count={failed_count}",
        )
    )
    gates.append(
        _threshold_gate(
            "stratified_acceptance_metadata",
            missing_count == 0,
            f"missing_metadata_count={missing_count}",
        )
    )


def _check_ci_lower(
    name: str,
    payload: dict[str, Any],
    threshold: float | None,
    gates: list[dict[str, str]],
) -> None:
    if threshold is None:
        return
    lower = payload.get("lower")
    if lower is None:
        gates.append(_gate(name, "fail", f"CI lower bound missing; required >= {threshold:.4f}"))
        return
    lower = float(lower)
    gates.append(_threshold_gate(name, lower >= threshold, f"{lower:.4f} >= {threshold:.4f}"))


def _check_benchmark(benchmark: dict[str, Any], gates: list[dict[str, str]], max_p95_ms: float | None) -> None:
    p95 = (benchmark.get("latency_ms", {}) or {}).get("p95")
    if not isinstance(p95, int | float):
        gates.append(_gate("benchmark_p95_present", "fail", "benchmark.latency_ms.p95 is missing"))
        return
    gates.append(_gate("benchmark_p95_present", "pass", f"p95={float(p95):.3f}ms"))
    if max_p95_ms is not None:
        gates.append(_threshold_gate("benchmark_p95_limit", float(p95) <= max_p95_ms, f"{float(p95):.3f}ms <= {max_p95_ms:.3f}ms"))


def _check_holdout_verify(
    holdout_verify: dict[str, Any],
    gates: list[dict[str, str]],
    max_count_error_rate: float,
    min_cluster_count: int | None,
) -> None:
    status = holdout_verify.get("status")
    gates.append(
        _gate(
            "holdout_verified",
            "pass" if status == "pass" else "fail",
            "holdout verification passed" if status == "pass" else f"holdout verification status={status}",
        )
    )
    min_detectable = holdout_verify.get("min_detectable_count_error_rate")
    if min_detectable is None:
        gates.append(_gate("holdout_resolution", "fail", "min_detectable_count_error_rate missing"))
        return
    min_detectable = float(min_detectable)
    gates.append(
        _threshold_gate(
            "holdout_resolution",
            min_detectable <= max_count_error_rate,
            f"min_detectable_count_error_rate={min_detectable:.4f} <= {max_count_error_rate:.4f}",
        )
    )
    if min_cluster_count is not None:
        cluster_count = int(holdout_verify.get("cluster_count", 0))
        gates.append(
            _threshold_gate(
                "holdout_min_cluster_count",
                cluster_count >= min_cluster_count,
                f"{cluster_count} >= {min_cluster_count}",
            )
        )


def _check_truth_coverage(
    truth_coverage: dict[str, Any],
    gates: list[dict[str, str]],
    min_truth_count: int | None,
    min_cluster_count: int | None,
    min_cluster_truth_count: int | None,
    min_cluster_images: int | None,
    min_cluster_truth_fraction: float | None,
) -> None:
    status = truth_coverage.get("status")
    gates.append(
        _gate(
            "truth_coverage_passed",
            "pass" if status == "pass" else "fail",
            "truth coverage report passed" if status == "pass" else f"truth coverage status={status}",
        )
    )
    if min_truth_count is not None:
        truth_count = int(truth_coverage.get("truth_count", 0))
        gates.append(_threshold_gate("truth_coverage_truth_count", truth_count >= min_truth_count, f"{truth_count} >= {min_truth_count}"))
    if min_cluster_count is not None:
        cluster_count = int(truth_coverage.get("cluster_count", 0))
        gates.append(
            _threshold_gate(
                "truth_coverage_cluster_count",
                cluster_count >= min_cluster_count,
                f"{cluster_count} >= {min_cluster_count}",
            )
        )
    if min_cluster_truth_count is not None:
        cluster_truth_count = int(truth_coverage.get("cluster_truth_count", 0))
        gates.append(
            _threshold_gate(
                "truth_coverage_cluster_truth_count",
                cluster_truth_count >= min_cluster_truth_count,
                f"{cluster_truth_count} >= {min_cluster_truth_count}",
            )
        )
    if min_cluster_images is not None:
        cluster_image_count = int(truth_coverage.get("cluster_image_count", 0))
        gates.append(
            _threshold_gate(
                "truth_coverage_cluster_images",
                cluster_image_count >= min_cluster_images,
                f"{cluster_image_count} >= {min_cluster_images}",
            )
        )
    if min_cluster_truth_fraction is not None:
        cluster_fraction = float(truth_coverage.get("cluster_truth_fraction", 0.0))
        gates.append(
            _threshold_gate(
                "truth_coverage_cluster_fraction",
                cluster_fraction >= min_cluster_truth_fraction,
                f"{cluster_fraction:.4f} >= {min_cluster_truth_fraction:.4f}",
            )
        )


def _check_stratified_truth_coverage(
    report: dict[str, Any],
    gates: list[dict[str, str]],
    validation_plan: dict[str, Any] | None,
) -> None:
    status = str(report.get("status", "missing"))
    gates.append(
        _gate(
            "stratified_truth_coverage_passed",
            "pass" if status == "pass" else "fail",
            "stratified truth coverage passed"
            if status == "pass"
            else f"stratified truth coverage status={status}",
        )
    )
    stratum_count = _int_value(report.get("stratum_count"), 0)
    failed_count = _int_value(report.get("failed_stratum_count"), 0)
    missing_count = _int_value(report.get("missing_metadata_count"), 0)
    gates.append(_threshold_gate("stratified_truth_coverage_strata", stratum_count > 0, f"stratum_count={stratum_count}"))
    gates.append(
        _threshold_gate(
            "stratified_truth_coverage_failed_strata",
            failed_count == 0,
            f"failed_stratum_count={failed_count}",
        )
    )
    gates.append(
        _threshold_gate(
            "stratified_truth_coverage_metadata",
            missing_count == 0,
            f"missing_metadata_count={missing_count}",
        )
    )
    if validation_plan is None:
        return
    support = validation_plan.get("minimum_support", {}) or {}
    domain = validation_plan.get("operating_domain", {}) or {}
    expected_conditions = domain.get("condition_count")
    if expected_conditions is not None:
        expected_conditions = _int_value(expected_conditions, 0)
        gates.append(
            _threshold_gate(
                "stratified_truth_coverage_condition_count",
                stratum_count >= expected_conditions,
                f"{stratum_count} >= {expected_conditions}",
            )
        )
    thresholds = report.get("thresholds", {}) or {}
    _check_stratified_threshold(
        "stratified_truth_coverage_truth_per_condition",
        thresholds,
        "min_truth_count",
        support.get("plants_per_condition"),
        gates,
    )
    _check_stratified_threshold(
        "stratified_truth_coverage_clusters_per_condition",
        thresholds,
        "min_cluster_count",
        support.get("cluster_mats_per_condition"),
        gates,
    )
    _check_stratified_threshold(
        "stratified_truth_coverage_cluster_truth_per_condition",
        thresholds,
        "min_cluster_truth_count",
        support.get("cluster_truth_per_condition"),
        gates,
    )


def _check_stratified_threshold(
    gate_name: str,
    thresholds: dict[str, Any],
    threshold_key: str,
    required: Any,
    gates: list[dict[str, str]],
) -> None:
    if required is None:
        return
    observed = _int_value(thresholds.get(threshold_key), 0)
    required_value = _int_value(required, 0)
    gates.append(_threshold_gate(gate_name, observed >= required_value, f"{observed} >= {required_value}"))


def _check_quality(name: str, report: dict[str, Any], gates: list[dict[str, str]], allow_warn: bool) -> None:
    status = str(report.get("status", "missing"))
    if status == "pass":
        gates.append(_gate(name, "pass", "quality report passed"))
    elif status == "warn" and allow_warn:
        gates.append(_gate(name, "warn", "quality report warned; allowed by threshold"))
    else:
        gates.append(_gate(name, "fail", f"quality status={status}"))


def _check_flight_check(report: dict[str, Any], gates: list[dict[str, str]]) -> None:
    status = str(report.get("status", "missing"))
    gates.append(
        _gate(
            "flight_check_passed",
            "pass" if status == "pass" else "fail",
            "flight profile passed" if status == "pass" else f"flight profile status={status}",
        )
    )
    failed_checks = [
        str(check.get("name"))
        for check in report.get("checks", [])
        if isinstance(check, dict) and check.get("status") == "fail"
    ]
    gates.append(
        _gate(
            "flight_check_failures",
            "pass" if not failed_checks else "fail",
            "no failed flight checks" if not failed_checks else ", ".join(failed_checks),
        )
    )


def _check_flight_log(report: dict[str, Any], gates: list[dict[str, str]]) -> None:
    status = str(report.get("status", "missing"))
    gates.append(
        _gate(
            "flight_log_passed",
            "pass" if status == "pass" else "fail",
            "flight log audit passed" if status == "pass" else f"flight log status={status}",
        )
    )
    summary = report.get("summary", {}) or {}
    row_count = _int_value(summary.get("row_count"), 0)
    fail_count = _int_value(summary.get("fail_count"), 0)
    gates.append(_threshold_gate("flight_log_rows", row_count > 0, f"row_count={row_count}"))
    gates.append(_threshold_gate("flight_log_failures", fail_count == 0, f"fail_count={fail_count}"))


def _check_domain_check(report: dict[str, Any], gates: list[dict[str, str]]) -> None:
    status = str(report.get("status", "missing"))
    gates.append(
        _gate(
            "domain_check_passed",
            "pass" if status == "pass" else "fail",
            "domain check passed" if status == "pass" else f"domain check status={status}",
        )
    )
    outlier_fraction = float(report.get("outlier_fraction", 1.0))
    maximum = float((report.get("thresholds") or {}).get("max_outlier_fraction", 0.0))
    gates.append(
        _threshold_gate(
            "domain_outlier_fraction",
            outlier_fraction <= maximum,
            f"{outlier_fraction:.4f} <= {maximum:.4f}",
        )
    )
    reference_count = int(report.get("reference_image_count", 0))
    minimum = int((report.get("thresholds") or {}).get("min_reference_images", 1))
    gates.append(
        _threshold_gate(
            "domain_reference_support",
            reference_count >= minimum,
            f"{reference_count} >= {minimum}",
        )
    )


def _check_geo_accuracy(
    report: dict[str, Any],
    gates: list[dict[str, str]],
    max_geo_rmse_m: float,
    max_geo_p95_m: float | None,
    min_geo_recall: float,
) -> None:
    status = str(report.get("status", "missing"))
    gates.append(
        _gate(
            "geo_accuracy_passed",
            "pass" if status == "pass" else "fail",
            "geo accuracy report passed" if status == "pass" else f"geo accuracy status={status}",
        )
    )
    metrics = report.get("metrics", {}) or {}
    thresholds = report.get("thresholds", {}) or {}
    truth_count = _int_value(metrics.get("truth_count"), 0)
    matched_count = _int_value(metrics.get("matched_count"), 0)
    gates.append(_threshold_gate("geo_truth_support", truth_count > 0, f"truth_count={truth_count}"))
    gates.append(_threshold_gate("geo_match_support", matched_count > 0, f"matched_count={matched_count}"))

    rmse = _float_value(metrics.get("rmse_m"), float("inf"))
    gates.append(_threshold_gate("geo_rmse", rmse <= max_geo_rmse_m, f"{rmse:.4f}m <= {max_geo_rmse_m:.4f}m"))
    configured_rmse = thresholds.get("max_rmse_m")
    if configured_rmse is None:
        gates.append(_gate("geo_rmse_threshold", "fail", "thresholds.max_rmse_m is missing"))
    else:
        configured_rmse = _float_value(configured_rmse, float("inf"))
        gates.append(
            _threshold_gate(
                "geo_rmse_threshold",
                configured_rmse <= max_geo_rmse_m,
                f"{configured_rmse:.4f}m <= {max_geo_rmse_m:.4f}m",
            )
        )

    recall = _float_value(metrics.get("recall"), 0.0)
    gates.append(_threshold_gate("geo_recall", recall >= min_geo_recall, f"{recall:.4f} >= {min_geo_recall:.4f}"))
    configured_recall = thresholds.get("min_recall")
    if configured_recall is None:
        gates.append(_gate("geo_recall_threshold", "fail", "thresholds.min_recall is missing"))
    else:
        configured_recall = _float_value(configured_recall, 0.0)
        gates.append(
            _threshold_gate(
                "geo_recall_threshold",
                configured_recall >= min_geo_recall,
                f"{configured_recall:.4f} >= {min_geo_recall:.4f}",
            )
        )

    if max_geo_p95_m is not None:
        p95 = _float_value(metrics.get("p95_m"), float("inf"))
        gates.append(_threshold_gate("geo_p95", p95 <= max_geo_p95_m, f"{p95:.4f}m <= {max_geo_p95_m:.4f}m"))
        configured_p95 = thresholds.get("max_p95_m")
        if configured_p95 is None:
            gates.append(_gate("geo_p95_threshold", "fail", "thresholds.max_p95_m is missing"))
        else:
            configured_p95 = _float_value(configured_p95, float("inf"))
            gates.append(
                _threshold_gate(
                    "geo_p95_threshold",
                    configured_p95 <= max_geo_p95_m,
                    f"{configured_p95:.4f}m <= {max_geo_p95_m:.4f}m",
                )
            )


def _check_model_manifest(model: dict[str, Any], gates: list[dict[str, str]]) -> None:
    gates.append(_gate("model_sha256_present", "pass" if model.get("model_sha256") else "fail", "model_sha256 present" if model.get("model_sha256") else "model_sha256 missing"))
    promotion = model.get("promotion") or {}
    if promotion:
        gates.append(_gate("model_promoted", "pass" if promotion.get("status") == "promoted" else "fail", f"promotion status={promotion.get('status')}"))
    else:
        gates.append(_gate("model_promoted", "warn", "promotion metadata missing"))


def _check_file(name: str, path: str | Path | None, gates: list[dict[str, str]]) -> None:
    if path is None:
        gates.append(_gate(f"{name}_present", "warn", "artifact path not provided"))
        return
    path = Path(path)
    gates.append(_gate(f"{name}_present", "pass" if path.exists() else "warn", str(path) if path.exists() else f"artifact does not exist: {path}"))


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _threshold_gate(name: str, passed: bool, detail: str) -> dict[str, str]:
    return _gate(name, "pass" if passed else "fail", detail)


def _gate(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _overall_status(gates: list[dict[str, str]]) -> str:
    if any(gate["status"] == "fail" for gate in gates):
        return "fail"
    if any(gate["status"] == "warn" for gate in gates):
        return "warn"
    return "pass"
