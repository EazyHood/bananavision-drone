from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .evaluation import batch_acceptance_passed
from .metrics import BatchMetrics, PointMetrics, aggregate_point_metrics, batch_statistical_evidence
from .runtime import utc_now_iso


def build_stratified_acceptance_report(
    acceptance_report: str | Path,
    metadata_csv: str | Path,
    output_json: str | Path,
    strata_keys: list[str],
    max_count_error_rate: float,
    min_precision: float,
    min_recall: float,
    min_f1: float,
    max_mean_image_count_error_rate: float | None = None,
    min_truth_count: int | None = None,
    min_precision_ci_lower: float | None = None,
    min_recall_ci_lower: float | None = None,
    max_mean_image_count_error_rate_ci_upper: float | None = None,
    min_cluster_recall: float | None = None,
    min_cluster_full_detection_rate: float | None = None,
    min_cluster_count: int | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if not strata_keys:
        raise ValueError("At least one --strata key is required")
    acceptance_path = Path(acceptance_report)
    metadata_path = Path(metadata_csv)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    metadata = _read_metadata(metadata_path, strata_keys)
    image_reports = acceptance.get("images", []) or []
    strata, missing_metadata = _group_images(image_reports, metadata, strata_keys)
    thresholds = {
        "max_count_error_rate": max_count_error_rate,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "min_f1": min_f1,
        "max_mean_image_count_error_rate": max_mean_image_count_error_rate,
        "min_truth_count": min_truth_count,
        "min_precision_ci_lower": min_precision_ci_lower,
        "min_recall_ci_lower": min_recall_ci_lower,
        "max_mean_image_count_error_rate_ci_upper": max_mean_image_count_error_rate_ci_upper,
        "min_cluster_recall": min_cluster_recall,
        "min_cluster_full_detection_rate": min_cluster_full_detection_rate,
        "min_cluster_count": min_cluster_count,
        "confidence_level": confidence_level,
    }
    rows = [
        _stratum_report(key, reports, strata_keys, thresholds)
        for key, reports in sorted(strata.items(), key=lambda item: item[0])
    ]
    checks = _checks(rows, missing_metadata, image_reports)
    report = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "status": _overall_status(checks),
        "acceptance_report": str(acceptance_path),
        "metadata_csv": str(metadata_path),
        "strata_keys": strata_keys,
        "thresholds": thresholds,
        "image_count": len(image_reports),
        "metadata_row_count": len(metadata),
        "stratum_count": len(rows),
        "missing_metadata_count": len(missing_metadata),
        "failed_stratum_count": sum(1 for row in rows if not row["passed"]),
        "checks": checks,
        "missing_metadata_images": missing_metadata,
        "strata": rows,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_stratified_acceptance_csv(report, output_json.with_suffix(".csv"))
    return report


def write_stratified_acceptance_csv(report: dict[str, Any], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *report.get("strata_keys", []),
        "passed",
        "image_count",
        "truth_count",
        "prediction_count",
        "count_error_rate",
        "mean_abs_image_count_error_rate",
        "precision",
        "recall",
        "f1",
        "cluster_count",
        "cluster_recall",
        "fully_detected_cluster_rate",
        "failed_gates",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("strata", []):
            metrics = row["metrics"]
            values = {key: row["stratum"].get(key, "") for key in report.get("strata_keys", [])}
            failed = [gate["name"] for gate in row["gates"] if gate["status"] == "fail"]
            writer.writerow(
                {
                    **values,
                    "passed": row["passed"],
                    "image_count": metrics["images"],
                    "truth_count": metrics["truth_count"],
                    "prediction_count": metrics["prediction_count"],
                    "count_error_rate": metrics["count_error_rate"],
                    "mean_abs_image_count_error_rate": metrics["mean_abs_image_count_error_rate"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "cluster_count": metrics.get("cluster_count", 0),
                    "cluster_recall": metrics.get("cluster_recall", 0.0),
                    "fully_detected_cluster_rate": metrics.get("fully_detected_cluster_rate", 0.0),
                    "failed_gates": ",".join(failed),
                }
            )
    return output_csv


def _read_metadata(path: Path, strata_keys: list[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError("Metadata CSV must contain an image column")
        missing = [key for key in strata_keys if key not in reader.fieldnames]
        if missing:
            raise ValueError("Metadata CSV missing strata column(s): " + ", ".join(missing))
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            image = str(row.get("image", "")).strip()
            if not image:
                continue
            clean = {key: str(row.get(key, "")).strip() or "missing" for key in strata_keys}
            rows[image] = clean
            rows[Path(image).name] = clean
    return rows


def _group_images(
    image_reports: list[dict[str, Any]],
    metadata: dict[str, dict[str, str]],
    strata_keys: list[str],
) -> tuple[dict[tuple[str, ...], list[dict[str, Any]]], list[str]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    missing_metadata = []
    for image in image_reports:
        image_value = str(image.get("image", ""))
        row = metadata.get(image_value) or metadata.get(Path(image_value).name)
        if row is None:
            missing_metadata.append(image_value)
            continue
        key = tuple(row[strata] for strata in strata_keys)
        groups.setdefault(key, []).append(image)
    return groups, missing_metadata


def _stratum_report(
    key: tuple[str, ...],
    image_reports: list[dict[str, Any]],
    strata_keys: list[str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    point_metrics = [_point_metrics(image["metrics"]) for image in image_reports]
    truth_counts = [int(image.get("truth_count", image["metrics"].get("truth_count", 0)) or 0) for image in image_reports]
    prediction_counts = [
        int(image.get("prediction_count", image["metrics"].get("prediction_count", 0)) or 0)
        for image in image_reports
    ]
    metrics = aggregate_point_metrics(point_metrics, truth_counts, prediction_counts)
    statistics = batch_statistical_evidence(
        metrics,
        [float(item.count_error_rate) for item in point_metrics],
        confidence_level=float(thresholds["confidence_level"]),
    )
    passed = batch_acceptance_passed(
        metrics,
        max_count_error_rate=float(thresholds["max_count_error_rate"]),
        min_precision=float(thresholds["min_precision"]),
        min_recall=float(thresholds["min_recall"]),
        min_f1=float(thresholds["min_f1"]),
        max_mean_image_count_error_rate=thresholds.get("max_mean_image_count_error_rate"),
        min_truth_count=thresholds.get("min_truth_count"),
        min_precision_ci_lower=thresholds.get("min_precision_ci_lower"),
        min_recall_ci_lower=thresholds.get("min_recall_ci_lower"),
        max_mean_image_count_error_rate_ci_upper=thresholds.get("max_mean_image_count_error_rate_ci_upper"),
        min_cluster_recall=thresholds.get("min_cluster_recall"),
        min_cluster_full_detection_rate=thresholds.get("min_cluster_full_detection_rate"),
        min_cluster_count=thresholds.get("min_cluster_count"),
        statistics=statistics,
    )
    return {
        "stratum": dict(zip(strata_keys, key, strict=False)),
        "passed": passed,
        "metrics": asdict(metrics),
        "statistics": statistics,
        "gates": _stratum_gates(metrics, statistics, thresholds),
        "images": [str(image.get("image", "")) for image in image_reports],
    }


def _point_metrics(payload: dict[str, Any]) -> PointMetrics:
    accepted = {field.name for field in fields(PointMetrics)}
    values = {key: payload.get(key) for key in accepted if key in payload}
    defaults = {
        "cluster_count": 0,
        "cluster_truth_count": 0,
        "cluster_matched_count": 0,
        "cluster_recall": 0.0,
        "fully_detected_cluster_count": 0,
        "fully_detected_cluster_rate": 0.0,
        "under_split_cluster_count": 0,
        "over_split_cluster_count": 0,
        "cluster_extra_prediction_count": 0,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    return PointMetrics(**values)


def _stratum_gates(
    metrics: BatchMetrics,
    statistics: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, str]]:
    gates = [
        _threshold_gate(
            "count_error_rate",
            metrics.count_error_rate <= float(thresholds["max_count_error_rate"]),
            f"{metrics.count_error_rate:.4f} <= {float(thresholds['max_count_error_rate']):.4f}",
        ),
        _threshold_gate(
            "precision",
            metrics.precision >= float(thresholds["min_precision"]),
            f"{metrics.precision:.4f} >= {float(thresholds['min_precision']):.4f}",
        ),
        _threshold_gate(
            "recall",
            metrics.recall >= float(thresholds["min_recall"]),
            f"{metrics.recall:.4f} >= {float(thresholds['min_recall']):.4f}",
        ),
        _threshold_gate(
            "f1",
            metrics.f1 >= float(thresholds["min_f1"]),
            f"{metrics.f1:.4f} >= {float(thresholds['min_f1']):.4f}",
        ),
    ]
    if thresholds.get("max_mean_image_count_error_rate") is not None:
        limit = float(thresholds["max_mean_image_count_error_rate"])
        gates.append(
            _threshold_gate(
                "mean_image_count_error_rate",
                metrics.mean_abs_image_count_error_rate <= limit,
                f"{metrics.mean_abs_image_count_error_rate:.4f} <= {limit:.4f}",
            )
        )
    if thresholds.get("min_truth_count") is not None:
        required = int(thresholds["min_truth_count"])
        gates.append(_threshold_gate("truth_support", metrics.truth_count >= required, f"{metrics.truth_count} >= {required}"))
    if thresholds.get("min_cluster_count") is not None:
        required = int(thresholds["min_cluster_count"])
        gates.append(
            _threshold_gate("cluster_support", metrics.cluster_count >= required, f"{metrics.cluster_count} >= {required}")
        )
    if thresholds.get("min_cluster_recall") is not None:
        required = float(thresholds["min_cluster_recall"])
        gates.append(
            _threshold_gate(
                "cluster_recall",
                metrics.cluster_truth_count > 0 and metrics.cluster_recall >= required,
                f"{metrics.cluster_recall:.4f} >= {required:.4f}; cluster_truth_count={metrics.cluster_truth_count}",
            )
        )
    if thresholds.get("min_cluster_full_detection_rate") is not None:
        required = float(thresholds["min_cluster_full_detection_rate"])
        gates.append(
            _threshold_gate(
                "cluster_full_detection_rate",
                metrics.cluster_count > 0 and metrics.fully_detected_cluster_rate >= required,
                f"{metrics.fully_detected_cluster_rate:.4f} >= {required:.4f}; cluster_count={metrics.cluster_count}",
            )
        )
    _ci_gate(gates, "precision_ci_lower", statistics.get("precision_wilson_ci", {}), thresholds.get("min_precision_ci_lower"))
    _ci_gate(gates, "recall_ci_lower", statistics.get("recall_wilson_ci", {}), thresholds.get("min_recall_ci_lower"))
    mean_ci = statistics.get("mean_abs_image_count_error_rate_ci", {})
    if thresholds.get("max_mean_image_count_error_rate_ci_upper") is not None:
        limit = float(thresholds["max_mean_image_count_error_rate_ci_upper"])
        upper = mean_ci.get("upper")
        gates.append(
            _threshold_gate(
                "mean_image_count_error_rate_ci_upper",
                upper is not None and float(upper) <= limit,
                "missing" if upper is None else f"{float(upper):.4f} <= {limit:.4f}",
            )
        )
    return gates


def _checks(
    rows: list[dict[str, Any]],
    missing_metadata: list[str],
    image_reports: list[dict[str, Any]],
) -> list[dict[str, str]]:
    checks = [
        _threshold_gate("images_present", bool(image_reports), f"image_count={len(image_reports)}"),
        _threshold_gate("metadata_complete", not missing_metadata, f"missing_metadata_count={len(missing_metadata)}"),
        _threshold_gate("strata_present", bool(rows), f"stratum_count={len(rows)}"),
    ]
    for index, row in enumerate(rows, start=1):
        label = ",".join(f"{key}={value}" for key, value in row["stratum"].items())
        checks.append(_threshold_gate(f"stratum:{index}", bool(row["passed"]), label))
    return checks


def _ci_gate(
    gates: list[dict[str, str]],
    name: str,
    payload: dict[str, Any],
    threshold: float | None,
) -> None:
    if threshold is None:
        return
    lower = payload.get("lower")
    gates.append(
        _threshold_gate(
            name,
            lower is not None and float(lower) >= float(threshold),
            "missing" if lower is None else f"{float(lower):.4f} >= {float(threshold):.4f}",
        )
    )


def _threshold_gate(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


def _overall_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"
