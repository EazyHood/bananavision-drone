from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .runtime import utc_now_iso
from .truth_coverage import truth_coverage_image_reports


def build_stratified_truth_coverage_report(
    truth_path: str | Path,
    metadata_csv: str | Path,
    output_json: str | Path,
    strata_keys: list[str],
    image_path: str | Path | None = None,
    min_truth_count: int = 0,
    min_cluster_count: int = 0,
    min_cluster_truth_count: int = 0,
    min_cluster_images: int = 0,
    min_cluster_truth_fraction: float = 0.0,
) -> dict[str, Any]:
    if not strata_keys:
        raise ValueError("At least one --strata key is required")
    truth_path = Path(truth_path)
    metadata_path = Path(metadata_csv)
    image_path_value = None if image_path is None else Path(image_path)
    image_reports = truth_coverage_image_reports(truth_path, image_path_value)
    metadata = _read_metadata(metadata_path, strata_keys)
    strata, missing_metadata = _group_images(image_reports, metadata, strata_keys)
    thresholds = {
        "min_truth_count": min_truth_count,
        "min_cluster_count": min_cluster_count,
        "min_cluster_truth_count": min_cluster_truth_count,
        "min_cluster_images": min_cluster_images,
        "min_cluster_truth_fraction": min_cluster_truth_fraction,
    }
    rows = [
        _stratum_report(key, reports, strata_keys, thresholds)
        for key, reports in sorted(strata.items(), key=lambda item: item[0])
    ]
    checks = _checks(rows, missing_metadata)
    report = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "status": _overall_status(checks),
        "truth_path": str(truth_path),
        "image_path": None if image_path_value is None else str(image_path_value),
        "metadata_csv": str(metadata_path),
        "strata_keys": strata_keys,
        "thresholds": thresholds,
        "image_count": len(image_reports),
        "metadata_row_count": len(metadata),
        "stratum_count": len(rows),
        "missing_metadata_count": len(missing_metadata),
        "failed_stratum_count": sum(1 for row in rows if row["status"] == "fail"),
        "checks": checks,
        "missing_metadata_images": missing_metadata,
        "strata": rows,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_stratified_truth_coverage_csv(report, output_json.with_suffix(".csv"))
    return report


def write_stratified_truth_coverage_csv(report: dict[str, Any], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *report.get("strata_keys", []),
        "status",
        "image_count",
        "truth_count",
        "cluster_count",
        "cluster_truth_count",
        "cluster_image_count",
        "cluster_truth_fraction",
        "mean_cluster_size",
        "max_cluster_size",
        "truth_count_deficit",
        "cluster_count_deficit",
        "cluster_truth_count_deficit",
        "cluster_image_count_deficit",
        "cluster_truth_fraction_cluster_truth_deficit",
        "failed_checks",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("strata", []):
            failed = [check["name"] for check in row["checks"] if check["status"] == "fail"]
            writer.writerow(
                {
                    **{key: row["stratum"].get(key, "") for key in report.get("strata_keys", [])},
                    "status": row["status"],
                    "image_count": row["image_count"],
                    "truth_count": row["truth_count"],
                    "cluster_count": row["cluster_count"],
                    "cluster_truth_count": row["cluster_truth_count"],
                    "cluster_image_count": row["cluster_image_count"],
                    "cluster_truth_fraction": row["cluster_truth_fraction"],
                    "mean_cluster_size": row["mean_cluster_size"],
                    "max_cluster_size": row["max_cluster_size"],
                    "truth_count_deficit": row["deficits"]["truth_count"],
                    "cluster_count_deficit": row["deficits"]["cluster_count"],
                    "cluster_truth_count_deficit": row["deficits"]["cluster_truth_count"],
                    "cluster_image_count_deficit": row["deficits"]["cluster_image_count"],
                    "cluster_truth_fraction_cluster_truth_deficit": row["deficits"][
                        "cluster_truth_fraction_cluster_truth"
                    ],
                    "failed_checks": ",".join(failed),
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
    truth_count = sum(int(item.get("truth_count", 0) or 0) for item in image_reports)
    cluster_count = sum(int(item.get("cluster_count", 0) or 0) for item in image_reports)
    cluster_truth_count = sum(int(item.get("cluster_truth_count", 0) or 0) for item in image_reports)
    cluster_image_count = sum(1 for item in image_reports if int(item.get("cluster_count", 0) or 0) > 0)
    cluster_sizes = [int(size) for item in image_reports for size in item.get("cluster_sizes", [])]
    cluster_truth_fraction = cluster_truth_count / max(1, truth_count)
    deficits = _deficits(
        truth_count=truth_count,
        cluster_count=cluster_count,
        cluster_truth_count=cluster_truth_count,
        cluster_image_count=cluster_image_count,
        cluster_truth_fraction=cluster_truth_fraction,
        thresholds=thresholds,
    )
    checks = [
        _threshold_check(
            "truth_count",
            truth_count >= int(thresholds["min_truth_count"]),
            f"{truth_count} >= {int(thresholds['min_truth_count'])}",
        ),
        _threshold_check(
            "cluster_count",
            cluster_count >= int(thresholds["min_cluster_count"]),
            f"{cluster_count} >= {int(thresholds['min_cluster_count'])}",
        ),
        _threshold_check(
            "cluster_truth_count",
            cluster_truth_count >= int(thresholds["min_cluster_truth_count"]),
            f"{cluster_truth_count} >= {int(thresholds['min_cluster_truth_count'])}",
        ),
        _threshold_check(
            "cluster_image_count",
            cluster_image_count >= int(thresholds["min_cluster_images"]),
            f"{cluster_image_count} >= {int(thresholds['min_cluster_images'])}",
        ),
        _threshold_check(
            "cluster_truth_fraction",
            cluster_truth_fraction >= float(thresholds["min_cluster_truth_fraction"]),
            f"{cluster_truth_fraction:.4f} >= {float(thresholds['min_cluster_truth_fraction']):.4f}",
        ),
    ]
    return {
        "stratum": dict(zip(strata_keys, key, strict=False)),
        "status": _overall_status(checks),
        "image_count": len(image_reports),
        "truth_count": truth_count,
        "cluster_count": cluster_count,
        "cluster_truth_count": cluster_truth_count,
        "cluster_image_count": cluster_image_count,
        "cluster_truth_fraction": cluster_truth_fraction,
        "mean_cluster_size": 0.0 if not cluster_sizes else sum(cluster_sizes) / len(cluster_sizes),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "min_detectable_count_error_rate": 0.0 if truth_count <= 0 else 1.0 / truth_count,
        "min_detectable_cluster_recall_loss": 0.0 if cluster_truth_count <= 0 else 1.0 / cluster_truth_count,
        "min_detectable_cluster_full_detection_loss": 0.0 if cluster_count <= 0 else 1.0 / cluster_count,
        "deficits": deficits,
        "actions": _actions(deficits),
        "checks": checks,
        "images": [str(image.get("image", "")) for image in image_reports],
    }


def _deficits(
    truth_count: int,
    cluster_count: int,
    cluster_truth_count: int,
    cluster_image_count: int,
    cluster_truth_fraction: float,
    thresholds: dict[str, Any],
) -> dict[str, int]:
    min_truth_count = int(thresholds["min_truth_count"])
    min_cluster_count = int(thresholds["min_cluster_count"])
    min_cluster_truth_count = int(thresholds["min_cluster_truth_count"])
    min_cluster_images = int(thresholds["min_cluster_images"])
    min_cluster_truth_fraction = float(thresholds["min_cluster_truth_fraction"])
    fraction_deficit = 0
    if cluster_truth_fraction < min_cluster_truth_fraction:
        fraction_deficit = max(0, math.ceil(min_cluster_truth_fraction * max(1, truth_count)) - cluster_truth_count)
    return {
        "truth_count": max(0, min_truth_count - truth_count),
        "cluster_count": max(0, min_cluster_count - cluster_count),
        "cluster_truth_count": max(0, min_cluster_truth_count - cluster_truth_count),
        "cluster_image_count": max(0, min_cluster_images - cluster_image_count),
        "cluster_truth_fraction_cluster_truth": fraction_deficit,
    }


def _actions(deficits: dict[str, int]) -> list[str]:
    actions = []
    if deficits["truth_count"] > 0:
        actions.append(f"annotate at least {_count_label(deficits['truth_count'], 'additional plant')} in this stratum")
    if deficits["cluster_count"] > 0:
        actions.append(f"annotate at least {_count_label(deficits['cluster_count'], 'additional grouped banana mat')}")
    if deficits["cluster_truth_count"] > 0:
        actions.append(f"add at least {_count_label(deficits['cluster_truth_count'], 'grouped-mat plant annotation')}")
    if deficits["cluster_image_count"] > 0:
        actions.append(f"include at least {_count_label(deficits['cluster_image_count'], 'more image')} containing grouped mats")
    if deficits["cluster_truth_fraction_cluster_truth"] > 0:
        actions.append(
            "increase grouped-mat truth support by at least "
            f"{_count_label(deficits['cluster_truth_fraction_cluster_truth'], 'plant annotation')}"
        )
    return actions


def _count_label(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def _checks(rows: list[dict[str, Any]], missing_metadata: list[str]) -> list[dict[str, str]]:
    failed_count = sum(1 for row in rows if row["status"] == "fail")
    return [
        _threshold_check("metadata_complete", not missing_metadata, f"missing_metadata_count={len(missing_metadata)}"),
        _threshold_check("strata_present", bool(rows), f"stratum_count={len(rows)}"),
        _threshold_check("strata_passed", failed_count == 0, f"failed_stratum_count={failed_count}"),
    ]


def _threshold_check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


def _overall_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"
