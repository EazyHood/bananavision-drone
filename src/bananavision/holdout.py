from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import iter_images
from .runtime import file_sha256, stable_json_hash, utc_now_iso
from .truth import read_truth_points, truth_cluster_summary


def lock_holdout(
    image_path: str | Path,
    truth_path: str | Path,
    output_json: str | Path,
    name: str = "BananaVision locked holdout",
    target_count_error_rate: float = 0.01,
) -> dict[str, Any]:
    images = iter_images(image_path)
    truth_path = Path(truth_path)
    entries = []
    truth_total = 0
    cluster_total = 0
    cluster_truth_total = 0
    for image in images:
        truth_source = _truth_source_for_image(truth_path, image)
        points = read_truth_points(truth_source, image=image)
        cluster_summary = truth_cluster_summary(points)
        truth_total += len(points)
        cluster_total += cluster_summary["cluster_count"]
        cluster_truth_total += cluster_summary["cluster_truth_count"]
        entries.append(
            {
                "image": str(image),
                "image_sha256": file_sha256(image),
                "truth_source": str(truth_source),
                "truth_sha256": file_sha256(truth_source),
                "truth_count": len(points),
                "cluster_count": cluster_summary["cluster_count"],
                "cluster_truth_count": cluster_summary["cluster_truth_count"],
            }
        )
    payload = {
        "version": 1,
        "name": name,
        "created_at": utc_now_iso(),
        "image_path": str(image_path),
        "truth_path": str(truth_path),
        "target_count_error_rate": target_count_error_rate,
        "image_count": len(entries),
        "truth_count": truth_total,
        "cluster_count": cluster_total,
        "cluster_truth_count": cluster_truth_total,
        "min_detectable_count_error_rate": 0.0 if truth_total <= 0 else 1.0 / truth_total,
        "min_detectable_cluster_recall_loss": 0.0 if cluster_truth_total <= 0 else 1.0 / cluster_truth_total,
        "min_detectable_cluster_full_detection_loss": 0.0 if cluster_total <= 0 else 1.0 / cluster_total,
        "claim_resolution_ok": truth_total > 0 and (1.0 / truth_total) <= target_count_error_rate,
        "entries": entries,
    }
    payload["lock_sha256"] = stable_json_hash(payload)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_holdout_lock(
    lock_json: str | Path,
    output_json: str | Path | None = None,
    expected_image_path: str | Path | None = None,
    expected_truth_path: str | Path | None = None,
) -> dict[str, Any]:
    lock_json = Path(lock_json)
    lock = json.loads(lock_json.read_text(encoding="utf-8"))
    issues: list[dict[str, Any]] = []
    _verify_expected_path("image_path", lock.get("image_path"), expected_image_path, issues)
    _verify_expected_path("truth_path", lock.get("truth_path"), expected_truth_path, issues)
    for entry in lock.get("entries", []):
        _verify_entry(entry, issues)
    status = "pass" if not issues else "fail"
    report = {
        "created_at": utc_now_iso(),
        "lock_json": str(lock_json),
        "status": status,
        "image_path": lock.get("image_path"),
        "truth_path": lock.get("truth_path"),
        "image_count": lock.get("image_count", 0),
        "truth_count": lock.get("truth_count", 0),
        "cluster_count": lock.get("cluster_count", 0),
        "cluster_truth_count": lock.get("cluster_truth_count", 0),
        "target_count_error_rate": lock.get("target_count_error_rate"),
        "min_detectable_count_error_rate": lock.get("min_detectable_count_error_rate"),
        "min_detectable_cluster_recall_loss": lock.get("min_detectable_cluster_recall_loss"),
        "min_detectable_cluster_full_detection_loss": lock.get("min_detectable_cluster_full_detection_loss"),
        "claim_resolution_ok": lock.get("claim_resolution_ok"),
        "verified_entries": len(lock.get("entries", [])),
        "issue_count": len(issues),
        "issues": issues,
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _verify_expected_path(
    label: str,
    locked_path: str | None,
    expected_path: str | Path | None,
    issues: list[dict[str, Any]],
) -> None:
    if expected_path is None or locked_path is None:
        return
    if _normalized_path(locked_path) != _normalized_path(expected_path):
        issues.append(
            {
                "type": f"{label}_mismatch",
                "locked": str(locked_path),
                "expected": str(expected_path),
            }
        )


def _verify_entry(entry: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    image = Path(entry["image"])
    truth_source = Path(entry["truth_source"])
    if not image.exists():
        issues.append({"image": str(image), "type": "missing_image"})
    elif file_sha256(image) != entry.get("image_sha256"):
        issues.append({"image": str(image), "type": "image_hash_changed"})
    if not truth_source.exists():
        issues.append({"image": str(image), "truth_source": str(truth_source), "type": "missing_truth"})
        return
    if file_sha256(truth_source) != entry.get("truth_sha256"):
        issues.append({"image": str(image), "truth_source": str(truth_source), "type": "truth_hash_changed"})
    try:
        points = read_truth_points(truth_source, image=image)
        truth_count = len(points)
        cluster_summary = truth_cluster_summary(points)
    except Exception as exc:
        issues.append({"image": str(image), "truth_source": str(truth_source), "type": "truth_unreadable", "detail": str(exc)})
        return
    if truth_count != int(entry.get("truth_count", -1)):
        issues.append(
            {
                "image": str(image),
                "truth_source": str(truth_source),
                "type": "truth_count_changed",
                "expected": entry.get("truth_count"),
                "actual": truth_count,
            }
        )
    _verify_locked_count(
        entry,
        issues,
        image=image,
        truth_source=truth_source,
        key="cluster_count",
        actual=cluster_summary["cluster_count"],
    )
    _verify_locked_count(
        entry,
        issues,
        image=image,
        truth_source=truth_source,
        key="cluster_truth_count",
        actual=cluster_summary["cluster_truth_count"],
    )


def _verify_locked_count(
    entry: dict[str, Any],
    issues: list[dict[str, Any]],
    image: Path,
    truth_source: Path,
    key: str,
    actual: int,
) -> None:
    if key not in entry:
        return
    expected = int(entry.get(key, -1))
    if actual != expected:
        issues.append(
            {
                "image": str(image),
                "truth_source": str(truth_source),
                "type": f"{key}_changed",
                "expected": expected,
                "actual": actual,
            }
        )


def _truth_source_for_image(truth_path: Path, image: Path) -> Path:
    if truth_path.is_file():
        return truth_path
    candidates = [
        truth_path / f"{image.stem}.truth.json",
        truth_path / f"{image.stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No truth file found for {image.name} in {truth_path}")


def _normalized_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve()).lower()
    except Exception:
        return str(path).replace("/", "\\").lower()
