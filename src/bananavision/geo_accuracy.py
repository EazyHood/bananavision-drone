from __future__ import annotations

import csv
import json
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from statistics import median
from typing import Any

from .runtime import utc_now_iso

EARTH_RADIUS_M = 6_371_008.8


def audit_geo_accuracy(
    predictions_geojson: str | Path,
    truth_geojson: str | Path,
    output_json: str | Path | None = None,
    tolerance_m: float = 1.0,
    max_rmse_m: float = 1.0,
    max_p95_m: float | None = None,
    min_recall: float | None = 0.99,
) -> dict[str, Any]:
    _require_positive("tolerance_m", tolerance_m)
    _require_positive("max_rmse_m", max_rmse_m)
    if max_p95_m is not None:
        _require_positive("max_p95_m", max_p95_m)
    if min_recall is not None and not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be between 0 and 1")

    predictions = _read_points(predictions_geojson, require_points=False)
    truth = _read_points(truth_geojson, require_points=True)
    matches, unmatched_predictions, unmatched_truth = _match_points(predictions, truth, tolerance_m)
    distances = [match["distance_m"] for match in matches]
    metrics = _metrics(
        prediction_count=len(predictions),
        truth_count=len(truth),
        matched_count=len(matches),
        distances=distances,
    )
    checks = [
        _check(
            "prediction_support",
            "pass" if metrics["prediction_count"] > 0 else "fail",
            f"prediction_count={metrics['prediction_count']}",
        ),
        _check(
            "matched_support",
            "pass" if metrics["matched_count"] > 0 else "fail",
            f"matched_count={metrics['matched_count']}",
        ),
        _check(
            "matched_rmse",
            "pass" if metrics["rmse_m"] <= max_rmse_m else "fail",
            f"{metrics['rmse_m']:.4f} <= {max_rmse_m:.4f}",
        )
    ]
    if max_p95_m is not None:
        checks.append(
            _check(
                "matched_p95",
                "pass" if metrics["p95_m"] <= max_p95_m else "fail",
                f"{metrics['p95_m']:.4f} <= {max_p95_m:.4f}",
            )
        )
    if min_recall is not None:
        checks.append(
            _check(
                "geo_recall",
                "pass" if metrics["recall"] >= min_recall else "fail",
                f"{metrics['recall']:.4f} >= {min_recall:.4f}",
            )
        )

    report = {
        "created_at": utc_now_iso(),
        "status": _overall_status(checks),
        "predictions_geojson": str(predictions_geojson),
        "truth_geojson": str(truth_geojson),
        "thresholds": {
            "tolerance_m": tolerance_m,
            "max_rmse_m": max_rmse_m,
            "max_p95_m": max_p95_m,
            "min_recall": min_recall,
        },
        "metrics": metrics,
        "checks": checks,
        "matches": matches,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_truth": unmatched_truth,
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_matches_csv(report, output_path.with_suffix(".csv"))
    return report


def _read_points(path: str | Path, require_points: bool) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features", []) if isinstance(payload, dict) else []
    points = []
    for index, feature in enumerate(features):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        properties = feature.get("properties") or {}
        points.append(
            {
                "index": index,
                "id": properties.get("id") or properties.get("plant_id") or f"point-{index + 1}",
                "x": float(coordinates[0]),
                "y": float(coordinates[1]),
                "crs": properties.get("crs"),
                "score": float(properties.get("score", 1.0)),
            }
        )
    if require_points and not points:
        raise ValueError(f"No point features found in GeoJSON: {path}")
    return points


def _match_points(
    predictions: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    tolerance_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matched_truth: set[int] = set()
    matches: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        best_truth = None
        best_distance = float("inf")
        for truth_point in truth:
            if int(truth_point["index"]) in matched_truth:
                continue
            distance = _distance_m(prediction, truth_point)
            if distance < best_distance:
                best_distance = distance
                best_truth = truth_point
        if best_truth is not None and best_distance <= tolerance_m:
            matched_truth.add(int(best_truth["index"]))
            matches.append(
                {
                    "prediction_id": prediction["id"],
                    "truth_id": best_truth["id"],
                    "prediction": [prediction["x"], prediction["y"]],
                    "truth": [best_truth["x"], best_truth["y"]],
                    "distance_m": round(best_distance, 6),
                }
            )
        else:
            unmatched_predictions.append(_point_summary(prediction))
    unmatched_truth = [_point_summary(point) for point in truth if int(point["index"]) not in matched_truth]
    return matches, unmatched_predictions, unmatched_truth


def _metrics(
    prediction_count: int,
    truth_count: int,
    matched_count: int,
    distances: list[float],
) -> dict[str, Any]:
    precision = matched_count / max(1, prediction_count)
    recall = matched_count / max(1, truth_count)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "prediction_count": prediction_count,
        "truth_count": truth_count,
        "matched_count": matched_count,
        "unmatched_prediction_count": prediction_count - matched_count,
        "unmatched_truth_count": truth_count - matched_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_error_m": _mean(distances),
        "median_error_m": median(distances) if distances else 0.0,
        "rmse_m": _rmse(distances),
        "p95_m": _percentile(distances, 0.95),
        "max_error_m": max(distances) if distances else 0.0,
    }


def _distance_m(left: dict[str, Any], right: dict[str, Any]) -> float:
    if _looks_like_lonlat(left["x"], left["y"]) and _looks_like_lonlat(right["x"], right["y"]):
        return _haversine_m(left["x"], left["y"], right["x"], right["y"])
    return sqrt((left["x"] - right["x"]) ** 2 + (left["y"] - right["y"]) ** 2)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * atan2(sqrt(a), sqrt(1 - a))


def _looks_like_lonlat(x: float, y: float) -> bool:
    return -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rmse(values: list[float]) -> float:
    return sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _point_summary(point: dict[str, Any]) -> dict[str, Any]:
    return {"id": point["id"], "coordinates": [point["x"], point["y"]], "crs": point.get("crs")}


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _overall_status(checks: list[dict[str, str]]) -> str:
    return "fail" if any(check["status"] == "fail" for check in checks) else "pass"


def _write_matches_csv(report: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prediction_id", "truth_id", "distance_m"])
        writer.writeheader()
        for match in report.get("matches", []):
            writer.writerow(
                {
                    "prediction_id": match.get("prediction_id", ""),
                    "truth_id": match.get("truth_id", ""),
                    "distance_m": match.get("distance_m", ""),
                }
            )


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
