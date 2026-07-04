import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.geo_accuracy import audit_geo_accuracy

runner = CliRunner()


def test_audit_geo_accuracy_passes_projected_coordinates(tmp_path: Path) -> None:
    predictions = _write_geojson(
        tmp_path / "predictions.geojson",
        [
            ("pred-1", (100.2, 200.1), 0.99),
            ("pred-2", (105.1, 205.1), 0.95),
        ],
    )
    truth = _write_geojson(
        tmp_path / "truth.geojson",
        [
            ("truth-1", (100.0, 200.0), 1.0),
            ("truth-2", (105.0, 205.0), 1.0),
        ],
    )

    report = audit_geo_accuracy(
        predictions,
        truth,
        tmp_path / "geo_accuracy.json",
        tolerance_m=1.0,
        max_rmse_m=0.5,
        max_p95_m=0.5,
        min_recall=1.0,
    )

    assert report["status"] == "pass"
    assert report["metrics"]["matched_count"] == 2
    assert report["metrics"]["rmse_m"] < 0.5
    assert (tmp_path / "geo_accuracy.csv").exists()


def test_audit_geo_accuracy_uses_haversine_for_lonlat(tmp_path: Path) -> None:
    predictions = _write_geojson(tmp_path / "predictions.geojson", [("pred-1", (-74.000005, 4.0), 0.9)])
    truth = _write_geojson(tmp_path / "truth.geojson", [("truth-1", (-74.0, 4.0), 1.0)])

    report = audit_geo_accuracy(
        predictions,
        truth,
        tolerance_m=2.0,
        max_rmse_m=1.0,
        min_recall=1.0,
    )

    assert report["status"] == "pass"
    assert 0.4 < report["metrics"]["rmse_m"] < 0.7


def test_audit_geo_accuracy_fails_high_location_error(tmp_path: Path) -> None:
    predictions = _write_geojson(tmp_path / "predictions.geojson", [("pred-1", (102.0, 200.0), 0.9)])
    truth = _write_geojson(tmp_path / "truth.geojson", [("truth-1", (100.0, 200.0), 1.0)])

    report = audit_geo_accuracy(
        predictions,
        truth,
        tolerance_m=5.0,
        max_rmse_m=1.0,
        min_recall=1.0,
    )

    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert report["status"] == "fail"
    assert "matched_rmse" in failed


def test_audit_geo_accuracy_fails_empty_predictions(tmp_path: Path) -> None:
    predictions = _write_geojson(tmp_path / "predictions.geojson", [])
    truth = _write_geojson(tmp_path / "truth.geojson", [("truth-1", (100.0, 200.0), 1.0)])

    report = audit_geo_accuracy(
        predictions,
        truth,
        tolerance_m=1.0,
        max_rmse_m=1.0,
        min_recall=1.0,
    )

    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert report["status"] == "fail"
    assert "prediction_support" in failed
    assert "matched_support" in failed
    assert "geo_recall" in failed


def test_geo_accuracy_cli_writes_report(tmp_path: Path) -> None:
    predictions = _write_geojson(tmp_path / "predictions.geojson", [("pred-1", (100.2, 200.1), 0.99)])
    truth = _write_geojson(tmp_path / "truth.geojson", [("truth-1", (100.0, 200.0), 1.0)])
    output = tmp_path / "geo_accuracy.json"

    result = runner.invoke(
        app,
        [
            "geo-accuracy",
            str(predictions),
            str(truth),
            "--output",
            str(output),
            "--max-rmse-m",
            "0.5",
            "--max-p95-m",
            "0.5",
            "--min-recall",
            "1.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert output.with_suffix(".csv").exists()


def _write_geojson(path: Path, points: list[tuple[str, tuple[float, float], float]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": point_id, "score": score},
                        "geometry": {"type": "Point", "coordinates": [coordinates[0], coordinates[1]]},
                    }
                    for point_id, coordinates, score in points
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
