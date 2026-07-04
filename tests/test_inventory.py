import json
from pathlib import Path

import pytest

from bananavision.inventory import diff_inventories, update_inventory


def test_update_inventory_creates_and_updates_plants(tmp_path: Path) -> None:
    first = tmp_path / "first.geojson"
    second = tmp_path / "second.geojson"
    inventory_dir = tmp_path / "inventory"
    first.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    _feature("d1", 10.0, 10.0, "pixel", 0.9),
                    _feature("d2", 100.0, 100.0, "pixel", 0.8),
                ],
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    _feature("d3", 11.0, 10.0, "pixel", 1.0),
                    _feature("d4", 200.0, 200.0, "pixel", 0.7),
                ],
            }
        ),
        encoding="utf-8",
    )
    first_summary = update_inventory(first, inventory_dir, distance_threshold=2.0, observed_at="2026-01-01T00:00:00Z")
    second_summary = update_inventory(second, inventory_dir, distance_threshold=2.0, observed_at="2026-01-02T00:00:00Z")
    inventory = json.loads((inventory_dir / "inventory.json").read_text(encoding="utf-8"))
    assert first_summary["created"] == 2
    assert Path(first_summary["inventory_snapshot"]).exists()
    assert Path(first_summary["inventory_snapshot"]).name == "inventory_2026-01-01T00-00-00Z.json"
    assert second_summary["created"] == 1
    assert second_summary["updated"] == 1
    assert inventory["plant_count"] == 3
    updated = next(plant for plant in inventory["plants"] if plant["observations"] == 2)
    assert updated["plant_id"] == "banana-plant-000001"
    assert updated["x"] == 10.5
    assert (inventory_dir / "inventory.csv").exists()
    assert (inventory_dir / "inventory.geojson").exists()
    assert (inventory_dir / "inventory.kml").exists()

    diff = diff_inventories(first_summary["inventory_snapshot"], inventory_dir / "inventory.json", tmp_path / "diff")
    assert diff["new_count"] == 1
    assert diff["missing_count"] == 0
    assert diff["persistent_count"] == 2
    assert (tmp_path / "diff" / "inventory_diff.json").exists()
    assert (tmp_path / "diff" / "inventory_new.geojson").exists()

    reverse_diff = diff_inventories(inventory_dir / "inventory.json", first_summary["inventory_snapshot"], tmp_path / "reverse")
    assert reverse_diff["new_count"] == 0
    assert reverse_diff["missing_count"] == 1
    assert reverse_diff["missing_plants"][0]["plant_id"] == "banana-plant-000003"

    repeated_summary = update_inventory(first, inventory_dir, distance_threshold=2.0, observed_at="2026-01-02T00:00:00Z")
    assert Path(repeated_summary["inventory_snapshot"]).name == "inventory_2026-01-02T00-00-00Z_2.json"

    with pytest.raises(FileNotFoundError):
        diff_inventories(tmp_path / "missing.json", inventory_dir / "inventory.json", tmp_path / "bad-diff")


def _feature(detection_id: str, x: float, y: float, crs: str, score: float) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, y]},
        "properties": {
            "id": detection_id,
            "label": "banana_plant",
            "score": score,
            "source": "test",
            "crs": crs,
        },
    }
