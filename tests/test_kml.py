from pathlib import Path

from bananavision.io import write_kml
from bananavision.models import Detection, GeoPoint


def test_write_kml_with_lonlat(tmp_path: Path) -> None:
    detection = Detection(
        label="banana_plant",
        score=0.91,
        bbox=(0, 0, 10, 10),
        center=(5, 5),
        area_px=100,
        source="test",
        id="banana-1",
        geo=GeoPoint(-74.1, 4.6, "EPSG:4326"),
    )
    path = write_kml([detection], tmp_path / "detections.kml")
    text = path.read_text(encoding="utf-8")
    assert "<Placemark>" in text
    assert "-74.10000000,4.60000000,0" in text


def test_write_kml_skips_pixel_coordinates(tmp_path: Path) -> None:
    detection = Detection(
        label="banana_plant",
        score=0.91,
        bbox=(0, 0, 10, 10),
        center=(500, 500),
        area_px=100,
        source="test",
        id="banana-1",
    )
    path = write_kml([detection], tmp_path / "detections.kml")
    text = path.read_text(encoding="utf-8")
    assert "<Placemark>" not in text
    assert "Skipped non-lon/lat detections: 1" in text
