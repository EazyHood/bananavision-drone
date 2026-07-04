from bananavision.mission import deduplicate_detections
from bananavision.models import Detection, GeoPoint


def test_deduplicate_geo_detections() -> None:
    detections = [
        Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "a", geo=GeoPoint(100, 100)),
        Detection("banana_plant", 0.7, (0, 0, 10, 10), (5, 5), 100, "b", geo=GeoPoint(100.3, 100.2)),
        Detection("banana_plant", 0.8, (0, 0, 10, 10), (5, 5), 100, "c", geo=GeoPoint(105, 105)),
    ]
    deduped = deduplicate_detections(detections, geo_distance_m=1.2)
    assert len(deduped) == 2
    assert deduped[0].id == "mission-banana-00001"


def test_deduplicate_pixel_detections_without_geo() -> None:
    detections = [
        Detection("banana_plant", 0.9, (0, 0, 10, 10), (5, 5), 100, "a"),
        Detection("banana_plant", 0.7, (0, 0, 10, 10), (8, 8), 100, "b"),
        Detection("banana_plant", 0.8, (0, 0, 10, 10), (50, 50), 100, "c"),
    ]
    assert len(deduplicate_detections(detections, pixel_distance_px=8)) == 2
