from pathlib import Path

from PIL import Image

from bananavision.geo import (
    describe_image_georeference,
    has_image_georeference,
    load_geotransform,
    read_exif_geotag,
    read_world_file,
)


def test_read_world_file(tmp_path: Path) -> None:
    world = tmp_path / "image.jgw"
    world.write_text("2\n0\n0\n-2\n100\n200\n", encoding="utf-8")
    transform = read_world_file(world)
    point = transform.pixel_to_world(10, 5)
    assert point.x == 120
    assert point.y == 190


def test_load_geotransform_finds_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    (tmp_path / "image.jgw").write_text("1\n0\n0\n-1\n10\n20\n", encoding="utf-8")
    transform = load_geotransform(image)
    assert transform is not None
    assert transform.pixel_to_world(3, 4).x == 13


def test_read_exif_geotag(tmp_path: Path) -> None:
    image = tmp_path / "geotagged.jpg"
    _write_exif_geotagged_image(image)

    geotag = read_exif_geotag(image)
    assert geotag is not None
    assert round(geotag.latitude, 6) == 4.6
    assert round(geotag.longitude, 6) == -74.066667
    assert geotag.altitude_m == 123.0
    assert has_image_georeference(image)
    description = describe_image_georeference(image)
    assert description is not None
    assert description["type"] == "exif_gps"


def _write_exif_geotagged_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), (80, 160, 90))
    exif = Image.Exif()
    exif[34853] = {
        1: "N",
        2: (4.0, 36.0, 0.0),
        3: "W",
        4: (74.0, 4.0, 0.0),
        5: 0,
        6: 123.0,
    }
    image.save(path, exif=exif)
