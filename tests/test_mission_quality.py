from pathlib import Path

from PIL import Image, ImageFilter

from bananavision.mission_quality import MissionQualityThresholds, audit_mission_images


def test_audit_mission_images_flags_bad_capture(tmp_path: Path) -> None:
    good = tmp_path / "good.png"
    dark = tmp_path / "dark.png"
    _write_checker(good)
    _write_world_file(good.with_suffix(".pgw"))
    Image.new("RGB", (128, 128), (0, 0, 0)).save(dark)
    _write_world_file(dark.with_suffix(".pgw"))

    report = audit_mission_images(
        tmp_path,
        tmp_path / "quality" / "mission_quality_report.json",
        MissionQualityThresholds(
            min_width=64,
            min_height=64,
            min_focus_score=10.0,
            max_dark_fraction=0.35,
            require_georef=True,
        ),
    )

    assert report["status"] == "fail"
    assert report["image_count"] == 2
    assert report["pass_count"] == 1
    assert report["fail_count"] == 1
    assert (tmp_path / "quality" / "mission_quality_report.json").exists()
    assert (tmp_path / "quality" / "mission_quality_report.csv").exists()
    dark_row = next(row for row in report["images"] if row["image"].endswith("dark.png"))
    assert any("mean_luma below" in issue for issue in dark_row["issues"])


def test_audit_mission_images_warns_when_georef_is_optional(tmp_path: Path) -> None:
    image = tmp_path / "nogeo.png"
    _write_checker(image)

    report = audit_mission_images(
        image,
        tmp_path / "mission_quality_report.json",
        MissionQualityThresholds(min_width=64, min_height=64, min_focus_score=10.0),
    )

    assert report["status"] == "warn"
    assert report["warn_count"] == 1
    assert any("georeference missing" in issue for issue in report["images"][0]["issues"])


def test_audit_mission_images_accepts_exif_gps_when_georef_required(tmp_path: Path) -> None:
    image = tmp_path / "geotagged.jpg"
    _write_checker(image)
    _add_exif_geotag(image)

    report = audit_mission_images(
        image,
        tmp_path / "mission_quality_report.json",
        MissionQualityThresholds(min_width=64, min_height=64, min_focus_score=1.0, require_georef=True),
    )

    row = report["images"][0]
    assert report["status"] == "pass"
    assert row["georeferenced"] is True
    assert row["georeference_type"] == "exif_gps"
    assert round(row["exif_latitude"], 6) == 4.6
    assert round(row["exif_longitude"], 6) == -74.066667


def test_audit_mission_images_flags_blur(tmp_path: Path) -> None:
    image = tmp_path / "blurred.png"
    _write_checker(image)
    with Image.open(image) as original:
        original.filter(ImageFilter.GaussianBlur(radius=8)).save(image)
    _write_world_file(image.with_suffix(".pgw"))

    report = audit_mission_images(
        image,
        tmp_path / "mission_quality_report.json",
        MissionQualityThresholds(min_width=64, min_height=64, min_focus_score=1000.0, require_georef=True),
    )

    assert report["status"] == "fail"
    assert any("focus_score below" in issue for issue in report["images"][0]["issues"])


def _write_checker(path: Path) -> None:
    image = Image.new("RGB", (128, 128))
    pixels = image.load()
    for y in range(128):
        for x in range(128):
            value = 60 if ((x // 8) + (y // 8)) % 2 == 0 else 190
            pixels[x, y] = (value, value, value)
    image.save(path)


def _write_world_file(path: Path) -> None:
    path.write_text("0.02\n0\n0\n-0.02\n500000\n1000000\n", encoding="utf-8")


def _add_exif_geotag(path: Path) -> None:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    exif = Image.Exif()
    exif[34853] = {
        1: "N",
        2: (4.0, 36.0, 0.0),
        3: "W",
        4: (74.0, 4.0, 0.0),
        5: 0,
        6: 123.0,
    }
    rgb.save(path, exif=exif)
