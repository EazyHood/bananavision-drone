import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.domain import (
    DomainProfileSettings,
    DomainShiftThresholds,
    audit_domain_shift,
    build_domain_profile,
)

runner = CliRunner()


def test_build_domain_profile_and_check_passes_similar_images(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    mission = tmp_path / "mission"
    reference.mkdir()
    mission.mkdir()
    _write_image(reference / "ref1.jpg", (70, 145, 80))
    _write_image(reference / "ref2.jpg", (75, 150, 82))
    _write_image(mission / "mission.jpg", (72, 148, 80))

    profile = build_domain_profile(reference, tmp_path / "domain_profile.json")
    report = audit_domain_shift(mission, tmp_path / "domain_profile.json", tmp_path / "domain_check.json")

    assert profile["image_count"] == 2
    assert report["status"] == "pass"
    assert report["outlier_count"] == 0
    assert (tmp_path / "domain_check.csv").exists()


def test_domain_check_fails_visually_different_images(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    mission = tmp_path / "mission"
    reference.mkdir()
    mission.mkdir()
    _write_image(reference / "ref.jpg", (70, 145, 80))
    _write_image(mission / "mission.jpg", (210, 60, 45))

    build_domain_profile(reference, tmp_path / "domain_profile.json")
    report = audit_domain_shift(
        mission,
        tmp_path / "domain_profile.json",
        thresholds=DomainShiftThresholds(max_histogram_distance=0.10, max_feature_z=3.0),
    )

    assert report["status"] == "fail"
    assert report["outlier_count"] == 1
    assert report["images"][0]["issues"]


def test_domain_profile_cli_and_domain_check_cli(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    mission = tmp_path / "mission"
    profile_json = tmp_path / "profile.json"
    check_json = tmp_path / "check.json"
    reference.mkdir()
    mission.mkdir()
    _write_image(reference / "ref.jpg", (70, 145, 80))
    _write_image(mission / "mission.jpg", (70, 145, 80))

    profile_result = runner.invoke(
        app,
        ["domain-profile", str(reference), "--output", str(profile_json), "--bins", "8"],
    )
    check_result = runner.invoke(
        app,
        ["domain-check", str(mission), str(profile_json), "--output", str(check_json)],
    )

    assert profile_result.exit_code == 0
    assert check_result.exit_code == 0
    assert json.loads(check_json.read_text(encoding="utf-8"))["status"] == "pass"


def test_domain_profile_rejects_too_few_bins(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    _write_image(image, (70, 145, 80))

    try:
        build_domain_profile(image, settings=DomainProfileSettings(bins=2))
    except ValueError as exc:
        assert "bins" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (96, 80), color).save(path)
