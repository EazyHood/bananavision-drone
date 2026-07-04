import json
from pathlib import Path

from bananavision.holdout import lock_holdout, verify_holdout_lock
from bananavision.synthetic import generate_scene


def test_lock_and_verify_holdout_manifest(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "scene.jpg"
    truth = tmp_path / "scene.truth.json"
    generate_scene(image, truth, width=160, height=120, plant_count=4, seed=3)
    centers = json.loads(truth.read_text(encoding="utf-8"))["centers"]
    manifest = tmp_path / "truth_manifest.json"
    manifest.write_text(json.dumps({"images": [{"image": "scene.jpg", "centers": centers}]}), encoding="utf-8")

    lock = lock_holdout(image_dir, manifest, tmp_path / "holdout_lock.json", target_count_error_rate=0.25)
    assert lock["image_count"] == 1
    assert lock["truth_count"] == 4
    assert lock["cluster_count"] == 0
    assert lock["cluster_truth_count"] == 0
    assert lock["claim_resolution_ok"] is True
    assert lock["lock_sha256"]

    report = verify_holdout_lock(tmp_path / "holdout_lock.json", tmp_path / "verify.json")
    assert report["status"] == "pass"
    assert report["issue_count"] == 0
    assert report["cluster_count"] == 0
    assert (tmp_path / "verify.json").exists()


def test_verify_holdout_detects_truth_changes(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "scene.truth.json"
    generate_scene(image, truth, width=160, height=120, plant_count=4, seed=3)
    lock_holdout(image, truth, tmp_path / "holdout_lock.json", target_count_error_rate=0.25)
    truth.write_text(json.dumps({"centers": [[1, 2]]}), encoding="utf-8")

    report = verify_holdout_lock(tmp_path / "holdout_lock.json")
    issue_types = {issue["type"] for issue in report["issues"]}
    assert report["status"] == "fail"
    assert "truth_hash_changed" in issue_types
    assert "truth_count_changed" in issue_types


def test_verify_holdout_detects_cluster_annotation_changes(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "scene.truth.json"
    generate_scene(image, truth, width=160, height=120, plant_count=3, seed=3)
    truth.write_text(
        json.dumps(
            {
                "centers": [
                    {"x": 10, "y": 10, "group_id": "mat-a"},
                    {"x": 20, "y": 10, "group_id": "mat-a"},
                    {"x": 80, "y": 80},
                ]
            }
        ),
        encoding="utf-8",
    )
    lock = lock_holdout(image, truth, tmp_path / "holdout_lock.json", target_count_error_rate=0.25)
    truth.write_text(json.dumps({"centers": [[10, 10], [20, 10], [80, 80]]}), encoding="utf-8")

    report = verify_holdout_lock(tmp_path / "holdout_lock.json")
    issue_types = {issue["type"] for issue in report["issues"]}

    assert lock["cluster_count"] == 1
    assert lock["cluster_truth_count"] == 2
    assert report["status"] == "fail"
    assert "truth_hash_changed" in issue_types
    assert "truth_count_changed" not in issue_types
    assert "cluster_count_changed" in issue_types
    assert "cluster_truth_count_changed" in issue_types


def test_verify_holdout_detects_expected_path_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "scene.truth.json"
    generate_scene(image, truth, width=160, height=120, plant_count=4, seed=3)
    lock_holdout(image, truth, tmp_path / "holdout_lock.json", target_count_error_rate=0.25)

    report = verify_holdout_lock(
        tmp_path / "holdout_lock.json",
        expected_image_path=tmp_path / "other.jpg",
        expected_truth_path=truth,
    )

    issue_types = {issue["type"] for issue in report["issues"]}
    assert report["status"] == "fail"
    assert "image_path_mismatch" in issue_types
