import json
from pathlib import Path

from bananavision.truth import read_truth_centers, read_truth_points, truth_cluster_summary


def test_read_simple_truth_centers(tmp_path: Path) -> None:
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"centers": [{"x": 1, "y": 2}, [3, 4]]}), encoding="utf-8")
    assert read_truth_centers(truth) == [(1.0, 2.0), (3.0, 4.0)]


def test_read_image_specific_truth_centers(tmp_path: Path) -> None:
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps({"images": [{"image": "a.jpg", "centers": [[1, 2]]}, {"image": "b.jpg", "centers": [[3, 4]]}]}),
        encoding="utf-8",
    )
    assert read_truth_centers(truth, image="b.jpg") == [(3.0, 4.0)]


def test_read_truth_points_with_banana_group_ids(tmp_path: Path) -> None:
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "centers": [
                    {"x": 1, "y": 2, "group_id": "mat-a"},
                    {"x": 3, "y": 4, "cluster": "mat-a"},
                    [5, 6, "mat-b"],
                ]
            }
        ),
        encoding="utf-8",
    )

    points = read_truth_points(truth)

    assert [point.center for point in points] == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert [point.group_id for point in points] == ["mat-a", "mat-a", "mat-b"]
    assert truth_cluster_summary(points) == {"cluster_count": 1, "cluster_truth_count": 2}
