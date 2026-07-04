import pytest

from bananavision.models import InferenceConfig


def test_config_rejects_invalid_tile_overlap() -> None:
    with pytest.raises(ValueError, match="tile_overlap"):
        InferenceConfig.from_mapping({"tile_size": 128, "tile_overlap": 128})


def test_config_requires_model_for_yolo() -> None:
    with pytest.raises(ValueError, match="model_path"):
        InferenceConfig.from_mapping({"detector": "yolo-seg"})


def test_config_rejects_invalid_max_split_instances() -> None:
    with pytest.raises(ValueError, match="max_split_instances"):
        InferenceConfig.from_mapping({"max_split_instances": 0})


def test_config_rejects_invalid_center_distance_weight() -> None:
    with pytest.raises(ValueError, match="center_distance_weight"):
        InferenceConfig.from_mapping({"center_distance_weight": 1.2})
