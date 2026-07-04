from pathlib import Path

from bananavision.models import InferenceConfig
from bananavision.runtime import file_sha256, runtime_fingerprint, stable_json_hash


def test_file_sha256(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("banana", encoding="utf-8")
    assert file_sha256(path) == "b493d48364afe44d11c0165cf470a4164d1e2609911ef998be868d46ade3de4e"


def test_stable_json_hash_ignores_key_order() -> None:
    assert stable_json_hash({"a": 1, "b": 2}) == stable_json_hash({"b": 2, "a": 1})


def test_runtime_fingerprint_has_config_hash() -> None:
    fingerprint = runtime_fingerprint(InferenceConfig())
    assert fingerprint["config_sha256"]
    assert fingerprint["model"]["detector"] == "rgb-canopy"
