import json
from pathlib import Path

import pytest

from bananavision.models import InferenceConfig
from bananavision.registry import promote_model, register_model


def test_register_model(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"pretend weights")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"passed": True}), encoding="utf-8")
    config = InferenceConfig(detector="yolo-seg", model_path=str(model))
    manifest = register_model(model, tmp_path / "registry", "v1.0.0", config, acceptance_report=acceptance)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["version"] == "v1.0.0"
    assert payload["model_sha256"]
    assert payload["acceptance_report"]["passed"] is True
    assert (tmp_path / "registry" / "latest.json").exists()


def test_promote_model_requires_passing_gates(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"pretend weights")
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    acceptance.write_text(json.dumps({"passed": True}), encoding="utf-8")
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 42.0}}), encoding="utf-8")
    config = InferenceConfig(detector="yolo-seg", model_path=str(model))
    manifest = promote_model(
        model,
        tmp_path / "registry",
        "production-v1",
        config,
        acceptance_report=acceptance,
        benchmark_report=benchmark,
        max_p95_ms=50,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["promotion"]["status"] == "promoted"
    assert all(gate["status"] == "pass" for gate in payload["promotion"]["gates"])


def test_promote_model_rejects_failed_acceptance(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"pretend weights")
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    acceptance.write_text(json.dumps({"passed": False}), encoding="utf-8")
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 42.0}}), encoding="utf-8")
    config = InferenceConfig(detector="yolo-seg", model_path=str(model))
    with pytest.raises(ValueError, match="acceptance_passed"):
        promote_model(model, tmp_path / "registry", "bad-v1", config, acceptance, benchmark)


def test_promote_model_rejects_slow_benchmark(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"pretend weights")
    acceptance = tmp_path / "acceptance.json"
    benchmark = tmp_path / "benchmark.json"
    acceptance.write_text(json.dumps({"passed": True}), encoding="utf-8")
    benchmark.write_text(json.dumps({"latency_ms": {"p95": 150.0}}), encoding="utf-8")
    config = InferenceConfig(detector="yolo-seg", model_path=str(model))
    with pytest.raises(ValueError, match="benchmark_p95_limit"):
        promote_model(model, tmp_path / "registry", "slow-v1", config, acceptance, benchmark, max_p95_ms=100)
