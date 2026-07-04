from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bananavision.api import create_app
from bananavision.synthetic import generate_scene


def test_api_health_ready_and_infer(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "truth.json"
    generate_scene(image, truth, width=180, height=140, plant_count=4)
    app = create_app()
    client = TestClient(app)

    health = client.get("/health", headers={"X-Request-ID": "req-health"})
    ready = client.get("/ready", headers={"X-Request-ID": "req-ready"})
    with image.open("rb") as handle:
        infer = client.post(
            "/infer",
            headers={"X-Request-ID": "req-infer"},
            files={"file": ("scene.jpg", handle, "image/jpeg")},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["schema_version"] == 1
    assert health.json()["api_version"] == "v1"
    assert health.json()["request_id"] == "req-health"
    assert health.headers["X-Request-ID"] == "req-health"
    assert ready.status_code == 200
    payload = ready.json()
    assert payload["status"] == "ready"
    assert payload["schema_version"] == 1
    assert payload["api_version"] == "v1"
    assert payload["request_id"] == "req-ready"
    assert payload["runtime"]["config"]["detector"] == "rgb-canopy"
    assert payload["runtime"]["config_sha256"]
    assert payload["api"]["auth_required"] is False
    assert payload["api"]["max_upload_mb"] == 25.0
    assert payload["api"]["max_upload_bytes"] == 25 * 1024 * 1024
    assert infer.status_code == 200
    infer_payload = infer.json()
    assert infer_payload["status"] == "ok"
    assert infer_payload["api_version"] == "v1"
    assert infer_payload["request_id"] == "req-infer"
    assert infer_payload["count"] >= 0
    assert infer_payload["result"]["count"] == infer_payload["count"]
    assert infer_payload["upload"]["filename"] == "scene.jpg"


def test_api_infer_requires_key_when_configured(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "truth.json"
    generate_scene(image, truth, width=180, height=140, plant_count=4)
    app = create_app(api_key="secret")
    client = TestClient(app)

    with image.open("rb") as handle:
        unauthorized = client.post("/infer", files={"file": ("scene.jpg", handle, "image/jpeg")})
    with image.open("rb") as handle:
        authorized = client.post(
            "/infer",
            headers={"X-API-Key": "secret"},
            files={"file": ("scene.jpg", handle, "image/jpeg")},
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["status"] == "error"
    assert unauthorized.json()["error"]["code"] == "unauthorized"
    assert authorized.status_code == 200


def test_api_accepts_bearer_token_when_configured(tmp_path: Path) -> None:
    image = tmp_path / "scene.jpg"
    truth = tmp_path / "truth.json"
    generate_scene(image, truth, width=180, height=140, plant_count=4)
    app = create_app(api_key="secret")
    client = TestClient(app)

    with image.open("rb") as handle:
        response = client.post(
            "/infer",
            headers={"Authorization": "Bearer secret"},
            files={"file": ("scene.jpg", handle, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_rejects_oversized_upload() -> None:
    app = create_app(max_upload_mb=0.000001)
    client = TestClient(app)

    response = client.post("/infer", files={"file": ("scene.jpg", b"too large", "image/jpeg")})

    assert response.status_code == 413
    assert response.json()["status"] == "error"
    assert response.json()["error"]["code"] == "payload_too_large"


def test_api_rejects_unsupported_suffix() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post("/infer", files={"file": ("scene.txt", b"not an image", "text/plain")})

    assert response.status_code == 415
    assert response.json()["status"] == "error"
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_api_returns_stable_validation_error_shape() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.post("/infer")

    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["details"]


def test_api_fails_fast_for_missing_yolo_model(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "detector: yolo-seg\nmodel_path: missing.pt\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="missing.pt|Model|Ultralytics"):
        create_app(config)
