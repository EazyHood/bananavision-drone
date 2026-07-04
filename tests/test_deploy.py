import json
from pathlib import Path

from bananavision.deploy import (
    build_deployment_manifest,
    render_mission_watch_service,
    write_systemd_units,
)


def test_render_mission_watch_service_contains_preflight() -> None:
    service = render_mission_watch_service(
        install_dir="/opt/bananavision-drone",
        user="banana",
        bananavision_bin="/opt/bananavision-drone/.venv/bin/bananavision",
        config_path="/opt/bananavision-drone/configs/banana_uav.yaml",
        detector="yolo-seg",
        model_path="/models/best.pt",
        watch_dir="/data/incoming",
        mission_output_dir="/data/output",
    )
    assert "ExecStartPre=" in service
    assert "preflight" in service
    assert "mission-watch" in service
    assert "--model /models/best.pt" in service
    assert "\\models\\best.pt" not in service
    assert "/data/output/preflight_report.json" in service
    assert "Restart=always" in service
    assert "NoNewPrivileges=true" in service
    assert "ProtectSystem=full" in service
    assert "ReadWritePaths=/data/incoming /data/output" in service


def test_write_systemd_units(tmp_path: Path) -> None:
    artifacts = write_systemd_units(
        output_dir=tmp_path,
        install_dir="/opt/bananavision-drone",
        user="banana",
        config_path="/opt/bananavision-drone/configs/banana_uav.yaml",
        model_path="/models/best.pt",
    )
    files = {Path(file).name for file in artifacts.files}
    assert "bananavision-mission-watch.service" in files
    assert "bananavision-api.service" in files
    assert "README.systemd.md" in files
    assert "deployment_manifest.json" in files
    assert artifacts.manifest_path == str(tmp_path / "deployment_manifest.json")
    assert (tmp_path / "bananavision-api.service").read_text(encoding="utf-8").count("ExecStart=") == 1

    manifest = json.loads((tmp_path / "deployment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["model"]["model_path"] == "/models/best.pt"
    assert manifest["paths"]["preflight_report"] == "/data/mission/output/preflight_report.json"
    assert "deployment_manifest.json" in manifest["generated_files"]
    assert "curl -fsS http://127.0.0.1:8080/health" in manifest["services"]["api"]["health_commands"]
    assert "curl -fsS http://127.0.0.1:8080/ready" in manifest["services"]["api"]["health_commands"]
    service = (tmp_path / "bananavision-api.service").read_text(encoding="utf-8")
    assert "--api-key-env BANANAVISION_API_KEY" in service
    assert "--max-upload-mb 25" in service


def test_build_deployment_manifest_normalizes_target_paths() -> None:
    manifest = build_deployment_manifest(
        install_dir="C:\\opt\\bananavision-drone",
        user="banana",
        bananavision_bin="/opt/bananavision-drone/.venv/bin/bananavision",
        config_path="C:\\opt\\bananavision-drone\\configs\\banana_uav.yaml",
        detector="yolo-seg",
        model_path="C:\\models\\best.pt",
        watch_dir="C:\\data\\incoming",
        mission_output_dir="C:\\data\\output",
        api_host="0.0.0.0",
        api_port=9090,
        environment_file="C:\\etc\\bananavision.env",
    )

    encoded = json.dumps(manifest)
    assert "\\\\" not in encoded
    assert manifest["paths"]["watch_dir"] == "C:/data/incoming"
    assert manifest["target"]["environment_file"] == "C:/etc/bananavision.env"
    assert "curl -fsS http://127.0.0.1:9090/health" in manifest["services"]["api"]["health_commands"]
    assert "curl -fsS http://127.0.0.1:9090/ready" in manifest["services"]["api"]["health_commands"]
    assert "--api-key-env BANANAVISION_API_KEY" in manifest["services"]["api"]["start_command"]
