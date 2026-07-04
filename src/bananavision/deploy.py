from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from posixpath import join as posix_join
from shlex import quote
from typing import Any

from . import __version__
from .runtime import utc_now_iso


@dataclass(frozen=True)
class SystemdArtifacts:
    files: list[str]
    output_dir: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
        }


def write_systemd_units(
    output_dir: str | Path,
    install_dir: str | Path,
    user: str,
    bananavision_bin: str = "bananavision",
    config_path: str | Path = "configs/banana_uav.yaml",
    detector: str = "rgb-canopy",
    model_path: str | Path | None = None,
    watch_dir: str | Path = "/data/mission/incoming",
    mission_output_dir: str | Path = "/data/mission/output",
    api_host: str = "0.0.0.0",
    api_port: int = 8080,
    environment_file: str | Path | None = None,
) -> SystemdArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "bananavision-mission-watch.service": render_mission_watch_service(
            install_dir=install_dir,
            user=user,
            bananavision_bin=bananavision_bin,
            config_path=config_path,
            detector=detector,
            model_path=model_path,
            watch_dir=watch_dir,
            mission_output_dir=mission_output_dir,
            environment_file=environment_file,
        ),
        "bananavision-api.service": render_api_service(
            install_dir=install_dir,
            user=user,
            bananavision_bin=bananavision_bin,
            config_path=config_path,
            api_host=api_host,
            api_port=api_port,
            environment_file=environment_file,
        ),
        "README.systemd.md": render_systemd_readme(),
    }
    manifest = build_deployment_manifest(
        install_dir=install_dir,
        user=user,
        bananavision_bin=bananavision_bin,
        config_path=config_path,
        detector=detector,
        model_path=model_path,
        watch_dir=watch_dir,
        mission_output_dir=mission_output_dir,
        api_host=api_host,
        api_port=api_port,
        environment_file=environment_file,
        generated_files=[*files, "deployment_manifest.json"],
    )
    files["deployment_manifest.json"] = json.dumps(manifest, indent=2)
    written = []
    for name, content in files.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(str(path))
    return SystemdArtifacts(
        files=written,
        output_dir=str(output_dir),
        manifest_path=str(output_dir / "deployment_manifest.json"),
    )


def build_deployment_manifest(
    install_dir: str | Path,
    user: str,
    bananavision_bin: str,
    config_path: str | Path,
    detector: str,
    model_path: str | Path | None,
    watch_dir: str | Path,
    mission_output_dir: str | Path,
    api_host: str,
    api_port: int,
    environment_file: str | Path | None = None,
    generated_files: list[str] | None = None,
) -> dict[str, Any]:
    model_arg = "" if model_path is None else f" --model {quote(_target_path(model_path))}"
    config_arg = f"--config {quote(_target_path(config_path))} --detector {quote(detector)}{model_arg}"
    preflight_report = _target_join(_target_path(mission_output_dir), "preflight_report.json")
    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "bananavision_version": __version__,
        "target": {
            "install_dir": _target_path(install_dir),
            "user": user,
            "environment_file": None if environment_file is None else _target_path(environment_file),
        },
        "model": {
            "detector": detector,
            "config_path": _target_path(config_path),
            "model_path": None if model_path is None else _target_path(model_path),
        },
        "paths": {
            "watch_dir": _target_path(watch_dir),
            "mission_output_dir": _target_path(mission_output_dir),
            "preflight_report": preflight_report,
        },
        "services": {
            "mission_watch": {
                "unit": "bananavision-mission-watch.service",
                "preflight_command": (
                    f"{quote(bananavision_bin)} preflight --input {quote(_target_path(watch_dir))} "
                    f"--output {quote(preflight_report)} {config_arg}"
                ),
                "start_command": (
                    f"{quote(bananavision_bin)} mission-watch {quote(_target_path(watch_dir))} "
                    f"--output {quote(_target_path(mission_output_dir))} {config_arg}"
                ),
                "health_commands": [
                    "systemctl is-active bananavision-mission-watch.service",
                    "journalctl -u bananavision-mission-watch.service -n 100 --no-pager",
                ],
            },
            "api": {
                "unit": "bananavision-api.service",
                "start_command": _api_start_command(bananavision_bin, config_path, api_host, api_port),
                "health_commands": [
                    "systemctl is-active bananavision-api.service",
                    f"curl -fsS http://127.0.0.1:{api_port}/health",
                    f"curl -fsS http://127.0.0.1:{api_port}/ready",
                    "journalctl -u bananavision-api.service -n 100 --no-pager",
                ],
            },
        },
        "operational_gates": [
            "Run the generated preflight command before flight.",
            "Use only a model registry manifest that passed locked holdout acceptance.",
            "Run benchmark on the target drone computer after installing the exported model.",
            "Review prediction-quality flags before publishing counts or updating inventory.",
        ],
        "generated_files": generated_files or [],
    }


def render_mission_watch_service(
    install_dir: str | Path,
    user: str,
    bananavision_bin: str,
    config_path: str | Path,
    detector: str,
    model_path: str | Path | None,
    watch_dir: str | Path,
    mission_output_dir: str | Path,
    environment_file: str | Path | None = None,
) -> str:
    config_args = _config_args(config_path, detector, model_path)
    preflight_report = _target_join(_target_path(mission_output_dir), "preflight_report.json")
    preflight = (
        f"{quote(bananavision_bin)} preflight --input {quote(_target_path(watch_dir))} "
        f"--output {quote(preflight_report)} {config_args}"
    )
    start = (
        f"{quote(bananavision_bin)} mission-watch {quote(_target_path(watch_dir))} "
        f"--output {quote(_target_path(mission_output_dir))} {config_args}"
    )
    env = _environment_file_line(environment_file)
    hardening = _service_hardening(read_write_paths=[watch_dir, mission_output_dir])
    return f"""[Unit]
Description=BananaVision mission-watch
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={_target_path(install_dir)}
Environment=PYTHONUNBUFFERED=1
{env}ExecStartPre={preflight}
ExecStart={start}
Restart=always
RestartSec=5
TimeoutStopSec=30
KillSignal=SIGINT
{hardening}

[Install]
WantedBy=multi-user.target
"""


def render_api_service(
    install_dir: str | Path,
    user: str,
    bananavision_bin: str,
    config_path: str | Path,
    api_host: str,
    api_port: int,
    environment_file: str | Path | None = None,
) -> str:
    env = _environment_file_line(environment_file)
    start = _api_start_command(bananavision_bin, config_path, api_host, api_port)
    hardening = _service_hardening()
    return f"""[Unit]
Description=BananaVision API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
WorkingDirectory={_target_path(install_dir)}
Environment=PYTHONUNBUFFERED=1
{env}ExecStart={start}
Restart=always
RestartSec=5
{hardening}

[Install]
WantedBy=multi-user.target
"""


def _api_start_command(
    bananavision_bin: str,
    config_path: str | Path,
    api_host: str,
    api_port: int,
) -> str:
    return (
        f"{quote(bananavision_bin)} serve --config {quote(_target_path(config_path))} "
        f"--host {quote(api_host)} --port {api_port} "
        "--api-key-env BANANAVISION_API_KEY --max-upload-mb 25"
    )


def render_systemd_readme() -> str:
    return """# BananaVision systemd deployment

Copy the generated service files to `/etc/systemd/system/`, then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable bananavision-mission-watch.service
sudo systemctl start bananavision-mission-watch.service
sudo systemctl status bananavision-mission-watch.service
```

For the API:

```bash
sudo systemctl enable bananavision-api.service
sudo systemctl start bananavision-api.service
```

Check logs:

```bash
journalctl -u bananavision-mission-watch.service -f
journalctl -u bananavision-api.service -f
```

Health checks:

```bash
systemctl is-active bananavision-mission-watch.service
systemctl is-active bananavision-api.service
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
```

`deployment_manifest.json` records the exact generated commands, target paths,
service names, and operational gates that must pass before flight.
"""


def _config_args(config_path: str | Path, detector: str, model_path: str | Path | None) -> str:
    args = f"--config {quote(_target_path(config_path))} --detector {quote(detector)}"
    if model_path:
        args += f" --model {quote(_target_path(model_path))}"
    return args


def _environment_file_line(environment_file: str | Path | None) -> str:
    if environment_file is None:
        return ""
    return f"EnvironmentFile={_target_path(environment_file)}\n"


def _target_join(base: str, child: str) -> str:
    return posix_join(base.rstrip("/"), child)


def _target_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _service_hardening(read_write_paths: list[str | Path] | None = None) -> str:
    lines = [
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "ProtectSystem=full",
        "ProtectHome=read-only",
        "RestrictSUIDSGID=true",
    ]
    if read_write_paths:
        paths = " ".join(_target_path(path) for path in read_write_paths)
        lines.append(f"ReadWritePaths={paths}")
    return "\n".join(lines)
