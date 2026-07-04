from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .runtime import utc_now_iso


@dataclass(frozen=True)
class PublicationCheck:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "Dockerfile",
    ".github/workflows/ci.yml",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/OPERATOR_MANUAL.md",
    "docs/FAILURE_MODES.md",
    "docs/VALIDATION_PROTOCOL.md",
    "docs/DRONE_DEPLOYMENT.md",
    "docs/COMMERCIAL_READINESS.md",
]


def audit_publication(root: str | Path = ".", output_json: str | Path | None = None) -> dict[str, Any]:
    root = Path(root)
    checks: list[PublicationCheck] = []
    checks.extend(_check_required_paths(root))
    checks.append(_check_ci(root / ".github" / "workflows" / "ci.yml"))
    checks.append(_check_pyproject(root / "pyproject.toml"))
    checks.append(_check_readme(root / "README.md"))
    checks.append(_check_dockerfile(root / "Dockerfile"))

    report = {
        "created_at": utc_now_iso(),
        "root": str(root),
        "status": _overall_status(checks),
        "checks": [check.to_dict() for check in checks],
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _check_required_paths(root: Path) -> list[PublicationCheck]:
    checks = []
    for relative in REQUIRED_PATHS:
        path = root / relative
        checks.append(
            PublicationCheck(
                f"required:{relative}",
                "pass" if path.exists() else "fail",
                "present" if path.exists() else "missing",
            )
        )
    return checks


def _check_ci(path: Path) -> PublicationCheck:
    if not path.exists():
        return PublicationCheck("ci_workflow", "fail", "CI workflow missing")
    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) or {}
    except Exception as exc:
        return PublicationCheck("ci_workflow", "fail", f"CI workflow unreadable: {exc}")
    commands = text.lower()
    has_ruff = "ruff check" in commands
    has_pytest = "pytest" in commands
    has_matrix = "matrix" in json.dumps(payload).lower()
    if has_ruff and has_pytest and has_matrix:
        return PublicationCheck("ci_workflow", "pass", "runs ruff and pytest across a Python matrix")
    missing = []
    if not has_ruff:
        missing.append("ruff")
    if not has_pytest:
        missing.append("pytest")
    if not has_matrix:
        missing.append("python matrix")
    return PublicationCheck("ci_workflow", "fail", "missing " + ", ".join(missing))


def _check_pyproject(path: Path) -> PublicationCheck:
    if not path.exists():
        return PublicationCheck("pyproject_metadata", "fail", "pyproject.toml missing")
    text = path.read_text(encoding="utf-8")
    required = ["name = ", "version = ", "license = ", "[project.scripts]", "bananavision"]
    missing = [item for item in required if item not in text]
    if missing:
        return PublicationCheck("pyproject_metadata", "fail", "missing " + ", ".join(missing))
    return PublicationCheck("pyproject_metadata", "pass", "package metadata and CLI entry point present")


def _check_readme(path: Path) -> PublicationCheck:
    if not path.exists():
        return PublicationCheck("readme_publication_flow", "fail", "README.md missing")
    text = path.read_text(encoding="utf-8").lower()
    required = [
        "release-audit",
        "release-package",
        "release-package-verify",
        "evidence-manifest",
        "deployment-smoke-test",
        "drone-ready",
        "deployment-audit",
        "cluster-benchmark",
        "cluster-review",
        "validation-plan",
        "stratified-truth-coverage",
        "stratified-acceptance",
        "truth-quality",
        "truth-coverage",
        "holdout-lock",
        "flight-check",
        "flight-log-audit",
        "capture-coverage",
        "mission-audit",
        "domain-check",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        return PublicationCheck("readme_publication_flow", "fail", "missing " + ", ".join(missing))
    return PublicationCheck("readme_publication_flow", "pass", "README documents validation and release flow")


def _check_dockerfile(path: Path) -> PublicationCheck:
    if not path.exists():
        return PublicationCheck("docker_healthcheck", "fail", "Dockerfile missing")
    text = path.read_text(encoding="utf-8").lower()
    if "healthcheck" in text and "/ready" in text:
        return PublicationCheck("docker_healthcheck", "pass", "container checks /ready")
    return PublicationCheck("docker_healthcheck", "fail", "Dockerfile must healthcheck /ready")


def _overall_status(checks: list[PublicationCheck]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"
