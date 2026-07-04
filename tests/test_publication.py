import json
from pathlib import Path

from typer.testing import CliRunner

from bananavision.cli import app
from bananavision.publication import REQUIRED_PATHS, audit_publication

runner = CliRunner()


def test_audit_publication_passes_complete_repo(tmp_path: Path) -> None:
    _write_publication_fixture(tmp_path)

    report = audit_publication(tmp_path, tmp_path / "publication_audit.json")

    assert report["status"] == "pass"
    assert (tmp_path / "publication_audit.json").exists()
    assert all(check["status"] == "pass" for check in report["checks"])


def test_audit_publication_fails_missing_required_files(tmp_path: Path) -> None:
    _write_publication_fixture(tmp_path)
    (tmp_path / "SECURITY.md").unlink()

    report = audit_publication(tmp_path)

    failed = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert report["status"] == "fail"
    assert "required:SECURITY.md" in failed


def test_publication_audit_cli_writes_report(tmp_path: Path) -> None:
    _write_publication_fixture(tmp_path)
    output = tmp_path / "runs" / "publication.json"

    result = runner.invoke(app, ["publication-audit", str(tmp_path), "--output", str(output)])

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def _write_publication_fixture(root: Path) -> None:
    for relative in REQUIRED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_content_for(relative), encoding="utf-8")


def _content_for(relative: str) -> str:
    if relative == "pyproject.toml":
        return """
[project]
name = "bananavision-drone"
version = "0.1.0"
license = { text = "Apache-2.0" }

[project.scripts]
bananavision = "bananavision.cli:app"
"""
    if relative == ".github/workflows/ci.yml":
        return """
name: CI
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - run: python -m ruff check .
      - run: python -m pytest -q
"""
    if relative == "README.md":
        return "validation-plan holdout-lock stratified-truth-coverage stratified-acceptance truth-quality truth-coverage cluster-benchmark cluster-review flight-check flight-log-audit capture-coverage mission-audit domain-check release-audit evidence-manifest release-package release-package-verify deployment-smoke-test drone-ready deployment-audit\n"
    if relative == "Dockerfile":
        return "FROM python:3.11\nHEALTHCHECK CMD python -c \"print('/ready')\"\n"
    return f"# {relative}\n"
