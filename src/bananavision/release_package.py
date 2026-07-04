from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .runtime import file_sha256, stable_json_hash, utc_now_iso


@dataclass(frozen=True)
class PackageArtifact:
    label: str
    path: str | Path
    required: bool = True


DEPLOYMENT_REQUIRED_ARTIFACTS = (
    "release_audit",
    "model",
    "config",
    "model_manifest",
    "model_card",
    "field_report",
    "evidence_manifest",
    "acceptance_report",
    "stratified_acceptance_report",
    "benchmark_report",
    "validation_plan_report",
    "truth_quality_report",
    "truth_coverage_report",
    "stratified_truth_coverage_report",
    "mission_quality_report",
    "prediction_quality_report",
    "holdout_verify_report",
    "flight_check_report",
    "domain_check_report",
    "geo_accuracy_report",
    "deployment_manifest",
    "project_readme",
    "license",
    "package_readme",
)


def build_release_package(
    output_dir: str | Path,
    release_audit_report: str | Path,
    artifacts: list[PackageArtifact] | None = None,
    package_name: str = "bananavision-release",
    allow_failed_audit: bool = False,
    create_zip: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    release_audit_report = Path(release_audit_report)
    if not release_audit_report.exists():
        raise FileNotFoundError(f"Release audit report does not exist: {release_audit_report}")

    audit = json.loads(release_audit_report.read_text(encoding="utf-8"))
    audit_status = str(audit.get("status", "missing"))
    if audit_status != "pass" and not allow_failed_audit:
        raise ValueError(f"Release audit status must be pass before packaging; got {audit_status}")

    package_root = output_dir / _safe_name(package_name)
    artifact_root = package_root / "artifacts"
    zip_path = output_dir / f"{_safe_name(package_name)}.zip" if create_zip else None
    _prepare_output(package_root, zip_path, overwrite=overwrite)
    artifact_root.mkdir(parents=True, exist_ok=True)

    package_artifacts = [PackageArtifact("release_audit", release_audit_report), *(artifacts or [])]
    copied = [_copy_artifact(item, artifact_root) for item in package_artifacts]
    readme_path = package_root / "README.release.md"
    readme_path.write_text(_package_readme(package_name, audit_status), encoding="utf-8")
    copied.append(_manifest_entry("package_readme", readme_path, package_root))

    manifest = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "package_name": package_name,
        "package_root": str(package_root),
        "package_status": "release" if audit_status == "pass" else "exploratory",
        "bananavision_version": __version__,
        "release_audit_status": audit_status,
        "release_audit_summary": _audit_summary(audit),
        "artifact_count": len(copied),
        "artifacts": copied,
    }
    manifest_path = package_root / "release_package_manifest.json"
    manifest["manifest_path"] = str(manifest_path.relative_to(package_root)).replace("\\", "/")
    manifest["zip_path"] = None if zip_path is None else str(zip_path.name)
    manifest["manifest_sha256"] = stable_json_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if create_zip:
        _write_zip(package_root, zip_path)
        manifest["zip_sha256"] = file_sha256(zip_path)
    return manifest


def parse_artifact_specs(specs: list[str] | None) -> list[PackageArtifact]:
    artifacts: list[PackageArtifact] = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"Artifact must use LABEL=PATH format: {spec}")
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Artifact must use LABEL=PATH format: {spec}")
        artifacts.append(PackageArtifact(label, path))
    return artifacts


def verify_release_package(
    package_path: str | Path,
    output_json: str | Path | None = None,
    allow_exploratory: bool = False,
    require_deployment_artifacts: bool = False,
) -> dict[str, Any]:
    package_path = Path(package_path)
    if package_path.is_file() and package_path.suffix.lower() == ".zip":
        report = _verify_zip_package(
            package_path,
            allow_exploratory=allow_exploratory,
            require_deployment_artifacts=require_deployment_artifacts,
        )
    else:
        report = _verify_directory_package(
            package_path,
            allow_exploratory=allow_exploratory,
            require_deployment_artifacts=require_deployment_artifacts,
        )
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _copy_artifact(artifact: PackageArtifact, artifact_root: Path) -> dict[str, Any]:
    source = Path(artifact.path)
    if not source.exists():
        if artifact.required:
            raise FileNotFoundError(f"Required package artifact does not exist: {source}")
        return {
            "label": artifact.label,
            "source_path": str(source),
            "packaged_path": None,
            "missing": True,
            "required": artifact.required,
        }
    if not source.is_file():
        raise ValueError(f"Package artifact must be a file: {source}")
    target = _unique_target(artifact_root, artifact.label, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return _manifest_entry(artifact.label, target, artifact_root.parent, source_path=source)


def _verify_directory_package(
    package_path: Path,
    allow_exploratory: bool,
    require_deployment_artifacts: bool,
) -> dict[str, Any]:
    if package_path.is_dir():
        package_root = package_path
        manifest_path = package_root / "release_package_manifest.json"
    else:
        manifest_path = package_path
        package_root = manifest_path.parent

    checks: list[dict[str, str]] = []
    if not manifest_path.exists():
        checks.append(_package_check("manifest_present", "fail", f"manifest does not exist: {manifest_path}"))
        return _verification_report(
            package_path,
            manifest_path,
            {},
            checks,
            verified_artifacts=0,
            require_deployment_artifacts=require_deployment_artifacts,
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        checks.append(_package_check("manifest_readable", "fail", f"could not read manifest: {exc}"))
        return _verification_report(
            package_path,
            manifest_path,
            {},
            checks,
            verified_artifacts=0,
            require_deployment_artifacts=require_deployment_artifacts,
        )
    checks.extend(
        _manifest_checks(
            manifest,
            allow_exploratory=allow_exploratory,
            require_deployment_artifacts=require_deployment_artifacts,
        )
    )
    verified_artifacts = 0
    for artifact in manifest.get("artifacts", []) or []:
        artifact_checks, verified = _verify_directory_artifact(package_root, artifact)
        checks.extend(artifact_checks)
        verified_artifacts += int(verified)
    return _verification_report(
        package_path,
        manifest_path,
        manifest,
        checks,
        verified_artifacts=verified_artifacts,
        require_deployment_artifacts=require_deployment_artifacts,
    )


def _verify_zip_package(
    package_path: Path,
    allow_exploratory: bool,
    require_deployment_artifacts: bool,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    if not package_path.exists():
        checks.append(_package_check("zip_present", "fail", f"zip does not exist: {package_path}"))
        return _verification_report(
            package_path,
            None,
            {},
            checks,
            verified_artifacts=0,
            require_deployment_artifacts=require_deployment_artifacts,
        )

    try:
        archive_handle = zipfile.ZipFile(package_path)
    except zipfile.BadZipFile as exc:
        checks.append(_package_check("zip_readable", "fail", f"could not read zip: {exc}"))
        return _verification_report(
            package_path,
            None,
            {},
            checks,
            verified_artifacts=0,
            require_deployment_artifacts=require_deployment_artifacts,
        )

    with archive_handle as archive:
        manifest_names = [
            name
            for name in archive.namelist()
            if name == "release_package_manifest.json" or name.endswith("/release_package_manifest.json")
        ]
        if len(manifest_names) != 1:
            checks.append(_package_check("manifest_present", "fail", f"expected 1 manifest in zip, found {len(manifest_names)}"))
            return _verification_report(
                package_path,
                None,
                {},
                checks,
                verified_artifacts=0,
                require_deployment_artifacts=require_deployment_artifacts,
            )
        manifest_name = manifest_names[0]
        try:
            manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        except Exception as exc:
            checks.append(_package_check("manifest_readable", "fail", f"could not read manifest: {exc}"))
            return _verification_report(
                package_path,
                manifest_name,
                {},
                checks,
                verified_artifacts=0,
                require_deployment_artifacts=require_deployment_artifacts,
            )
        checks.append(_package_check("zip_present", "pass", str(package_path)))
        checks.extend(
            _manifest_checks(
                manifest,
                allow_exploratory=allow_exploratory,
                require_deployment_artifacts=require_deployment_artifacts,
            )
        )
        root_prefix = manifest_name.rsplit("/", 1)[0] if "/" in manifest_name else ""
        verified_artifacts = 0
        for artifact in manifest.get("artifacts", []) or []:
            artifact_checks, verified = _verify_zip_artifact(archive, root_prefix, artifact)
            checks.extend(artifact_checks)
            verified_artifacts += int(verified)
    return _verification_report(
        package_path,
        manifest_name,
        manifest,
        checks,
        verified_artifacts=verified_artifacts,
        require_deployment_artifacts=require_deployment_artifacts,
    )


def _manifest_checks(
    manifest: dict[str, Any],
    allow_exploratory: bool,
    require_deployment_artifacts: bool,
) -> list[dict[str, str]]:
    checks = [
        _package_check("manifest_present", "pass", "release_package_manifest.json loaded"),
        _package_check(
            "schema_version",
            "pass" if manifest.get("schema_version") == 1 else "fail",
            f"schema_version={manifest.get('schema_version')}",
        ),
    ]
    expected_hash = manifest.get("manifest_sha256")
    if expected_hash:
        hash_payload = dict(manifest)
        hash_payload.pop("manifest_sha256", None)
        hash_payload.pop("zip_sha256", None)
        actual_hash = stable_json_hash(hash_payload)
        checks.append(
            _package_check(
                "manifest_hash",
                "pass" if actual_hash == expected_hash else "fail",
                f"{actual_hash} == {expected_hash}",
            )
        )
    else:
        checks.append(_package_check("manifest_hash", "fail", "manifest_sha256 missing"))

    package_status = str(manifest.get("package_status", "missing"))
    if package_status == "release":
        checks.append(_package_check("package_status", "pass", "release"))
    elif package_status == "exploratory" and allow_exploratory:
        checks.append(_package_check("package_status", "warn", "exploratory package allowed"))
    else:
        checks.append(_package_check("package_status", "fail", f"package_status={package_status}"))

    audit_status = str(manifest.get("release_audit_status", "missing"))
    audit_status_ok = audit_status == "pass"
    audit_status_warn = package_status == "exploratory" and allow_exploratory
    checks.append(
        _package_check(
            "release_audit_status",
            "pass" if audit_status_ok else "warn" if audit_status_warn else "fail",
            f"release_audit_status={audit_status}",
        )
    )
    artifacts = manifest.get("artifacts", []) or []
    expected_count = _safe_int(manifest.get("artifact_count"), -1)
    checks.append(
        _package_check(
            "artifact_count",
            "pass" if expected_count == len(artifacts) else "fail",
            f"{len(artifacts)} artifact(s), manifest expected {expected_count}",
        )
    )
    labels = {str(artifact.get("label")) for artifact in artifacts if isinstance(artifact, dict)}
    for required_label in ["release_audit", "package_readme"]:
        checks.append(
            _package_check(
                f"required_artifact:{required_label}",
                "pass" if required_label in labels else "fail",
                "present" if required_label in labels else "missing",
            )
        )
    if require_deployment_artifacts:
        for required_label in DEPLOYMENT_REQUIRED_ARTIFACTS:
            checks.append(
                _package_check(
                    f"deployment_artifact:{required_label}",
                    "pass" if required_label in labels else "fail",
                    "present" if required_label in labels else "missing",
                )
            )
    return checks


def _verify_directory_artifact(package_root: Path, artifact: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
    label = str(artifact.get("label", "artifact"))
    relative = artifact.get("packaged_path")
    required = bool(artifact.get("required", True))
    if not relative:
        status = "fail" if required else "warn"
        return [_package_check(f"artifact:{label}:path", status, "packaged_path missing")], False
    path = package_root / str(relative)
    if not path.exists() or not path.is_file():
        status = "fail" if required else "warn"
        return [_package_check(f"artifact:{label}:present", status, f"missing: {path}")], False
    return _artifact_integrity_checks(label, artifact, file_sha256(path), path.stat().st_size)


def _verify_zip_artifact(
    archive: zipfile.ZipFile,
    root_prefix: str,
    artifact: dict[str, Any],
) -> tuple[list[dict[str, str]], bool]:
    label = str(artifact.get("label", "artifact"))
    relative = artifact.get("packaged_path")
    required = bool(artifact.get("required", True))
    if not relative:
        status = "fail" if required else "warn"
        return [_package_check(f"artifact:{label}:path", status, "packaged_path missing")], False
    name = "/".join(part for part in [root_prefix, str(relative).replace("\\", "/")] if part)
    try:
        info = archive.getinfo(name)
    except KeyError:
        status = "fail" if required else "warn"
        return [_package_check(f"artifact:{label}:present", status, f"missing: {name}")], False
    digest = _sha256_bytes(archive.read(info))
    return _artifact_integrity_checks(label, artifact, digest, info.file_size)


def _artifact_integrity_checks(
    label: str,
    artifact: dict[str, Any],
    actual_sha256: str | None,
    actual_size: int,
) -> tuple[list[dict[str, str]], bool]:
    expected_sha256 = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    checks = [
        _package_check(f"artifact:{label}:present", "pass", str(artifact.get("packaged_path", ""))),
        _package_check(
            f"artifact:{label}:sha256",
            "pass" if actual_sha256 == expected_sha256 else "fail",
            f"{actual_sha256} == {expected_sha256}",
        ),
    ]
    if expected_size is not None:
        expected_size = _safe_int(expected_size, -1)
        checks.append(
            _package_check(
                f"artifact:{label}:size",
                "pass" if actual_size == expected_size else "fail",
                f"{actual_size} == {expected_size}",
            )
        )
    return checks, all(check["status"] == "pass" for check in checks)


def _verification_report(
    package_path: Path,
    manifest_path: str | Path | None,
    manifest: dict[str, Any],
    checks: list[dict[str, str]],
    verified_artifacts: int,
    require_deployment_artifacts: bool,
) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "status": _package_overall_status(checks),
        "package_path": str(package_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "package_name": manifest.get("package_name"),
        "package_status": manifest.get("package_status"),
        "release_audit_status": manifest.get("release_audit_status"),
        "artifact_count": len(manifest.get("artifacts", []) or []),
        "verified_artifact_count": verified_artifacts,
        "deployment_artifacts_required": require_deployment_artifacts,
        "checks": checks,
    }


def _manifest_entry(
    label: str,
    path: Path,
    package_root: Path,
    source_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "source_path": None if source_path is None else str(source_path),
        "packaged_path": str(path.relative_to(package_root)).replace("\\", "/"),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "required": True,
    }


def _unique_target(artifact_root: Path, label: str, source: Path) -> Path:
    stem = _safe_name(label)
    suffix = "".join(source.suffixes)
    target = artifact_root / f"{stem}{suffix}"
    index = 2
    while target.exists():
        target = artifact_root / f"{stem}-{index}{suffix}"
        index += 1
    return target


def _write_zip(package_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_root.parent))


def _prepare_output(package_root: Path, zip_path: Path | None, overwrite: bool) -> None:
    existing = [path for path in [package_root, zip_path] if path is not None and path.exists()]
    if not existing:
        return
    if not overwrite:
        details = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Release package output already exists: {details}")
    if package_root.exists():
        shutil.rmtree(package_root)
    if zip_path is not None and zip_path.exists():
        zip_path.unlink()


def _audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    gates = audit.get("gates", []) or []
    return {
        "status": audit.get("status"),
        "gate_count": len(gates),
        "fail_count": sum(1 for gate in gates if gate.get("status") == "fail"),
        "warn_count": sum(1 for gate in gates if gate.get("status") == "warn"),
        "pass_count": sum(1 for gate in gates if gate.get("status") == "pass"),
    }


def _package_readme(package_name: str, audit_status: str) -> str:
    return f"""# {package_name}

This package was generated by BananaVision Drone.

- Release audit status: `{audit_status}`
- `release_package_manifest.json` contains SHA256 hashes for every packaged artifact.
- `artifacts/release_audit.json` is the publication gate for this package.
- Verify this folder or ZIP before deployment with `bananavision release-package-verify --require-deployment-artifacts`.

Do not publish or deploy this package as a production model unless the release
audit status is `pass` and the manifest hashes match the files in this folder.
"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _package_check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _package_overall_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "fail"
    if any(check["status"] == "warn" for check in checks):
        return "warn"
    return "pass"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    return cleaned.strip("-") or "artifact"
