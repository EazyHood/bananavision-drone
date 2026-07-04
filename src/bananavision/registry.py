from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import InferenceConfig
from .runtime import file_sha256, runtime_fingerprint, stable_json_hash, utc_now_iso


def register_model(
    model_path: str | Path,
    registry_dir: str | Path,
    version: str,
    config: InferenceConfig,
    acceptance_report: str | Path | None = None,
    benchmark_report: str | Path | None = None,
    promotion: dict[str, Any] | None = None,
    notes: str = "",
) -> Path:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": version,
        "created_at": utc_now_iso(),
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "runtime": runtime_fingerprint(config),
        "acceptance_report": _load_optional_json(acceptance_report),
        "benchmark_report": _load_optional_json(benchmark_report),
        "promotion": promotion,
        "notes": notes,
    }
    manifest["manifest_sha256"] = stable_json_hash(manifest)
    path = registry_dir / f"{_safe_version(version)}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    latest = registry_dir / "latest.json"
    latest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def promote_model(
    model_path: str | Path,
    registry_dir: str | Path,
    version: str,
    config: InferenceConfig,
    acceptance_report: str | Path,
    benchmark_report: str | Path,
    max_p95_ms: float | None = None,
    notes: str = "",
) -> Path:
    acceptance = _load_required_json(acceptance_report)
    benchmark = _load_required_json(benchmark_report)
    gates = _promotion_gates(acceptance, benchmark, max_p95_ms=max_p95_ms)
    failed = [gate for gate in gates if gate["status"] == "fail"]
    if failed:
        details = "; ".join(f"{gate['name']}: {gate['detail']}" for gate in failed)
        raise ValueError(f"Model promotion failed: {details}")
    return register_model(
        model_path=model_path,
        registry_dir=registry_dir,
        version=version,
        config=config,
        acceptance_report=acceptance_report,
        benchmark_report=benchmark_report,
        promotion={
            "status": "promoted",
            "created_at": utc_now_iso(),
            "gates": gates,
            "max_p95_ms": max_p95_ms,
        },
        notes=notes,
    )


def _load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_required_json(path: str | Path) -> dict[str, Any]:
    return _load_optional_json(path) or {}


def _promotion_gates(
    acceptance: dict[str, Any],
    benchmark: dict[str, Any],
    max_p95_ms: float | None,
) -> list[dict[str, Any]]:
    gates = []
    acceptance_passed = acceptance.get("passed") is True
    gates.append(
        {
            "name": "acceptance_passed",
            "status": "pass" if acceptance_passed else "fail",
            "detail": "Acceptance report passed" if acceptance_passed else "Acceptance report did not pass",
        }
    )
    latency = benchmark.get("latency_ms", {}) or {}
    p95 = latency.get("p95")
    has_p95 = isinstance(p95, int | float)
    gates.append(
        {
            "name": "benchmark_p95_present",
            "status": "pass" if has_p95 else "fail",
            "detail": f"p95={p95}" if has_p95 else "benchmark.latency_ms.p95 is missing",
        }
    )
    if max_p95_ms is not None and has_p95:
        passed = float(p95) <= max_p95_ms
        gates.append(
            {
                "name": "benchmark_p95_limit",
                "status": "pass" if passed else "fail",
                "detail": f"p95={float(p95):.3f}ms <= {max_p95_ms:.3f}ms"
                if passed
                else f"p95={float(p95):.3f}ms > {max_p95_ms:.3f}ms",
            }
        )
    return gates


def _safe_version(version: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in version).strip("-")
