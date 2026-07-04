from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from . import __version__
from .pipeline import load_config, make_detector, predict_image
from .runtime import runtime_fingerprint, utc_now_iso

ALLOWED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
API_SCHEMA_VERSION = 1
API_VERSION = "v1"


def create_app(
    config_path: str | Path | None = None,
    api_key: str | None = None,
    max_upload_mb: float = 25.0,
):
    try:
        from fastapi import FastAPI, File, Header, HTTPException, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Install API dependencies with: pip install 'bananavision-drone[api]'") from exc
    globals()["Request"] = Request

    if max_upload_mb <= 0:
        raise ValueError("max_upload_mb must be positive")
    config = load_config(config_path)
    detector = make_detector(config)
    readiness = {
        "schema_version": API_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "status": "ready",
        "loaded_at": utc_now_iso(),
        "config_path": None if config_path is None else str(config_path),
        "runtime": runtime_fingerprint(config),
        "api": {
            "auth_required": bool(api_key),
            "max_upload_mb": max_upload_mb,
            "max_upload_bytes": int(max_upload_mb * 1024 * 1024),
            "allowed_suffixes": sorted(ALLOWED_UPLOAD_SUFFIXES),
        },
    }

    app = FastAPI(title="BananaVision Drone API", version=__version__)
    app.state.config = config
    app.state.detector = detector
    app.state.readiness = readiness

    @app.middleware("http")
    async def add_contract_headers(request: Request, call_next: Any):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-BananaVision-API-Version"] = API_VERSION
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return _error_response(
            request,
            JSONResponse,
            status_code=int(exc.status_code),
            code=_error_code(int(exc.status_code)),
            message=str(exc.detail),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return _error_response(
            request,
            JSONResponse,
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            details=exc.errors(),
        )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "schema_version": API_SCHEMA_VERSION,
            "api_version": API_VERSION,
            "status": "ok",
            "version": __version__,
            "request_id": _request_id(request),
            "generated_at": utc_now_iso(),
        }

    @app.get("/ready")
    def ready(request: Request) -> dict[str, Any]:
        return {
            **app.state.readiness,
            "request_id": _request_id(request),
            "generated_at": utc_now_iso(),
        }

    @app.post("/infer")
    async def infer(
        request: Request,
        file: Any = File(...),
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _check_api_key(api_key, authorization, x_api_key, HTTPException)
        suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
        if suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"Unsupported image suffix: {suffix}")
        content = await file.read()
        max_upload_bytes = app.state.readiness["api"]["max_upload_bytes"]
        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {max_upload_mb:.2f} MB limit")
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / f"upload{suffix}"
            image_path.write_bytes(content)
            result = predict_image(image_path, app.state.config, detector=app.state.detector)
            payload = result.to_dict()
            return {
                **payload,
                "schema_version": API_SCHEMA_VERSION,
                "api_version": API_VERSION,
                "status": "ok",
                "request_id": _request_id(request),
                "generated_at": utc_now_iso(),
                "upload": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size_bytes": len(content),
                    "suffix": suffix.lower(),
                },
                "result": payload,
            }

    return app


def _error_response(
    request: Any,
    json_response: type,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
):
    request_id = _request_id(request)
    payload = {
        "schema_version": API_SCHEMA_VERSION,
        "api_version": API_VERSION,
        "status": "error",
        "request_id": request_id,
        "generated_at": utc_now_iso(),
        "detail": message,
        "error": {
            "code": code,
            "message": message,
            "status_code": status_code,
            "details": details,
        },
    }
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id
    response_headers["X-BananaVision-API-Version"] = API_VERSION
    return json_response(status_code=status_code, content=payload, headers=response_headers)


def _request_id(request: Any) -> str:
    return str(getattr(getattr(request, "state", None), "request_id", "") or request.headers.get("X-Request-ID") or uuid4().hex)


def _error_code(status_code: int) -> str:
    codes = {
        401: "unauthorized",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
    }
    return codes.get(status_code, "http_error")


def _check_api_key(
    expected: str | None,
    authorization: str | None,
    x_api_key: str | None,
    http_exception: type[Exception],
) -> None:
    if not expected:
        return
    bearer = _bearer_token(authorization)
    if x_api_key == expected or bearer == expected:
        return
    raise http_exception(status_code=401, detail="Missing or invalid API key")


def _bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    prefix = "bearer "
    if value.lower().startswith(prefix):
        return value[len(prefix) :].strip()
    return None
