from __future__ import annotations

from pathlib import Path
from typing import Any


def train_yolo(
    data_yaml: str | Path,
    model: str = "yolo26n-seg.pt",
    epochs: int = 120,
    imgsz: int = 1024,
    batch: int = 8,
    device: str | None = None,
    project: str = "runs/banana",
    name: str = "seg",
) -> Any:
    yolo = _load_yolo(model)
    return yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        task="segment",
        patience=30,
        plots=True,
    )


def validate_yolo(
    data_yaml: str | Path,
    model: str | Path,
    imgsz: int = 1024,
    batch: int = 8,
    device: str | None = None,
) -> Any:
    yolo = _load_yolo(str(model))
    return yolo.val(data=str(data_yaml), imgsz=imgsz, batch=batch, device=device, task="segment")


def export_yolo(
    model: str | Path,
    fmt: str = "onnx",
    imgsz: int = 1024,
    half: bool = False,
    int8: bool = False,
    device: str | None = None,
) -> Any:
    yolo = _load_yolo(str(model))
    return yolo.export(format=fmt, imgsz=imgsz, half=half, int8=int8, device=device)


def _load_yolo(model: str):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Install ML dependencies with: pip install 'bananavision-drone[ml]'") from exc
    return YOLO(model)
