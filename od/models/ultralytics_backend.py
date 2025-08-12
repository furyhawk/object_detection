from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO


class UltralyticsBackend:
    """Thin wrapper around Ultralytics YOLO for training and validation via config."""

    def __init__(self, arch: str = "yolov8n.pt", device: Optional[str] = None):
        self.model = YOLO(arch)
        self.device = device

    def train(
        self,
        data: str | Path,
        project: str = "runs/train",
        name: str = "exp",
        imgsz: int = 640,
        epochs: int = 100,
        batch: int = 16,
        lr: float = 0.001,
        seed: int = 42,
        resume: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        args = dict(
            data=str(data),
            project=project,
            name=name,
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            lr0=lr,
            seed=seed,
            device=self.device,
            resume=resume,
        )
        if extra:
            # Extract augmentation config if provided
            aug_cfg = extra.pop("augmentation", None)
            if aug_cfg and getattr(aug_cfg, "enable", False):  # dataclass-like
                ultra_args = getattr(aug_cfg, "ultralytics", {}) or {}
                # Don't overwrite explicitly set args
                for k, v in ultra_args.items():
                    args.setdefault(k, v)
            # Remaining extras (user overrides)
            args.update(extra)
        return self.model.train(**args)

    def validate(
        self,
        data: str | Path,
        project: str = "runs/val",
        name: str = "exp",
        imgsz: int = 640,
        batch: int = 16,
        seed: int = 42,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        args = dict(
            data=str(data),
            project=project,
            name=name,
            imgsz=imgsz,
            batch=batch,
            seed=seed,
            device=self.device,
        )
        if extra:
            args.update(extra)
        return self.model.val(**args)

    def export(self, format: str = "onnx", **kwargs: Any) -> Any:
        return self.model.export(format=format, **kwargs)
