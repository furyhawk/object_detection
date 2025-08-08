from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from omegaconf import OmegaConf
import os


@dataclass
class RoboflowConfig:
    api_key: Optional[str] = None  # if None, read from ROBOFLOW_API_KEY env
    workspace: str | None = None
    project: str | None = None
    version: int | None = None
    split: str = "yolov8"  # supports: yolov8, coco
    format: str | None = None  # override if needed
    location: str = "./data"
    data_yaml: Optional[str] = None  # optional explicit path to data.yaml


@dataclass
class ModelConfig:
    backend: str = "ultralytics"  # ultralytics | transformers (future)
    arch: str = "yolov8n.pt"  # can be a model name or path to checkpoint
    imgsz: int = 640
    lr: float = 0.001
    epochs: int = 100
    batch: int = 16
    # OmegaConf doesn't support Unions with containers well; pass device as a string.
    # Examples: "cpu", "cuda:0", "0,1" (Ultralytics accepts GPU indices as comma-separated string)
    device: str | None = None
    seed: int = 42


@dataclass
class TrainConfig:
    project_dir: str = "runs/train"
    name: str = "exp"
    resume: bool = False


@dataclass
class ValConfig:
    project_dir: str = "runs/val"
    name: str = "exp"


@dataclass
class Config:
    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)

    @staticmethod
    def load(path: str | None = None, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        base = OmegaConf.unsafe_merge(
            OmegaConf.structured(Config()),
            OmegaConf.create(overrides or {}),
        )
        if path and os.path.exists(path):
            cfg = OmegaConf.merge(base, OmegaConf.load(path))
        else:
            cfg = base
        return OmegaConf.to_object(cfg)  # type: ignore[return-value]

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(OmegaConf.structured(self))
