from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
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
class CvatConfig:
    # Source for CVAT datasets. Provide either a zip file exported from CVAT or a root directory.
    zip_path: Optional[str] = None          # path to a CVAT export zip (YOLO or COCO). If provided, will be extracted.
    root: Optional[str] = None              # alternatively, path to an already-extracted dataset directory
    location: str = "./data"               # where to extract/prep the dataset (if zip provided)
    format: Optional[str] = None            # optional hint: "yolo" or "coco"; if None, will auto-detect
    data_yaml: Optional[str] = None         # optional explicit path to data.yaml if already present
    names: Optional[List[str]] = None       # optional class names; used when not discoverable in export


@dataclass
class DatasetConfig:
    # Which dataset source to use. Supported: 'roboflow' (default), 'cvat'
    source: str = "roboflow"


@dataclass
class ModelConfig:
    backend: str = "ultralytics"  # ultralytics | transformers
    arch: str = "yolov8n.pt"  # yolov8n.pt or HF id e.g. SenseTime/deformable-detr-with-box-refine
    imgsz: int = 640
    lr: float = 0.001
    epochs: int = 100
    batch: int = 16
    # OmegaConf doesn't support Unions with containers well; pass device as a string.
    # Examples: "cpu", "cuda:0", "0,1" (Ultralytics accepts GPU indices as comma-separated string)
    device: str | None = None
    seed: int = 42
    # Minimum prediction confidence (score) to retain detections during validation / visualization
    # Used primarily by the transformers backend (Deformable DETR). Can be overridden via CLI override:
    #   -o val_score_thresh=0.5
    pred_score_thresh: float = 0.25


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
class AugmentationConfig:
        """Augmentation settings for training.

        Generic structure so both backends can consume:
            - enable: master switch
            - ultralytics: dict of args forwarded directly to YOLO (e.g. mosaic=0.5, mixup=0.1)
            - basic params used by transformers backend to build an Albumentations pipeline.
        """
        enable: bool = True
        # Passed straight through to Ultralytics' train() (if backend == ultralytics)
        ultralytics: Dict[str, Any] = field(default_factory=dict)
        # Transformers backend simple knobs (probabilities / magnitudes)
        hflip: float = 0.5          # horizontal flip probability
        brightness: float = 0.2     # max delta for RandomBrightnessContrast (brightness_limit)
        contrast: float = 0.2       # contrast_limit
        hue: float = 0.02           # HSV hue shift fraction (approx)
        saturation: float = 0.2     # saturation shift fraction
        scale_min: float = 0.9      # random scale min (affine)
        scale_max: float = 1.1      # random scale max
        rotate: int = 0             # max absolute rotation degrees (0 disables)
        blur: float = 0.0           # probability of applying a small blur
        cutout: int = 0             # number of cutout holes (0 disables)


@dataclass
class Config:
    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)
    cvat: CvatConfig = field(default_factory=CvatConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

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
