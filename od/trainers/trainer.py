from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from od.utils.config import Config
from od.data import ensure_roboflow_dataset, find_data_yaml
from od.models import UltralyticsBackend


def _select_backend(name: str, arch: str, device: Optional[str]):
    if name.lower() == "ultralytics":
        return UltralyticsBackend(arch=arch, device=device)
    raise NotImplementedError(f"Backend '{name}' is not supported yet")


essential_yaml_error = (
    "Could not find a data YAML in the downloaded Roboflow dataset. "
    "Please ensure the export format is 'yolov8' or provide a path via overrides."
)


def run_train(cfg: Config, overrides: Optional[Dict[str, Any]] = None) -> Any:
    # If an explicit data.yaml is provided, use it and skip download
    if cfg.roboflow.data_yaml:
        data_yaml_opt: Optional[Path] = Path(cfg.roboflow.data_yaml)
        ds_path = data_yaml_opt.parent if data_yaml_opt else Path(cfg.roboflow.location)
    else:
        # Download dataset
        ds_path = ensure_roboflow_dataset(
            api_key=cfg.roboflow.api_key,
            workspace=cfg.roboflow.workspace,
            project=cfg.roboflow.project,
            version=cfg.roboflow.version,
            split=cfg.roboflow.split,
            location=cfg.roboflow.location,
            format=cfg.roboflow.format,
        )
        data_yaml_opt = find_data_yaml(Path(ds_path))
    if not data_yaml_opt:
        raise FileNotFoundError(essential_yaml_error)
    data_yaml = data_yaml_opt

    backend = _select_backend(cfg.model.backend, cfg.model.arch, cfg.model.device)

    return backend.train(
    data=str(data_yaml),
        project=cfg.train.project_dir,
        name=cfg.train.name,
        imgsz=cfg.model.imgsz,
        epochs=cfg.model.epochs,
        batch=cfg.model.batch,
        lr=cfg.model.lr,
        seed=cfg.model.seed,
        resume=cfg.train.resume,
        extra=overrides,
    )


def run_validate(cfg: Config, overrides: Optional[Dict[str, Any]] = None) -> Any:
    if cfg.roboflow.data_yaml:
        data_yaml = Path(cfg.roboflow.data_yaml)
        ds_path = data_yaml.parent
    else:
        ds_path = ensure_roboflow_dataset(
            api_key=cfg.roboflow.api_key,
            workspace=cfg.roboflow.workspace,
            project=cfg.roboflow.project,
            version=cfg.roboflow.version,
            split=cfg.roboflow.split,
            location=cfg.roboflow.location,
            format=cfg.roboflow.format,
        )
        data_yaml_opt = find_data_yaml(Path(ds_path))
        if not data_yaml_opt:
            raise FileNotFoundError(essential_yaml_error)
        data_yaml = data_yaml_opt

    backend = _select_backend(cfg.model.backend, cfg.model.arch, cfg.model.device)

    return backend.validate(
        data=str(data_yaml),
        project=cfg.val.project_dir,
        name=cfg.val.name,
        imgsz=cfg.model.imgsz,
        batch=cfg.model.batch,
        seed=cfg.model.seed,
        extra=overrides,
    )
