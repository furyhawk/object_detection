from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from od.utils.config import Config
from od.data import ensure_roboflow_dataset, find_data_yaml
import importlib
from od.models import UltralyticsBackend, TransformersDeformableDetrBackend


def _select_backend(name: str, arch: str, device: Optional[str]):
    if name.lower() == "ultralytics":
        return UltralyticsBackend(arch=arch, device=device)
    if name.lower() == "transformers":
        return TransformersDeformableDetrBackend(arch=arch, device=device)
    raise NotImplementedError(f"Backend '{name}' is not supported yet")


essential_yaml_error = (
    "Could not find a data YAML in the downloaded Roboflow dataset. "
    "Please ensure the export format is 'yolov8' or provide a path via overrides."
)


def run_train(cfg: Config, overrides: Optional[Dict[str, Any]] = None) -> Any:
    # Select dataset source
    source = (cfg.dataset.source or "roboflow").lower()
    if source == "roboflow":
        if cfg.roboflow.data_yaml:
            data_yaml_opt: Optional[Path] = Path(cfg.roboflow.data_yaml)
            ds_path = data_yaml_opt.parent if data_yaml_opt else Path(cfg.roboflow.location)
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
    elif source == "cvat":
        cvat_mod = importlib.import_module("od.data.cvat")
        if cfg.cvat.data_yaml:
            data_yaml_opt = Path(cfg.cvat.data_yaml)
            ds_path = data_yaml_opt.parent
        else:
            ds_path = cvat_mod.prepare_cvat_dataset(
                zip_path=cfg.cvat.zip_path,
                root=cfg.cvat.root,
                location=cfg.cvat.location,
                format=cfg.cvat.format,
                names=cfg.cvat.names,
            )
            data_yaml_opt = find_data_yaml(Path(ds_path))
    else:
        raise NotImplementedError(f"Dataset source '{source}' is not supported.")
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
        extra={
            **(overrides or {}),
            # Provide augmentation sub-dict for backends to interpret
            "augmentation": getattr(cfg, "augmentation", None),
        },
    )


def run_validate(cfg: Config, overrides: Optional[Dict[str, Any]] = None) -> Any:
    source = (cfg.dataset.source or "roboflow").lower()
    if source == "roboflow":
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
    elif source == "cvat":
        cvat_mod = importlib.import_module("od.data.cvat")
        if cfg.cvat.data_yaml:
            data_yaml = Path(cfg.cvat.data_yaml)
            ds_path = data_yaml.parent
        else:
            ds_path = cvat_mod.prepare_cvat_dataset(
                zip_path=cfg.cvat.zip_path,
                root=cfg.cvat.root,
                location=cfg.cvat.location,
                format=cfg.cvat.format,
                names=cfg.cvat.names,
            )
            data_yaml_opt = find_data_yaml(Path(ds_path))
            if not data_yaml_opt:
                raise FileNotFoundError("Could not find a data YAML in the prepared CVAT dataset.")
            data_yaml = data_yaml_opt
    else:
        raise NotImplementedError(f"Dataset source '{source}' is not supported.")

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
