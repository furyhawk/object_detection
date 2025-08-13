from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from od.utils.config import Config
from od.data import ensure_roboflow_dataset, find_data_yaml
import importlib
from od.models import UltralyticsBackend, TransformersDeformableDetrBackend
import os


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
            # unify threshold key for transformers backend (train epoch mAP filtering)
            "val_score_thresh": getattr(cfg.model, "pred_score_thresh", 0.25),
        },
    )


def _extract_names_from_yaml(yaml_path: Path) -> list[str]:
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception:
        return []
    names = data.get("names")
    if isinstance(names, dict):
        try:
            return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        except Exception:
            return list(names.values())
    if isinstance(names, list):
        return names
    return []


def _read_run_dir_class_names(run_dir: Path) -> list[str]:
    # Priority: classes.txt, then args.yaml names, then data.yaml referenced inside args.yaml
    try:
        if (run_dir / "classes.txt").is_file():
            return [l.strip() for l in (run_dir / "classes.txt").read_text().splitlines() if l.strip()]
        import yaml  # type: ignore
        args_file = run_dir / "args.yaml"
        if args_file.is_file():
            args = yaml.safe_load(args_file.read_text()) or {}
            if isinstance(args, dict):
                nm = args.get("names")
                if isinstance(nm, list):
                    return nm
                if isinstance(nm, dict):
                    try:
                        return [nm[k] for k in sorted(nm.keys(), key=lambda x: int(x))]
                    except Exception:
                        return list(nm.values())
                data_ref = args.get("data")
                if isinstance(data_ref, str):
                    p = Path(data_ref)
                    if p.is_file():
                        return _extract_names_from_yaml(p)
    except Exception:
        return []
    return []


def _find_best_trained_model(cfg: Config, data_yaml: Path) -> Optional[str]:
    """Attempt to locate the 'best' trained model artifact produced by a prior run_train.

    Strategy (lightweight, no metrics parsing):
      1. If cfg.model.arch already points to an existing file/dir, respect it.
      2. Look for a run directory matching train.project_dir/train.name.
         - Ultralytics: <run>/weights/best.pt
         - Transformers: <run>/config.json (directory acts as arch)
      3. If not found, scan all subdirectories of train.project_dir for newest matching pattern.
    Returns path (file or directory) or None if nothing suitable was found.
    """
    arch_path = Path(getattr(cfg.model, "arch", ""))
    if arch_path.exists():  # user already pointing to a concrete artifact
        return str(arch_path)

    backend = (cfg.model.backend or "").lower()
    train_root = Path(cfg.train.project_dir)
    # Candidate: explicit training name
    candidate_dir = train_root / cfg.train.name

    def _ultra_best(d: Path) -> Optional[Path]:
        p = d / "weights" / "best.pt"
        return p if p.is_file() else None

    def _transformers_dir(d: Path) -> Optional[Path]:
        cfg_json = d / "config.json"
        # Consider it a completed transformers training dir if config.json and at least one weight file exist
        if cfg_json.is_file() and any((d / w).exists() for w in ("pytorch_model.bin", "model.safetensors")):
            return d
        return None

    dataset_class_names = _extract_names_from_yaml(data_yaml)

    def _compatible(run_dir: Path) -> bool:
        if not dataset_class_names:  # Can't validate, accept
            return True
        run_names = _read_run_dir_class_names(run_dir)
        # Accept only if identical length and either same order or same set (fallback)
        if not run_names:
            return True  # can't judge; len mismatch could cause mislabels but we allow with warning
        if len(run_names) != len(dataset_class_names):
            return False
        if run_names == dataset_class_names:
            return True
        return set(run_names) == set(dataset_class_names)

    if backend == "ultralytics":
        best_candidate = _ultra_best(candidate_dir)
        if best_candidate and _compatible(candidate_dir):
            return str(best_candidate)
    elif backend == "transformers":
        best_candidate = _transformers_dir(candidate_dir)
        if best_candidate and _compatible(candidate_dir):
            return str(best_candidate)

    # Fallback: scan all subdirectories for most recent matching artifact
    if not train_root.is_dir():
        return None
    matches: list[tuple[float, Path]] = []
    for child in train_root.iterdir():
        if not child.is_dir():
            continue
        try:
            if backend == "ultralytics":
                bp = _ultra_best(child)
                if bp and _compatible(child):
                    matches.append((bp.stat().st_mtime, bp))
            elif backend == "transformers":
                td = _transformers_dir(child)
                if td and _compatible(child):
                    # use dir mtime
                    matches.append((td.stat().st_mtime, td))
        except Exception:
            continue
    if not matches:
        return None
    # Pick most recent
    matches.sort(key=lambda x: x[0], reverse=True)
    return str(matches[0][1])


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

    # Auto-discover best trained model artifact if available
    # Force preference: if runs/train/<name>/weights/best.pt exists use it (Ultralytics)
    preferred_ultra_best = Path(cfg.train.project_dir) / cfg.train.name / "weights" / "best.pt"
    auto_arch = None
    if cfg.model.backend.lower() == "ultralytics" and preferred_ultra_best.is_file():
        auto_arch = str(preferred_ultra_best)
    else:
        auto_arch = _find_best_trained_model(cfg, data_yaml)
    selected_arch = auto_arch or cfg.model.arch
    if auto_arch:
        try:
            print(f"[INFO] Auto-selected best trained model for validation: {auto_arch}")
        except Exception:
            pass
    backend = _select_backend(cfg.model.backend, selected_arch, cfg.model.device)

    # Special case: Ultralytics best.pt file should replace underlying model if discovered after instantiation
    if auto_arch and cfg.model.backend.lower() == "ultralytics":
        try:
            from ultralytics import YOLO  # local import to avoid cost when not needed
            if Path(auto_arch).is_file():
                backend.model = YOLO(auto_arch)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            try:
                print(f"[WARN] Failed to load auto-selected weights '{auto_arch}': {e}")
            except Exception:
                pass

    # Validate label alignment (helps diagnose 'incorrect labels' issues)
    try:
        dataset_names = _extract_names_from_yaml(data_yaml)
        trained_names: list[str] = []
        # For ultralytics, search run directory for classes.txt (same location as best.pt's parent/parent)
        if cfg.model.backend.lower() == "ultralytics":
            run_dir = None
            if auto_arch:
                p = Path(auto_arch)
                if p.name == "best.pt":
                    run_dir = p.parent.parent  # .../run_name
                elif p.name.endswith(".pt"):
                    run_dir = p.parent.parent if (p.parent / "weights").is_dir() else p.parent
            if run_dir and (run_dir / "classes.txt").is_file():
                trained_names = [l.strip() for l in (run_dir / "classes.txt").read_text().splitlines() if l.strip()]
        elif cfg.model.backend.lower() == "transformers":
            # Look at project train dir for classes.txt aligned with training name
            run_dir = Path(cfg.train.project_dir) / cfg.train.name
            if run_dir.is_dir() and (run_dir / "classes.txt").is_file():
                trained_names = [l.strip() for l in (run_dir / "classes.txt").read_text().splitlines() if l.strip()]
        if dataset_names and trained_names:
            if len(dataset_names) != len(trained_names):
                print("[ERROR] Dataset class count != trained model class count. This will cause incorrect labels.")
                print(f"         dataset({len(dataset_names)}): {dataset_names}")
                print(f"         model({len(trained_names)}): {trained_names}")
            else:
                if dataset_names != trained_names:
                    print("[WARN] Class name order differs between dataset and trained model.\n"
                          f"       dataset: {dataset_names}\n       model:   {trained_names}\n"
                          "       Predictions may display mismatched labels if order is inconsistent.")
                else:
                    print("[INFO] Class names match between dataset and trained model.")
    except Exception as e:  # pragma: no cover
        try:
            print(f"[WARN] Label alignment check skipped due to error: {e}")
        except Exception:
            pass

    return backend.validate(
        data=str(data_yaml),
        project=cfg.val.project_dir,
        name=cfg.val.name,
        imgsz=cfg.model.imgsz,
        batch=cfg.model.batch,
        seed=cfg.model.seed,
        extra={
            **(overrides or {}),
            "val_score_thresh": getattr(cfg.model, "pred_score_thresh", 0.25),
        },
    )
