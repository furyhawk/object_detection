from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from omegaconf import OmegaConf


def _detect_cvat_format(root: Path) -> str:
    """Detect CVAT export format under root. Returns 'yolo' or 'coco'. Raises if unknown.

    Preference: if COCO annotations are present, classify as COCO even if an images/ folder exists.
    For YOLO, require stronger evidence than just an images/ directory (e.g., data.yaml or labels present).
    """
    # COCO strong signals first
    coco_candidates = [
        root / "annotations" / "instances_train.json",
        root / "annotations" / "instances_val.json",
        root / "annotations" / "instances_valid.json",
        root / "annotations" / "instances_default.json",
        root / "annotations" / "train.json",
        root / "annotations" / "val.json",
        root / "annotations.json",
    ]
    if any(p.exists() for p in coco_candidates):
        return "coco"

    # YOLO signals: explicit files or paired images/labels directories
    if (root / "data.yaml").exists() or (root / "obj.names").exists() or (root / "obj.data").exists():
        return "yolo"
    # paired dirs at root
    if (root / "images").exists() and (root / "labels").exists():
        return "yolo"
    # paired dirs under train/
    if (root / "train" / "images").exists() and (root / "train" / "labels").exists():
        return "yolo"
    # paired dirs images/train and labels/train
    if (root / "images" / "train").exists() and (root / "labels" / "train").exists():
        return "yolo"

    # Try recursive hints (avoid heavy scan)
    for p in root.rglob("*.json"):
        if p.name.startswith("instances_") or p.name in {"train.json", "val.json", "annotations.json"}:
            return "coco"
    for p in root.rglob("data.yaml"):
        return "yolo"

    raise ValueError("Could not detect CVAT export format. Provide cvat.format as 'yolo' or 'coco'.")


def _extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # If archive contains single top-level folder, extract as-is; else into out_dir/zip_stem
        top_names = {Path(n).parts[0] for n in zf.namelist() if not n.endswith("/")}
        extract_root = out_dir
        if len(top_names) != 1:
            extract_root = out_dir / zip_path.stem
            extract_root.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_root)
        # If there is a single folder under out_dir, return it; else out_dir/zip_stem
        if extract_root == out_dir:
            subs = [p for p in out_dir.iterdir() if p.is_dir()]
            if len(subs) == 1:
                return subs[0]
        return extract_root


def _write_yaml(path: Path, data: Dict) -> None:
    """Write a very small subset of YAML (sufficient for Ultralytics data.yaml) without extra deps."""
    def dump(obj, indent: int = 0) -> str:
        pref = " " * indent
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pref}{k}:")
                    lines.append(dump(v, indent + 2))
                else:
                    lines.append(f"{pref}{k}: {v}")
            return "\n".join(lines)
        if isinstance(obj, list):
            lines = []
            for v in obj:
                if isinstance(v, (dict, list)):
                    lines.append(f"{pref}-")
                    lines.append(dump(v, indent + 2))
                else:
                    lines.append(f"{pref}- {v}")
            return "\n".join(lines)
        return f"{pref}{obj}"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(dump(data) + "\n")


def _ensure_yolo_yaml(root: Path, names: Optional[List[str]] = None, overwrite: bool = False) -> Path:
    """Create or return data.yaml pointing to train/val/test splits. Best-effort.

    Expects typical CVAT YOLO layout:
      - images/{train,val,test} and labels/{train,val,test}
      or train/images, train/labels, val/images, val/labels, etc.
    """
    # If a data.yaml exists somewhere, prefer that
    existing = list(root.rglob("data.yaml"))
    if existing and not overwrite:
        return existing[0]

    # Heuristics for split dirs (case-insensitive aliases)
    def try_paths(base: Path) -> Optional[Tuple[Path, Path]]:
        img = base / "images"
        lbl = base / "labels"
        if img.exists() and lbl.exists():
            return img, lbl
        return None

    splits: Dict[str, Tuple[Path, Path]] = {}
    # alias mapping for case-insensitive detection
    split_aliases = {
        "train": ["train", "Train", "TRAIN"],
        "val": ["val", "Val", "VAL", "valid", "Valid", "VALID", "validation", "Validation", "VALIDATION"],
        "test": ["test", "Test", "TEST"],
    }
    # common pattern A: root/{split}/{images,labels}
    for key, aliases in split_aliases.items():
        for split in aliases:
            pair = try_paths(root / split)
            if pair:
                splits["val" if key == "val" else key] = pair
                break

    # pattern B: root/{images,labels}/{train,val,test}
    if not splits:
        images_root = root / "images"
        labels_root = root / "labels"
        if images_root.exists() and labels_root.exists():
            for key, aliases in split_aliases.items():
                for split in aliases:
                    img = images_root / split
                    lbl = labels_root / split
                    if img.exists() and lbl.exists():
                        splits["val" if key == "val" else key] = (img, lbl)
                        break

    # If still nothing, as a last attempt, if we at least have train/ and val/ anywhere
    if not splits:
        for base in [root] + [p for p in root.iterdir() if p.is_dir()]:
            # try case-insensitive train
            pair_train = None
            for split in split_aliases["train"]:
                pair_train = try_paths(base / split)
                if pair_train:
                    break
            pair_val = None
            for split in split_aliases["val"]:
                pair_val = try_paths(base / split)
                if pair_val:
                    break
            if pair_train and pair_val:
                splits["train"] = pair_train
                splits["val"] = pair_val
                break

    if not splits:
        raise FileNotFoundError("Could not infer YOLO images/labels structure under the provided CVAT dataset.")

    # Build data.yaml content
    def to_str(p: Path) -> str:
        return str(p.resolve())

    # Resolve train/val image directories explicitly to satisfy type checkers
    if "train" in splits:
        train_imgs = splits["train"][0]
    else:
        train_imgs = splits["val"][0]
    if "val" in splits:
        val_imgs = splits["val"][0]
    else:
        val_imgs = train_imgs

    data: Dict = {
        "path": str(root.resolve()),
        "train": to_str(train_imgs),
        "val": to_str(val_imgs),
    }
    if "test" in splits:
        data["test"] = to_str(splits["test"][0])

    # classes
    if names:
        data["names"] = names
    else:
        # Try to read from an existing YAML or obj.names if available
        # Existing YAML names
        existing_yaml = existing[0] if existing else None
        loaded_names: Optional[List[str]] = None
        if existing_yaml and existing_yaml.exists():
            try:
                yobj = OmegaConf.to_object(OmegaConf.load(existing_yaml))  # type: ignore[arg-type]
                if isinstance(yobj, dict) and "names" in yobj:
                    nm = yobj["names"]
                    if isinstance(nm, list):
                        loaded_names = nm
                    elif isinstance(nm, dict):
                        # sort by int keys
                        try:
                            loaded_names = [v for k, v in sorted(((int(k), v) for k, v in nm.items()))]
                        except Exception:
                            loaded_names = list(nm.values())
            except Exception:
                pass
        if not loaded_names:
            obj_names = None
            for p in [root / "obj.names", root / "data" / "obj.names"]:
                if p.exists():
                    obj_names = [line.strip() for line in p.read_text().splitlines() if line.strip()]
                    break
            if obj_names:
                loaded_names = obj_names
        if loaded_names:
            data["names"] = loaded_names

    yaml_path = root / "data.yaml"
    _write_yaml(yaml_path, data)
    return yaml_path
def _yaml_has_keys(path: Path, keys: List[str]) -> bool:
    """Return True if YAML at path contains all top-level keys."""
    try:
        obj = OmegaConf.to_object(OmegaConf.load(path))  # type: ignore[arg-type]
        if isinstance(obj, dict):
            return all(k in obj for k in keys)
    except Exception:
        # Fallback to naive scan
        try:
            text = path.read_text()
            return all(f"{k}:" in text for k in keys)
        except Exception:
            return False
    return False



def prepare_cvat_dataset(
    zip_path: Optional[str] = None,
    root: Optional[str] = None,
    location: str = "./data",
    format: Optional[str] = None,
    names: Optional[List[str]] = None,
) -> Path:
    """
    Prepare a dataset exported from CVAT for use with Ultralytics, returning the dataset root path.

    If a zip is provided, it will be extracted under `location` and an Ultralytics-compatible data.yaml will be generated if missing.
    If a root directory is provided, will try to detect/generate data.yaml.
    """
    if not zip_path and not root:
        raise ValueError("Provide either cvat.zip_path or cvat.root in the config.")

    if zip_path:
        zp = Path(zip_path)
        if not zp.exists():
            raise FileNotFoundError(f"CVAT zip not found: {zp}")
        out_dir = Path(location)
        ds_root = _extract_zip(zp, out_dir)
    else:
        assert root is not None
        ds_root = Path(root).resolve()

    ds_root = ds_root.resolve()

    # If data.yaml exists, validate it has required keys; if not, we will regenerate
    for y in [ds_root / "data.yaml"] + list(ds_root.rglob("data.yaml")):
        if y.exists():
            if _yaml_has_keys(y, ["train", "val"]):
                return y.parent
            else:
                # Regenerate a proper Ultralytics data.yaml at dataset root
                try:
                    _ensure_yolo_yaml(ds_root, names=names, overwrite=True)
                    return ds_root
                except Exception:
                    # fall through to detection path below
                    break

    fmt = format or _detect_cvat_format(ds_root)
    if fmt == "yolo":
        _ensure_yolo_yaml(ds_root, names=names, overwrite=True)
        return ds_root
    elif fmt == "coco":
        # Ultralytics can read COCO JSON directly by pointing data.yaml to the annotations and images
        # Try to assemble a minimal data.yaml
        ann = None
        for cand in [
            ds_root / "annotations" / "instances_train.json",
            ds_root / "annotations" / "instances_default.json",
            ds_root / "annotations" / "train.json",
            ds_root / "annotations.json",
        ]:
            if cand.exists():
                ann = cand
                break
        images_dir = None
        for cand in [
            ds_root / "images" / "train",
            ds_root / "images" / "default",
            ds_root / "train" / "images",
            ds_root / "images",
        ]:
            if cand.exists():
                images_dir = cand
                break
        if not ann or not images_dir:
            raise FileNotFoundError("Could not infer COCO annotations/images paths in CVAT export.")
        data = {
            "path": str(ds_root),
            "train": str(images_dir),
            "val": str(images_dir),  # best-effort; users can override
            "annotations": str(ann),
        }
        _write_yaml(ds_root / "data.yaml", data)
        return ds_root
    else:
        raise ValueError(f"Unsupported CVAT format: {fmt}")


def find_data_yaml(dataset_dir: Path) -> Optional[Path]:
    for name in ["data.yaml", "dataset.yaml"]:
        p = dataset_dir / name
        if p.exists():
            return p
    for p in dataset_dir.rglob("*.yaml"):
        if p.name in {"data.yaml", "dataset.yaml"}:
            return p
    return None
