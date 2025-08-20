from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings("ignore", message=r"for .*meta parameter")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
try:  # torchmetrics is optional but listed as dependency
    from torchmetrics.detection.mean_ap import MeanAveragePrecision  # type: ignore
except Exception:  # pragma: no cover
    MeanAveragePrecision = None  # type: ignore
try:
    from omegaconf import OmegaConf  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    OmegaConf = None  # type: ignore
from transformers import (
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
    YolosForObjectDetection,
    get_cosine_schedule_with_warmup,
)


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    """Read a YAML into a plain dict, preferring OmegaConf if available."""
    if OmegaConf is not None:  # type: ignore
        return OmegaConf.to_object(OmegaConf.load(str(path)))  # type: ignore[return-value]
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - fallback if yaml missing
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@dataclass
class _Sample:
    image: Image.Image
    target: Dict[str, Any]


class YOLODetectionDataset(Dataset):
    """YOLO format dataset with optional Albumentations-like augmentation.

    transform: callable(image: PIL.Image, anns: List[dict]) -> (image, anns)
    """

    def __init__(self, data_yaml: Path, split: str = "train", transform: Optional[Callable[[Image.Image, List[Dict[str, Any]]], Tuple[Image.Image, List[Dict[str, Any]]]]] = None):
        self.split = split
        self.transform = transform
        yroot = _read_yaml(data_yaml)
        # Determine image paths for the split
        key = split if split in yroot else ("val" if split == "validation" else "validation")
        if key not in yroot:
            raise KeyError(f"Split '{split}' not present in data yaml")
        entry = yroot[key]
        paths: List[Path] = []

        def _resolve_path(raw: str) -> Path:
            """Resolve raw path strings that may contain incorrect '../' segments.

            Roboflow exports sometimes emit entries like '../train/images' even when
            'train/images' is correct relative to the data.yaml directory. We try a few
            progressively sanitized interpretations before failing.
            """
            p = Path(raw)
            # If already absolute and exists, return
            if p.is_absolute() and p.exists():
                return p
            # First attempt: relative to data.yaml parent (normal case)
            cand = (data_yaml.parent / raw).resolve()
            if cand.exists():
                return cand
            # If it starts with one or more '../', strip them and try within the dataset dir
            if raw.startswith("../"):
                parts = [seg for seg in raw.split("/") if seg and seg != ".."]
                if parts:
                    cand2 = (data_yaml.parent / Path(*parts)).resolve()
                    if cand2.exists():
                        return cand2
            # Final fallback: return original Path (will likely fail later)
            return p

        if isinstance(entry, str):
            p = _resolve_path(entry)
            if p.suffix.lower() == ".txt" and p.exists():
                with p.open() as f:
                    paths = [(_resolve_path(l.strip()) if not Path(l.strip()).is_absolute() else Path(l.strip())) for l in f if l.strip()]
            elif p.is_dir():
                paths = [pp for pp in p.rglob("*") if pp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            else:
                raise FileNotFoundError(f"Split path not found: {p}")
        elif isinstance(entry, list):
            for e in entry:
                ep = _resolve_path(e)
                if ep.is_dir():
                    paths.extend([pp for pp in ep.rglob("*") if pp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
                else:
                    paths.append(ep)
        else:
            raise TypeError("Unsupported split entry type")
        self.images = paths
        names = yroot.get("names")
        if isinstance(names, dict):
            ordered = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
            self.class_names = ordered
        else:
            self.class_names = names or []
        self.data_yaml = data_yaml

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _label_path(img_path: Path) -> Path:
        parts = list(img_path.parts)
        try:
            i = parts.index("images")
            parts[i] = "labels"
        except ValueError:
            pass
        return Path(*parts).with_suffix(".txt")

    def __getitem__(self, idx: int) -> _Sample:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        label_path = self._label_path(img_path)
        anns: List[Dict[str, Any]] = []
        if label_path.exists():
            with label_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, bw, bh = parts
                    c = int(float(cls))
                    fx, fy, fw, fh = map(float, (xc, yc, bw, bh))
                    x_min = (fx - fw / 2.0) * W
                    y_min = (fy - fh / 2.0) * H
                    bw_abs = fw * W
                    bh_abs = fh * H
                    anns.append({"bbox": [x_min, y_min, bw_abs, bh_abs], "category_id": c, "area": bw_abs * bh_abs, "iscrowd": 0})
        if self.transform and self.split == "train":
            try:
                img, anns = self.transform(img, anns)
                W, H = img.size
            except Exception as e:  # pragma: no cover
                print(f"[WARN] augmentation failed idx={idx}: {e}")
        target = {"image_id": idx, "annotations": anns, "width": W, "height": H}
        return _Sample(image=img, target=target)


class TransformersDeformableDetrBackend:
    """Backend that fine-tunes a pretrained Deformable DETR model from HuggingFace.

    This provides a minimal training loop compatible with the existing trainer interface.
    Configure with model.backend=transformers and model.arch="SenseTime/deformable-detr-with-box-refine".
    """

    def __init__(self, arch: str = "SenseTime/deformable-detr-with-box-refine", device: Optional[str] = None):
        self.arch = arch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained(arch)
        model = DeformableDetrForObjectDetection.from_pretrained(arch)
        model.to(self.device)  # type: ignore[arg-type]
        self.model = model  # attribute assignment without inline annotation to appease type checker

    def _prepare_dataloaders(
        self,
        data_yaml: str | Path,
        imgsz: int,
        batch: int,
        aug_builder: Optional[Callable[[], Optional[Callable[[Image.Image, List[Dict[str, Any]]], Tuple[Image.Image, List[Dict[str, Any]]]]]]] = None,
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        data_yaml = Path(data_yaml)
        transform_fn = None
        if aug_builder is not None:
            try:
                transform_fn = aug_builder()
            except Exception as e:  # pragma: no cover
                print(f"[WARN] augmentation builder failed: {e}")
        train_ds = YOLODetectionDataset(data_yaml, split="train", transform=transform_fn)
        try:
            val_ds = YOLODetectionDataset(data_yaml, split="val")
        except Exception:
            try:
                val_ds = YOLODetectionDataset(data_yaml, split="validation")
            except Exception:
                val_ds = train_ds  # fallback

        class_names = train_ds.class_names
        if class_names:
            # Align model classifier and loss to dataset classes (num_classes + background)
            num_classes = len(class_names)
            try:
                # --- Update config (num_labels, id2label/label2id) ---
                cfg = self.model.config  # type: ignore[assignment]
                try:
                    if getattr(cfg, "num_labels", None) != num_classes:  # type: ignore[union-attr]
                        cfg.num_labels = num_classes  # type: ignore[union-attr]
                except Exception:
                    pass
                try:
                    cfg.id2label = {int(i): str(n) for i, n in enumerate(class_names)}  # type: ignore[attr-defined]
                    cfg.label2id = {str(n): int(i) for i, n in enumerate(class_names)}  # type: ignore[attr-defined]
                except Exception:
                    pass

                # --- Replace classifier head to match new class count (num_classes + background) ---
                replaced = False
                for attr in ("class_labels_classifier", "class_predictor", "classifier", "class_embed"):
                    if hasattr(self.model, attr):
                        head = getattr(self.model, attr)
                        in_features = getattr(head, "in_features", None)
                        if isinstance(in_features, int):
                            # preserve dtype/device of old head
                            old_weight = getattr(head, "weight", None)
                            dtype = old_weight.dtype if hasattr(old_weight, "dtype") else None
                            device = old_weight.device if hasattr(old_weight, "device") else self.device
                            new_head = nn.Linear(in_features, num_classes + 1)
                            if dtype is not None:
                                new_head = new_head.to(dtype=dtype)
                            try:
                                new_head = new_head.to(device)  # move to same device as rest of model
                            except Exception:
                                pass
                            setattr(self.model, attr, new_head)
                            replaced = True
                            break
                if not replaced:
                    print("[WARN] Could not locate classifier head to adapt class count; proceeding anyway.")

                # --- Sanity check: ensure config.num_labels matches logits.size(-1) - 1 ---
                try:
                    self.model.eval()
                    with torch.no_grad():
                        # Minimal dry-run with zeros to inspect output shape
                        dummy = torch.zeros((1, 3, imgsz, imgsz), device=self.device)
                        out = self.model(pixel_values=dummy)  # type: ignore[arg-type]
                        logits = getattr(out, "logits", None)
                        if isinstance(logits, torch.Tensor) and logits.ndim == 3:
                            C = int(logits.shape[-1])
                            if getattr(cfg, "num_labels", None) != C - 1:  # type: ignore[union-attr]
                                try:
                                    cfg.num_labels = C - 1  # type: ignore[union-attr]
                                    print(f"[INFO] Adjusted config.num_labels to {cfg.num_labels} to match logits={C}.")
                                except Exception:
                                    pass
                except Exception:
                    # If dry-run fails (e.g., missing processor), skip this check
                    pass
            except Exception as _e:  # pragma: no cover
                print(f"[WARN] Could not adapt classifier head/config to {num_classes} classes: {_e}")

        def collate(samples: List[_Sample]):
            images = [s.image for s in samples]
            annotations = [s.target for s in samples]
            processed = self.image_processor(
                images,
                annotations=annotations,
                return_tensors="pt",
                size={"height": imgsz, "width": imgsz},
            )
            # Debug first batch only: show keys produced by image_processor
            if not hasattr(self, "_collate_debug_done"):
                try:
                    print("[DEBUG collate] processed keys:", list(processed.keys()))
                except Exception:
                    pass
                self._collate_debug_done = True  # type: ignore[attr-defined]
            # Always rebuild labels to ensure expected structure: boxes in cxcywh normalized [0,1]
            labels_list = []
            for ann in annotations:
                anns = ann.get("annotations", [])
                cxcywh = []
                classes = []
                w0 = float(ann.get("width", images[0].width))
                h0 = float(ann.get("height", images[0].height))
                sx = imgsz / w0
                sy = imgsz / h0
                for a in anns:
                    bbox = a.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    x, y, w, h = bbox  # top-left absolute in original dims
                    # scale to resized image space
                    x_s = x * sx
                    y_s = y * sy
                    w_s = w * sx
                    h_s = h * sy
                    # convert to center
                    cx = x_s + w_s / 2.0
                    cy = y_s + h_s / 2.0
                    # normalize by imgsz (after resize width=height=imgsz)
                    cxcywh.append([cx / imgsz, cy / imgsz, w_s / imgsz, h_s / imgsz])
                    classes.append(a.get("category_id", 0))
                if cxcywh:
                    labels_list.append({
                        "boxes": torch.tensor(cxcywh, dtype=torch.float32),
                        "class_labels": torch.tensor(classes, dtype=torch.long),
                    })
                else:
                    labels_list.append({
                        "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        "class_labels": torch.zeros((0,), dtype=torch.long),
                    })
            processed["labels"] = labels_list  # type: ignore[index]
            return processed

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=collate)
        return train_loader, val_loader, class_names

    def train(
        self,
        data: str | Path,
        project: str = "runs/train",
        name: str = "exp",
        imgsz: int = 640,
        epochs: int = 10,
        batch: int = 2,
        lr: float = 1e-5,
        seed: int = 42,
        resume: bool = False,  # noqa: ARG002 - not yet supported
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        out_dir = Path(project) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build augmentation pipeline if provided
        aug_cfg = (extra or {}).get("augmentation") if extra else None

        def _build_aug():  # returns callable or None
            if not aug_cfg or not getattr(aug_cfg, "enable", False):
                return None
            try:
                import albumentations as A  # type: ignore
                import numpy as np  # ensure available for inner closure
            except Exception:
                print("[WARN] Albumentations not installed; skipping augmentations.")
                return None
            # Translate simple scalar config into Albumentations pipeline
            hflip_p = float(getattr(aug_cfg, "hflip", 0.0))
            brightness = float(getattr(aug_cfg, "brightness", 0.0))
            contrast = float(getattr(aug_cfg, "contrast", 0.0))
            hue = float(getattr(aug_cfg, "hue", 0.0))
            saturation = float(getattr(aug_cfg, "saturation", 0.0))
            scale_min = float(getattr(aug_cfg, "scale_min", 1.0))
            scale_max = float(getattr(aug_cfg, "scale_max", 1.0))
            rotate = int(getattr(aug_cfg, "rotate", 0))
            blur_p = float(getattr(aug_cfg, "blur", 0.0))
            cutout_n = int(getattr(aug_cfg, "cutout", 0))

            augs = []
            if hflip_p > 0:
                augs.append(A.HorizontalFlip(p=hflip_p))
            if brightness > 0 or contrast > 0:
                augs.append(A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=0.75,
                ))
            if hue > 0 or saturation > 0:
                # HSV shift: convert hue fraction ~ degrees * 180? We'll approximate.
                augs.append(A.HueSaturationValue(
                    hue_shift_limit=int(hue * 180),
                    sat_shift_limit=int(saturation * 255),
                    val_shift_limit=0,
                    p=0.5,
                ))
            if (scale_min != 1.0 or scale_max != 1.0) or rotate > 0:
                augs.append(A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=max(scale_max - 1.0, 1.0 - scale_min),
                    rotate_limit=rotate,
                    border_mode=0,
                    p=0.7,
                ))
            if blur_p > 0:
                augs.append(A.Blur(blur_limit=3, p=blur_p))
            if cutout_n > 0:
                augs.append(A.Cutout(num_holes=cutout_n, max_w=imgsz // 10, max_h=imgsz // 10, fill_value=0, p=0.5))
            if not augs:
                return None
            transform = A.Compose(
                augs,
                bbox_params=A.BboxParams(format="coco", label_fields=["category_id"], min_visibility=0.1),
            )

            def _apply(pil_img: Image.Image, anns: List[Dict[str, Any]]):
                W, H = pil_img.size
                # Convert possibly out-of-range bboxes to safe COCO (x,y,w,h) within image bounds
                safe_bboxes = []
                labels = []
                for a in anns:
                    bb = a.get("bbox")
                    cid = a.get("category_id", 0)
                    if not bb or len(bb) != 4:
                        continue
                    x, y, w, h = bb
                    # Clamp extents
                    x = max(0.0, min(float(x), W - 1))
                    y = max(0.0, min(float(y), H - 1))
                    w = max(1.0, min(float(w), W - x))
                    h = max(1.0, min(float(h), H - y))
                    # Normalize to relative if values look normalized (>0 and <=1) and image is large? Actually export already absolute; keep as is.
                    safe_bboxes.append([x, y, w, h])
                    labels.append(cid)
                if not safe_bboxes:
                    res = transform(image=np.array(pil_img), bboxes=[], category_id=[])
                    out_img = Image.fromarray(res["image"])  # type: ignore[index]
                    return out_img, []
                res = transform(image=np.array(pil_img), bboxes=safe_bboxes, category_id=labels)
                out_img = Image.fromarray(res["image"])  # type: ignore[index]
                new_anns: List[Dict[str, Any]] = []
                for bb, cid in zip(res.get("bboxes", []), res.get("category_id", [])):
                    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                        continue
                    x, y, w, h = bb
                    # Ensure still within bounds after transform
                    x = max(0.0, min(float(x), out_img.width - 1))
                    y = max(0.0, min(float(y), out_img.height - 1))
                    w = max(1.0, min(float(w), out_img.width - x))
                    h = max(1.0, min(float(h), out_img.height - y))
                    new_anns.append({"bbox": [x, y, w, h], "category_id": int(cid), "area": w * h, "iscrowd": 0})
                return out_img, new_anns

            return _apply

        train_loader, val_loader, class_names = self._prepare_dataloaders(data, imgsz, batch, aug_builder=_build_aug)

        # Optimizer & scheduler
        wd = (extra or {}).get("weight_decay", 1e-4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        warmup_steps = min(100, max(10, len(train_loader)))
        total_steps = epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Informative header so users see progress in the console
        try:
            print(
                f"[INFO] Transformers training start | model={self.arch} device={self.device} "
                f"epochs={epochs} batch={batch} imgsz={imgsz} lr={lr}"
            )
        except Exception:
            pass

        self.model.train()
        history: List[Dict[str, float]] = []

        # Instrument matcher once to debug device placement
        if not hasattr(self, "_matcher_patched"):
            try:
                original_matcher_forward = self.model.loss_function.matcher.forward  # type: ignore[attr-defined]
                device_str = self.device
                def wrapped_matcher_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
                    outputs_, targets_ = args[0], args[1] if len(args) >= 2 else (kwargs.get("outputs"), kwargs.get("targets"))  # type: ignore[index]
                    try:
                        dbg = []
                        for ti, t in enumerate(targets_):  # type: ignore[iteration-over-annotation]
                            if isinstance(t, dict) and "boxes" in t and isinstance(t["boxes"], torch.Tensor):
                                dbg.append(f"t{ti}.boxes:{t['boxes'].device}")
                        print("[MATCHER DEBUG] target boxes devices:", ", ".join(dbg))
                    except Exception as e:  # pragma: no cover
                        print("[MATCHER DEBUG] exception while inspecting targets:", e)
                    return original_matcher_forward(*args, **kwargs)
                self.model.loss_function.matcher.forward = wrapped_matcher_forward  # type: ignore[assignment]
                self._matcher_patched = True  # type: ignore[attr-defined]
            except Exception as e:  # pragma: no cover
                print("[WARN] Could not patch matcher for debug:", e)

        def _to_device(obj: Any):  # recursive move supporting dict/list/tuple
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            if isinstance(obj, dict):
                return {k: _to_device(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_device(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_device(v) for v in obj)
            return obj
        def move_labels_inplace(batch_inputs: Any):
            # Handle dict-like (BatchEncoding) objects generically
            if not hasattr(batch_inputs, "keys"):
                return
            # pixel tensors
            try:
                if "pixel_values" in batch_inputs and isinstance(batch_inputs["pixel_values"], torch.Tensor):  # type: ignore[index]
                    batch_inputs["pixel_values"] = batch_inputs["pixel_values"].to(self.device)  # type: ignore[index]
                if "pixel_mask" in batch_inputs and isinstance(batch_inputs["pixel_mask"], torch.Tensor):  # type: ignore[index]
                    batch_inputs["pixel_mask"] = batch_inputs["pixel_mask"].to(self.device)  # type: ignore[index]
            except Exception:
                pass
            # labels
            try:
                labels = batch_inputs.get("labels")  # type: ignore[arg-type]
            except Exception:
                labels = None
            if isinstance(labels, list):
                for lab in labels:
                    if isinstance(lab, dict):
                        for k, v in list(lab.items()):
                            if isinstance(v, torch.Tensor) and v.device.type != self.device:
                                lab[k] = v.to(self.device)

        # Setup validation metric object once (re-used each epoch)
        map_metric = None
        if MeanAveragePrecision is not None:
            try:
                map_metric = MeanAveragePrecision()
            except Exception as e:  # pragma: no cover
                print(f"[WARN] Could not initialize MeanAveragePrecision: {e}")
                map_metric = None
        # Inference / metric options (used for validation within training loop)
        score_thresh = float((extra or {}).get("val_score_thresh", 0.05))
        with_nms: bool = bool((extra or {}).get("with_nms", False))
        nms_iou: float = float((extra or {}).get("nms_iou", 0.5))
        nms_class_agnostic: bool = bool((extra or {}).get("nms_class_agnostic", True))

        def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
            area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
            area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
            lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
            rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            union = area1[:, None] + area2 - inter
            return inter / union.clamp(min=1e-6)

        def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
            if boxes.numel() == 0:
                return torch.zeros((0,), dtype=torch.long, device=boxes.device)
            try:  # pragma: no cover
                from torchvision.ops import nms as tv_nms  # type: ignore
                return tv_nms(boxes, scores, iou_thresh)
            except Exception:
                pass
            keep: List[int] = []
            idxs = scores.argsort(descending=True)
            while idxs.numel() > 0:
                i = int(idxs[0])
                keep.append(i)
                if idxs.numel() == 1:
                    break
                ious = _box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
                idxs = idxs[1:][ious <= iou_thresh]
            return torch.tensor(keep, device=boxes.device, dtype=torch.long)

        # ------------------ EPOCH LOOP ------------------
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, batch_inputs in enumerate(train_loader):
                move_labels_inplace(batch_inputs)
                # One-time adaptive resize of loss weights to match logits classes
                if epoch == 0 and batch_idx == 0:
                    try:
                        # Build input without labels to infer class dimension cheaply
                        infer_inputs = {k: v for k, v in batch_inputs.items() if k != "labels"}
                        dry = self.model(**infer_inputs)  # type: ignore[arg-type]
                        logits = getattr(dry, "logits", None)
                        if isinstance(logits, torch.Tensor) and logits.ndim >= 3:
                            C = int(logits.shape[-1])
                            crit = getattr(self.model, "loss_function", None)
                            if crit is not None and hasattr(crit, "empty_weight"):
                                # Determine eos coef if present
                                eos_coef = None
                                for coef_name in ("eos_coef", "no_object_weight", "eos_weight"):
                                    if hasattr(crit, coef_name):
                                        eos_coef = float(getattr(crit, coef_name))
                                        break
                                if eos_coef is None:
                                    eos_coef = 0.1
                                new_w = torch.ones(C, device=self.device, dtype=torch.float32)
                                new_w[-1] = float(eos_coef)
                                ew = getattr(crit, "empty_weight")
                                try:
                                    if isinstance(ew, torch.nn.Parameter):
                                        if ew.data.shape != new_w.shape:
                                            ew.data = new_w  # type: ignore[assignment]
                                    else:
                                        if not isinstance(ew, torch.Tensor) or ew.shape != new_w.shape:
                                            setattr(crit, "empty_weight", new_w)
                                except Exception:
                                    setattr(crit, "empty_weight", new_w)
                    except Exception as _adp_e:  # pragma: no cover
                        print(f"[WARN] train adaptive loss weight setup failed: {_adp_e}")
                if epoch == 0 and batch_idx == 0:
                    try:
                        label_list = batch_inputs.get("labels") if isinstance(batch_inputs, dict) else None
                    except Exception:
                        label_list = None
                    pv = batch_inputs.get("pixel_values") if isinstance(batch_inputs, dict) else None
                    if isinstance(pv, torch.Tensor):
                        print("[DEBUG] pixel_values", pv.shape, pv.device)
                    if label_list is None and hasattr(batch_inputs, "get"):
                        try:
                            label_list = batch_inputs.get("labels")  # type: ignore[arg-type]
                        except Exception:
                            label_list = None
                    if isinstance(label_list, list):
                        print(f"[DEBUG] labels list len: {len(label_list)}")
                        for i, lab in enumerate(label_list):
                            if isinstance(lab, dict):
                                cls = lab.get("class_labels")
                                bx = lab.get("boxes")
                                if isinstance(cls, torch.Tensor):
                                    print(f"  labels[{i}].class_labels shape={cls.shape} device={cls.device}")
                                if isinstance(bx, torch.Tensor):
                                    print(f"  labels[{i}].boxes shape={bx.shape} device={bx.device}")
                    else:
                        print("[WARN] No labels present in batch_inputs!")
                outputs = self.model(**batch_inputs)  # type: ignore[arg-type]
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(1, len(train_loader))

            # ----- Validation phase per epoch -----
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                if map_metric is not None:
                    map_metric.reset()
                for batch_inputs in val_loader:
                    move_labels_inplace(batch_inputs)
                    outputs = self.model(**batch_inputs)  # type: ignore[arg-type]
                    val_loss += outputs.loss.item()
                    if map_metric is not None:
                        try:
                            logits = getattr(outputs, "logits", None)
                            pred_boxes = getattr(outputs, "pred_boxes", None)
                            if logits is None or pred_boxes is None:
                                continue
                            probs = logits.softmax(-1)[..., :-1]
                            scores_all, labels_pred = probs.max(-1)
                            boxes_cxcywh = pred_boxes
                            cx, cy, w_, h_ = boxes_cxcywh.unbind(-1)
                            x1 = (cx - 0.5 * w_) * imgsz
                            y1 = (cy - 0.5 * h_) * imgsz
                            x2 = (cx + 0.5 * w_) * imgsz
                            y2 = (cy + 0.5 * h_) * imgsz
                            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0, max=imgsz)
                            B, Q = scores_all.shape
                            batch_preds = []
                            for bi in range(B):
                                sc = scores_all[bi]
                                lb = labels_pred[bi]
                                bx = boxes_xyxy[bi]
                                keep = sc > score_thresh
                                if with_nms and keep.any():
                                    if nms_class_agnostic:
                                        sub_scores = sc[keep]
                                        sub_boxes = bx[keep]
                                        nms_keep_rel = _nms(sub_boxes, sub_scores, nms_iou)
                                        keep_indices = torch.nonzero(keep, as_tuple=False).flatten()
                                        keep_mask = torch.zeros_like(keep)
                                        keep_mask[keep_indices[nms_keep_rel]] = True
                                        keep = keep_mask
                                    else:
                                        train_keep_indices_total: List[int] = []
                                        filtered_indices = torch.nonzero(keep, as_tuple=False).flatten()
                                        flabels = lb[keep]
                                        fboxes = bx[keep]
                                        fscores = sc[keep]
                                        for cls in flabels.unique():
                                            cls_mask = flabels == cls
                                            cls_boxes = fboxes[cls_mask]
                                            cls_scores = fscores[cls_mask]
                                            cls_keep_rel = _nms(cls_boxes, cls_scores, nms_iou)
                                            orig_subset = filtered_indices[cls_mask]
                                            train_keep_indices_total.extend(orig_subset[cls_keep_rel].tolist())
                                        keep_mask = torch.zeros_like(keep)
                                        if train_keep_indices_total:
                                            keep_mask[torch.tensor(train_keep_indices_total, device=keep_mask.device)] = True
                                        keep = keep_mask
                                if keep.any():
                                    batch_preds.append({
                                        "boxes": bx[keep].detach().cpu(),
                                        "scores": sc[keep].detach().cpu(),
                                        "labels": lb[keep].detach().cpu(),
                                    })
                                else:
                                    batch_preds.append({
                                        "boxes": torch.zeros((0, 4)),
                                        "scores": torch.zeros((0,)),
                                        "labels": torch.zeros((0,), dtype=torch.long),
                                    })
                            # Targets conversion
                            tgt_list = []
                            targets = batch_inputs.get("labels") if isinstance(batch_inputs, dict) else None
                            if isinstance(targets, list):
                                for t in targets:
                                    if not isinstance(t, dict):
                                        tgt_list.append({"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
                                        continue
                                    tb = t.get("boxes")
                                    tl = t.get("class_labels")
                                    if isinstance(tb, torch.Tensor) and isinstance(tl, torch.Tensor):
                                        if tb.numel():
                                            cx_t, cy_t, w_t, h_t = tb.unbind(-1)
                                            x1_t = (cx_t - 0.5 * w_t) * imgsz
                                            y1_t = (cy_t - 0.5 * h_t) * imgsz
                                            x2_t = (cx_t + 0.5 * w_t) * imgsz
                                            y2_t = (cy_t + 0.5 * h_t) * imgsz
                                            boxes_t = torch.stack([x1_t, y1_t, x2_t, y2_t], dim=-1)
                                        else:
                                            boxes_t = torch.zeros((0, 4))
                                        tgt_list.append({"boxes": boxes_t.detach().cpu(), "labels": tl.detach().cpu()})
                                    else:
                                        tgt_list.append({"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
                            if tgt_list:
                                map_metric.update(batch_preds, tgt_list)  # type: ignore[arg-type]
                        except Exception as me:  # pragma: no cover
                            print(f"[WARN] mAP metric update failed: {me}")
                val_loss = val_loss / max(1, len(val_loader))
                map_vals: Dict[str, float] = {}
                if map_metric is not None:
                    try:
                        res = map_metric.compute()
                        def _grab(key: str) -> float:
                            v = res.get(key)
                            if v is None:
                                return float('nan')
                            try:
                                if isinstance(v, torch.Tensor):
                                    if v.ndim == 0:
                                        return float(v.item())
                                    if v.numel() > 0:
                                        return float(v.mean().item())
                                    return float('nan')
                            except Exception:
                                pass
                            try:
                                return float(v)  # type: ignore[arg-type]
                            except Exception:
                                return float('nan')
                        map_vals = {"map": _grab("map"), "map50": _grab("map_50"), "map75": _grab("map_75")}
                    except Exception as ce:  # pragma: no cover
                        print(f"[WARN] mAP compute failed: {ce}")
            self.model.train()

            metrics = {"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss, **map_vals}
            history.append(metrics)
            with (out_dir / "history.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            try:
                extra_msg = ""
                if "map" in metrics:
                    m = metrics
                    extra_msg = f" map={m.get('map', float('nan')):.4f} map50={m.get('map50', float('nan')):.4f} map75={m.get('map75', float('nan')):.4f}"
                print(f"[INFO] Epoch {epoch + 1}/{epochs} | train_loss={avg_loss:.4f} val_loss={val_loss:.4f}{extra_msg}")
            except Exception:
                pass

        # Save final model
        saved = False
        try:
            self.model.save_pretrained(out_dir)
            saved = True
        except Exception as e:
            print("[WARN] save_pretrained standard failed:", e)
            try:
                self.model.save_pretrained(out_dir, safe_serialization=False)
                saved = True
            except Exception as e2:
                print("[WARN] save_pretrained safe_serialization=False failed, fallback to state_dict:", e2)
                try:
                    torch.save(self.model.state_dict(), out_dir / "pytorch_model.bin")
                    (out_dir / "config.json").write_text(json.dumps(self.model.config.to_dict(), indent=2))
                    saved = True
                except Exception as e3:
                    print("[ERROR] Could not save model state_dict either:", e3)
        if saved:
            try:
                self.image_processor.save_pretrained(out_dir)
            except Exception as e:
                print("[WARN] Failed to save image processor:", e)
        if class_names:
            (out_dir / "classes.txt").write_text("\n".join(class_names))
        try:
            print(f"[INFO] Training finished. Artifacts saved to: {out_dir}")
        except Exception:
            pass
        return {"history": history, "output_dir": str(out_dir)}

    def validate(
        self,
        data: str | Path,
        project: str = "runs/val",
        name: str = "exp",
        imgsz: int = 640,
        batch: int = 2,
        seed: int = 42,
        extra: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        out_dir = Path(project) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        _, val_loader, class_names = self._prepare_dataloaders(data, imgsz, batch)
        try:
            print(
                f"[INFO] Transformers validation start | model={self.arch} device={self.device} "
                f"batch={batch} imgsz={imgsz}"
            )
        except Exception:
            pass
        self.model.eval()
        losses: List[float] = []
        preds_collected: int = 0
        # Visualization setup
        ex = extra or {}
        score_thresh = float(ex.get("val_score_thresh", 0.05))
        with_nms: bool = bool(ex.get("with_nms", False))
        nms_iou: float = float(ex.get("nms_iou", 0.5))
        nms_class_agnostic: bool = bool(ex.get("nms_class_agnostic", True))
        # Reuse simple NMS utilities (duplicated here to avoid coupling with train local scope)
        def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
            area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
            area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
            lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
            rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            union = area1[:, None] + area2 - inter
            return inter / union.clamp(min=1e-6)
        def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
            if boxes.numel() == 0:
                return torch.zeros((0,), dtype=torch.long, device=boxes.device)
            try:  # pragma: no cover
                from torchvision.ops import nms as tv_nms  # type: ignore
                return tv_nms(boxes, scores, iou_thresh)
            except Exception:
                pass
            keep: List[int] = []
            idxs = scores.argsort(descending=True)
            while idxs.numel() > 0:
                i = int(idxs[0])
                keep.append(i)
                if idxs.numel() == 1:
                    break
                ious = _box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
                idxs = idxs[1:][ious <= iou_thresh]
            return torch.tensor(keep, device=boxes.device, dtype=torch.long)
        vis_max = int(ex.get("val_vis_max", 20))  # number of validation images to dump (0 = disable)
        vis_dir = out_dir / "val_vis"
        if vis_max > 0:
            vis_dir.mkdir(parents=True, exist_ok=True)
        saved_images = 0
        mean = getattr(self.image_processor, "image_mean", [0.0, 0.0, 0.0])
        std = getattr(self.image_processor, "image_std", [1.0, 1.0, 1.0])
        # Setup mAP metric (map, map50, map75) if available
        map_metric = None
        if MeanAveragePrecision is not None:
            try:
                map_metric = MeanAveragePrecision()
            except Exception as e:  # pragma: no cover
                try:
                    print(f"[WARN] Could not initialize MeanAveragePrecision for validation: {e}")
                except Exception:
                    pass
                map_metric = None
        with torch.no_grad():
            for batch_inputs in val_loader:
                batch_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
                # One-time adaptive resize of loss weights to match logits classes (validation)
                if preds_collected == 0:
                    try:
                        infer_inputs = {k: v for k, v in batch_inputs.items() if k != "labels"}
                        dry = self.model(**infer_inputs)
                        logits = getattr(dry, "logits", None)
                        if isinstance(logits, torch.Tensor) and logits.ndim >= 3:
                            C = int(logits.shape[-1])
                            crit = getattr(self.model, "loss_function", None)
                            if crit is not None and hasattr(crit, "empty_weight"):
                                eos_coef = None
                                for coef_name in ("eos_coef", "no_object_weight", "eos_weight"):
                                    if hasattr(crit, coef_name):
                                        eos_coef = float(getattr(crit, coef_name))
                                        break
                                if eos_coef is None:
                                    eos_coef = 0.1
                                new_w = torch.ones(C, device=self.device, dtype=torch.float32)
                                new_w[-1] = float(eos_coef)
                                ew = getattr(crit, "empty_weight")
                                try:
                                    if isinstance(ew, torch.nn.Parameter):
                                        if ew.data.shape != new_w.shape:
                                            ew.data = new_w  # type: ignore[assignment]
                                    else:
                                        if not isinstance(ew, torch.Tensor) or ew.shape != new_w.shape:
                                            setattr(crit, "empty_weight", new_w)
                                except Exception:
                                    setattr(crit, "empty_weight", new_w)
                    except Exception as _adp2_e:  # pragma: no cover
                        try:
                            print(f"[WARN] val adaptive loss weight setup failed: {_adp2_e}")
                        except Exception:
                            pass
                # labels is a list[dict]; move nested tensors
                if isinstance(batch_inputs.get("labels"), list):  # type: ignore[truthy-bool]
                    moved_labels = []
                    for lab in batch_inputs["labels"]:  # type: ignore[index]
                        if isinstance(lab, dict):
                            moved_labels.append({lk: lv.to(self.device) if isinstance(lv, torch.Tensor) else lv for lk, lv in lab.items()})
                        else:
                            moved_labels.append(lab)
                    batch_inputs["labels"] = moved_labels  # type: ignore[index]
                outputs = self.model(**batch_inputs)
                losses.append(outputs.loss.item())
                preds_collected += batch_inputs["pixel_values"].shape[0]

                # mAP metric update
                if map_metric is not None:
                    try:
                        logits = getattr(outputs, "logits", None)
                        pred_boxes = getattr(outputs, "pred_boxes", None)
                        if logits is not None and pred_boxes is not None:
                            probs = logits.softmax(-1)[..., :-1]  # drop background
                            scores, labels_pred = probs.max(-1)
                            boxes_cxcywh = pred_boxes  # (B, Q, 4) normalized
                            cx, cy, w_, h_ = boxes_cxcywh.unbind(-1)
                            x1 = (cx - 0.5 * w_) * imgsz
                            y1 = (cy - 0.5 * h_) * imgsz
                            x2 = (cx + 0.5 * w_) * imgsz
                            y2 = (cy + 0.5 * h_) * imgsz
                            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0, max=imgsz)
                            B, Q = scores.shape
                            batch_preds = []
                            for bi in range(B):
                                sc = scores[bi]
                                lb = labels_pred[bi]
                                bx = boxes_xyxy[bi]
                                keep = sc > score_thresh
                                if with_nms and keep.any():
                                    if nms_class_agnostic:
                                        sub_scores = sc[keep]
                                        sub_boxes = bx[keep]
                                        nms_keep_rel = _nms(sub_boxes, sub_scores, nms_iou)
                                        keep_indices = torch.nonzero(keep, as_tuple=False).flatten()
                                        keep_mask = torch.zeros_like(keep)
                                        keep_mask[keep_indices[nms_keep_rel]] = True
                                        keep = keep_mask
                                    else:
                                        keep_indices_total: List[int] = []
                                        filtered_indices = torch.nonzero(keep, as_tuple=False).flatten()
                                        flabels = lb[keep]
                                        fboxes = bx[keep]
                                        fscores = sc[keep]
                                        for cls in flabels.unique():
                                            cls_mask = flabels == cls
                                            cls_boxes = fboxes[cls_mask]
                                            cls_scores = fscores[cls_mask]
                                            cls_keep_rel = _nms(cls_boxes, cls_scores, nms_iou)
                                            orig_subset = filtered_indices[cls_mask]
                                            keep_indices_total.extend(orig_subset[cls_keep_rel].tolist())
                                        keep_mask = torch.zeros_like(keep)
                                        if keep_indices_total:
                                            keep_mask[torch.tensor(keep_indices_total, device=keep_mask.device)] = True
                                        keep = keep_mask
                                if keep.any():
                                    batch_preds.append({
                                        "boxes": bx[keep].detach().cpu(),
                                        "scores": sc[keep].detach().cpu(),
                                        "labels": lb[keep].detach().cpu(),
                                    })
                                else:
                                    batch_preds.append({
                                        "boxes": torch.zeros((0, 4)),
                                        "scores": torch.zeros((0,)),
                                        "labels": torch.zeros((0,), dtype=torch.long),
                                    })
                            # Targets
                            tgt_list = []
                            targets = batch_inputs.get("labels") if isinstance(batch_inputs, dict) else None
                            if isinstance(targets, list):
                                for t in targets:
                                    if not isinstance(t, dict):
                                        tgt_list.append({"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
                                        continue
                                    tb = t.get("boxes")
                                    tl = t.get("class_labels")
                                    if isinstance(tb, torch.Tensor) and isinstance(tl, torch.Tensor):
                                        if tb.numel():
                                            cx_t, cy_t, w_t, h_t = tb.unbind(-1)
                                            x1_t = (cx_t - 0.5 * w_t) * imgsz
                                            y1_t = (cy_t - 0.5 * h_t) * imgsz
                                            x2_t = (cx_t + 0.5 * w_t) * imgsz
                                            y2_t = (cy_t + 0.5 * h_t) * imgsz
                                            boxes_t = torch.stack([x1_t, y1_t, x2_t, y2_t], dim=-1)
                                        else:
                                            boxes_t = torch.zeros((0, 4))
                                        tgt_list.append({"boxes": boxes_t.detach().cpu(), "labels": tl.detach().cpu()})
                                    else:
                                        tgt_list.append({"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
                            if tgt_list:
                                map_metric.update(batch_preds, tgt_list)  # type: ignore[arg-type]
                    except Exception as me:  # pragma: no cover
                        try:
                            print(f"[WARN] validation mAP metric update failed: {me}")
                        except Exception:
                            pass

                # Visualization logic (pred vs ground truth)
                if vis_max > 0 and saved_images < vis_max:
                    try:
                        logits = getattr(outputs, "logits", None)
                        pred_boxes = getattr(outputs, "pred_boxes", None)
                        if logits is None or pred_boxes is None:
                            continue
                        probs = logits.softmax(-1)[..., :-1]  # drop background
                        scores, labels_pred = probs.max(-1)
                        boxes_cxcywh = pred_boxes  # (B, Q, 4) normalized
                        B, Q, _ = boxes_cxcywh.shape
                        cx, cy, w_, h_ = boxes_cxcywh.unbind(-1)
                        x1 = (cx - 0.5 * w_) * imgsz
                        y1 = (cy - 0.5 * h_) * imgsz
                        x2 = (cx + 0.5 * w_) * imgsz
                        y2 = (cy + 0.5 * h_) * imgsz
                        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0, max=imgsz)

                        labels_list = batch_inputs.get("labels") if isinstance(batch_inputs, dict) else None
                        for bi in range(B):
                            if saved_images >= vis_max:
                                break
                            # Reconstruct image (unnormalize)
                            pv = batch_inputs["pixel_values"][bi].detach().cpu()
                            try:
                                img_np = pv.clone()
                                if isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == img_np.shape[0]:
                                    for c in range(img_np.shape[0]):
                                        img_np[c] = img_np[c] * std[c] + mean[c]
                                img_np = (img_np * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                            except Exception:
                                # Fallback simple scaling
                                img_np = (pv * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                            vis_img = Image.fromarray(img_np)
                            draw = ImageDraw.Draw(vis_img)

                            # Predicted boxes
                            sc = scores[bi]
                            lb = labels_pred[bi]
                            bx = boxes_xyxy[bi]
                            keep = sc > score_thresh
                            if with_nms and keep.any():
                                if nms_class_agnostic:
                                    sub_scores = sc[keep]
                                    sub_boxes = bx[keep]
                                    nms_keep_rel = _nms(sub_boxes, sub_scores, nms_iou)
                                    keep_indices = torch.nonzero(keep, as_tuple=False).flatten()
                                    keep_mask = torch.zeros_like(keep)
                                    keep_mask[keep_indices[nms_keep_rel]] = True
                                    keep = keep_mask
                                else:
                                    vis_keep_indices_total: List[int] = []
                                    filtered_indices = torch.nonzero(keep, as_tuple=False).flatten()
                                    flabels = lb[keep]
                                    fboxes = bx[keep]
                                    fscores = sc[keep]
                                    for cls in flabels.unique():
                                        cls_mask = flabels == cls
                                        cls_boxes = fboxes[cls_mask]
                                        cls_scores = fscores[cls_mask]
                                        cls_keep_rel = _nms(cls_boxes, cls_scores, nms_iou)
                                        orig_subset = filtered_indices[cls_mask]
                                        vis_keep_indices_total.extend(orig_subset[cls_keep_rel].tolist())
                                    keep_mask = torch.zeros_like(keep)
                                    if vis_keep_indices_total:
                                        keep_mask[torch.tensor(vis_keep_indices_total, device=keep_mask.device)] = True
                                    keep = keep_mask
                            for pj in torch.nonzero(keep, as_tuple=False).flatten().tolist():
                                box = bx[pj].detach().cpu().tolist()
                                cls_id = int(lb[pj].detach().cpu().item())
                                conf = float(sc[pj].detach().cpu().item())
                                cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
                                x1b, y1b, x2b, y2b = box
                                draw.rectangle([x1b, y1b, x2b, y2b], outline="red", width=2)
                                draw.text((x1b + 2, y1b + 2), f"P {cls_name}:{conf:.2f}", fill="red")

                            # Ground truth boxes (in labels_list normalized cxcywh)
                            if isinstance(labels_list, list) and bi < len(labels_list):
                                gt_entry = labels_list[bi]
                                if isinstance(gt_entry, dict):
                                    gtb = gt_entry.get("boxes")
                                    gtl = gt_entry.get("class_labels")
                                    if isinstance(gtb, torch.Tensor) and isinstance(gtl, torch.Tensor):
                                        if gtb.numel():
                                            cxg, cyg, wg, hg = gtb.unbind(-1)
                                            x1g = (cxg - 0.5 * wg) * imgsz
                                            y1g = (cyg - 0.5 * hg) * imgsz
                                            x2g = (cxg + 0.5 * wg) * imgsz
                                            y2g = (cyg + 0.5 * hg) * imgsz
                                            gt_xyxy = torch.stack([x1g, y1g, x2g, y2g], dim=-1).detach().cpu()
                                            for gj in range(gt_xyxy.shape[0]):
                                                boxg = gt_xyxy[gj].tolist()
                                                cls_idg = int(gtl[gj].detach().cpu().item())
                                                cls_name_g = class_names[cls_idg] if class_names and cls_idg < len(class_names) else str(cls_idg)
                                                draw.rectangle(boxg, outline="green", width=2)
                                                draw.text((boxg[0] + 2, boxg[1] + 2), f"G {cls_name_g}", fill="green")
                            # Save image
                            out_path = vis_dir / f"val_{saved_images:04d}.jpg"
                            vis_img.save(out_path)
                            saved_images += 1
                    except Exception as viz_e:  # pragma: no cover
                        try:
                            print(f"[WARN] Visualization failure: {viz_e}")
                        except Exception:
                            pass
        avg_loss = sum(losses) / max(1, len(losses))
        map_vals: Dict[str, float] = {}
        if map_metric is not None:
            try:
                res = map_metric.compute()
                def _grab(key: str) -> float:
                    v = res.get(key)
                    if v is None:
                        return float('nan')
                    try:
                        if isinstance(v, torch.Tensor):
                            if v.ndim == 0:
                                return float(v.item())
                            if v.numel() > 0:
                                return float(v.mean().item())
                            return float('nan')
                    except Exception:  # pragma: no cover
                        pass
                    try:
                        return float(v)  # type: ignore[arg-type]
                    except Exception:
                        return float('nan')
                map_vals = {
                    "map": _grab("map"),
                    "map50": _grab("map_50"),
                    "map75": _grab("map_75"),
                }
            except Exception as ce:  # pragma: no cover
                try:
                    print(f"[WARN] validation mAP compute failed: {ce}")
                except Exception:
                    pass
        metrics = {"val_loss": avg_loss, "samples": preds_collected, **map_vals}
        if vis_max > 0:
            metrics["visualizations"] = saved_images
        (out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        try:
            extra_vis = f" visualizations={saved_images} -> {vis_dir}" if vis_max > 0 else ""
            map_msg = ""
            if "map" in metrics:
                map_msg = f" map={metrics.get('map', float('nan')):.4f} map50={metrics.get('map50', float('nan')):.4f} map75={metrics.get('map75', float('nan')):.4f}"
            print(f"[INFO] Validation finished. samples={preds_collected} val_loss={avg_loss:.4f}{map_msg}{extra_vis} -> {out_dir}/val_metrics.json")
        except Exception:
            pass
        return metrics

    def export(self, format: str = "pt", **kwargs: Any) -> Any:  # noqa: D401
        """Export not fully implemented; returns path to saved directory."""
        return {"note": "Use model.save_pretrained(output_dir) after training.", **kwargs}


class TransformersYolosBackend(TransformersDeformableDetrBackend):
    """Backend that fine-tunes a YOLOS model (hustvl/yolos-small-dwr by default).

    Uses the same training/validation loop as the Deformable DETR backend since
    YOLOS follows the DETR-style API on Hugging Face (logits + pred_boxes, labels
    as normalized cxcywh with class_labels).
    """

    def __init__(self, arch: str = "hustvl/yolos-small-dwr", device: Optional[str] = None):
        self.arch = arch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained(arch)
        model = YolosForObjectDetection.from_pretrained(arch)
        model.to(self.device)  # type: ignore[arg-type]
        self.model = model
