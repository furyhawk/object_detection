from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings("ignore", message=r"for .*meta parameter")

import torch
from torch import nn
import numpy as np  # for metrics/array handling
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
    AutoModelForObjectDetection,
    DeformableDetrForObjectDetection,
    YolosForObjectDetection,
    get_cosine_schedule_with_warmup,
    Trainer,
    TrainingArguments,
)
try:
    import supervision as sv  # type: ignore
except Exception:  # pragma: no cover
    sv = None  # type: ignore


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
        # Prefer generic AutoModel to support RT-DETR and other DETR-like models seamlessly.
        try:
            model = AutoModelForObjectDetection.from_pretrained(arch)
        except Exception:
            # Fallback to Deformable DETR when AutoModel isn't available for the checkpoint
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
            # Align model to dataset classes using HF re-init of detection heads
            num_classes = len(class_names)
            id2label = {int(i): str(n) for i, n in enumerate(class_names)}
            label2id = {str(n): int(i) for i, n in enumerate(class_names)}
            try:
                model_overrides = getattr(self, "_model_overrides", {}) or {}
                new_model = AutoModelForObjectDetection.from_pretrained(
                    self.arch,
                    num_labels=num_classes,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                    **model_overrides,
                )
                new_model.to(self.device)  # type: ignore[arg-type]
                self.model = new_model
            except Exception as e:
                try:
                    print(f"[WARN] Couldn't reload model with num_labels={num_classes}: {e}. Falling back to in-place config update.")
                except Exception:
                    pass
                try:
                    cfg = self.model.config  # type: ignore[assignment]
                    cfg.num_labels = num_classes  # type: ignore[attr-defined]
                    cfg.id2label = id2label  # type: ignore[attr-defined]
                    cfg.label2id = label2id  # type: ignore[attr-defined]
                except Exception:
                    pass

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
        # Expose collate function for external Trainer usage
        try:
            self._collate_fn = collate  # type: ignore[attr-defined]
        except Exception:
            pass
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
        # Store optional model overrides for re-init (e.g., RT-DETR anchor_image_size=None)
        try:
            self._model_overrides = (extra or {}).get("model_overrides", {})
        except Exception:
            self._model_overrides = {}

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

        # Use existing dataloader prep to also re-init model heads and build datasets/augmentations
        train_loader, val_loader, class_names = self._prepare_dataloaders(data, imgsz, batch, aug_builder=_build_aug)

        # Informative header so users see progress in the console
        try:
            print(
                f"[INFO] Transformers training start (Trainer) | model={self.arch} device={self.device} "
                f"epochs={epochs} batch={batch} imgsz={imgsz} lr={lr}"
            )
        except Exception:
            pass

        # Build TrainingArguments for HF Trainer
        wd = float((extra or {}).get("weight_decay", 1e-4))
        warmup_steps = min(100, max(10, len(train_loader)))
        # Allow metric threshold override
        try:
            self._metric_score_thresh = float((extra or {}).get("metric_score_thresh", 0.05))
        except Exception:
            self._metric_score_thresh = 0.05  # type: ignore[attr-defined]
        _args: Dict[str, Any] = dict(
            output_dir=str(out_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            learning_rate=lr,
            weight_decay=wd,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=max(1, len(train_loader) // 10),
            save_total_limit=2,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            report_to=[],
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            # Important for detection tasks so label_ids remain a list[dict] per image
            eval_do_concat_batches=False,
        )
        # Handle API variation: some versions use eval_strategy, others evaluation_strategy
        try:
            _args["eval_strategy"] = "epoch"
        except Exception:
            _args["evaluation_strategy"] = "epoch"
        # Prefer saving as PyTorch .bin to avoid shared-tensor issues with safetensors
        # Some HF versions support `save_safetensors`; add it if available, else fall back.
        args: TrainingArguments
        try:
            _args_try = dict(_args)
            _args_try["save_safetensors"] = False  # avoid RuntimeError on shared weights
            args = TrainingArguments(**_args_try)
        except TypeError:
            # Older transformers versions may not accept this kwarg
            args = TrainingArguments(**_args)

        # Trainer expects datasets and a data_collator turning list[Any] -> Batch
        data_collator = getattr(self, "_collate_fn", None)
        if data_collator is None:
            # Fallback: simple wrapper around image_processor (shouldn't happen because _prepare_dataloaders sets it)
            def data_collator(samples: List[_Sample]):  # type: ignore[no-redef]
                images = [s.image for s in samples]
                annotations = [s.target for s in samples]
                processed = self.image_processor(images, annotations=annotations, return_tensors="pt", size={"height": imgsz, "width": imgsz})
                return processed

        # Build torchmetrics-based compute_metrics (MeanAveragePrecision)
        def _build_compute_metrics(imgsz_local: int):
            def _compute_metrics(eval_pred):  # type: ignore[no-redef]
                if MeanAveragePrecision is None:
                    return {}
                predictions = getattr(eval_pred, "predictions", None)
                targets = getattr(eval_pred, "label_ids", None)
                if predictions is None or targets is None:
                    return {}
                # Collect logits and boxes from predictions tuple/list
                seq = predictions if isinstance(predictions, (list, tuple)) else [predictions]
                pred_logits = None
                pred_boxes = None
                for arr in seq:
                    try:
                        sh = arr.shape  # type: ignore[attr-defined]
                    except Exception:
                        continue
                    if hasattr(arr, "shape") and len(sh) >= 3:
                        if sh[-1] == 4 and pred_boxes is None:
                            pred_boxes = arr
                        elif sh[-1] != 4 and pred_logits is None:
                            pred_logits = arr
                if pred_boxes is None or pred_logits is None:
                    return {}
                # Minimal object with logits and pred_boxes for post_process
                class _O:
                    pass
                o = _O()
                o.logits = torch.tensor(pred_logits)
                o.pred_boxes = torch.tensor(pred_boxes)
                N = o.pred_boxes.shape[0]
                tgt_sizes = torch.tensor([(imgsz_local, imgsz_local)] * N)
                try:
                    post = self.image_processor.post_process_object_detection(
                        o,
                        threshold=float(getattr(self, "_metric_score_thresh", 0.05)),
                        target_sizes=tgt_sizes,
                    )
                except Exception:
                    return {}
                # Prepare predictions for torchmetrics (list of dicts)
                preds_tm: List[Dict[str, torch.Tensor]] = []
                for res in post:
                    boxes = res.get("boxes")
                    scores = res.get("scores")
                    labels = res.get("labels")
                    if boxes is None or scores is None or labels is None:
                        preds_tm.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)})
                    else:
                        preds_tm.append({
                            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                            "scores": torch.as_tensor(scores, dtype=torch.float32),
                            "labels": torch.as_tensor(labels, dtype=torch.long),
                        })
                # Prepare targets for torchmetrics (list of dicts, xyxy absolute)
                t_list: List[Dict[str, Any]] = []
                if isinstance(targets, list) and all(isinstance(t, dict) for t in targets):
                    t_list = targets  # type: ignore[assignment]
                elif isinstance(targets, list) and all(isinstance(t, list) for t in targets):
                    t_list = [ti for batch in targets for ti in batch if isinstance(ti, dict)]
                targets_tm: List[Dict[str, torch.Tensor]] = []
                for t in t_list[:len(preds_tm)]:
                    tb = t.get("boxes")
                    tl = t.get("class_labels")
                    tb_t = tb if isinstance(tb, torch.Tensor) else torch.tensor(tb) if tb is not None else torch.zeros((0, 4))
                    tl_t = tl if isinstance(tl, torch.Tensor) else torch.tensor(tl, dtype=torch.long) if tl is not None else torch.zeros((0,), dtype=torch.long)
                    if tb_t.numel():
                        cx, cy, w_, h_ = tb_t.unbind(-1)
                        x1 = (cx - 0.5 * w_) * imgsz_local
                        y1 = (cy - 0.5 * h_) * imgsz_local
                        x2 = (cx + 0.5 * w_) * imgsz_local
                        y2 = (cy + 0.5 * h_) * imgsz_local
                        boxes_xyxy = torch.stack([x1, y1, x2, y2], -1)
                    else:
                        boxes_xyxy = torch.zeros((0, 4))
                    targets_tm.append({
                        "boxes": boxes_xyxy.to(dtype=torch.float32),
                        "labels": tl_t.to(dtype=torch.long),
                    })
                try:
                    mp = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
                    mp.update(preds=preds_tm, target=targets_tm)
                    res = mp.compute()
                    return {
                        "map50_95": float(res.get("map", torch.tensor(0.0)).item()),
                        "map50": float(res.get("map_50", torch.tensor(0.0)).item()),
                        "map75": float(res.get("map_75", torch.tensor(0.0)).item()),
                    }
                except Exception:
                    return {}
            return _compute_metrics

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_loader.dataset,  # type: ignore[arg-type]
            eval_dataset=val_loader.dataset,    # type: ignore[arg-type]
            data_collator=data_collator,        # type: ignore[arg-type]
            compute_metrics=_build_compute_metrics(imgsz) if MeanAveragePrecision is not None else None,
        )

        # Execute training via Trainer
        train_result = trainer.train(resume_from_checkpoint=resume or None)

        # Save final model and processor
        saved = False
        try:
            trainer.save_model(str(out_dir))  # saves model + config
            saved = True
        except Exception as e:
            print("[WARN] Trainer.save_model failed:", e)
            try:
                # Prefer disabling safetensors first to handle shared tensor ties
                self.model.save_pretrained(out_dir, safe_serialization=False)
                saved = True
            except Exception as e2:
                print("[WARN] save_pretrained(safe_serialization=False) failed:", e2)
                try:
                    # Try default behavior as a secondary attempt
                    self.model.save_pretrained(out_dir)
                    saved = True
                except Exception as e3:
                    print("[WARN] save_pretrained default failed, fallback to state_dict:", e3)
                    try:
                        torch.save(self.model.state_dict(), out_dir / "pytorch_model.bin")
                        (out_dir / "config.json").write_text(json.dumps(self.model.config.to_dict(), indent=2))
                        saved = True
                    except Exception as e4:
                        print("[ERROR] Could not save model state_dict either:", e4)
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

        # Return minimal metrics object to keep interface stable
        metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
        return {"metrics": metrics, "output_dir": str(out_dir)}

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
        vis_max = int(ex.get("val_vis_max", 300))  # number of validation images to dump (0 = disable)
        vis_dir = out_dir / "val_vis"
        if vis_max > 0:
            vis_dir.mkdir(parents=True, exist_ok=True)
        saved_images = 0
        mean = getattr(self.image_processor, "image_mean", [0.0, 0.0, 0.0])
        std = getattr(self.image_processor, "image_std", [1.0, 1.0, 1.0])
        # Supervision annotators (if available)
        box_annotator = None
        label_annotator = None
        if sv is not None:
            try:
                box_annotator = sv.BoxAnnotator()
                label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)
            except Exception:
                box_annotator = None
                label_annotator = None
        # Prepare torchmetrics MeanAveragePrecision for evaluation metrics
        use_tm_map = MeanAveragePrecision is not None
        mp = MeanAveragePrecision(box_format="xyxy", iou_type="bbox") if use_tm_map else None
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
                # Update torchmetrics mAP if available
                if use_tm_map and mp is not None:
                    try:
                        B = batch_inputs["pixel_values"].shape[0]
                        tgt_sizes_b = torch.tensor([(imgsz, imgsz)] * B, device=self.device)
                        post = self.image_processor.post_process_object_detection(
                            outputs,
                            threshold=score_thresh,
                            target_sizes=tgt_sizes_b,
                        )
                        preds_tm: List[Dict[str, torch.Tensor]] = []
                        for res in post:
                            boxes = res.get("boxes")
                            scores = res.get("scores")
                            labels = res.get("labels")
                            if boxes is None or scores is None or labels is None:
                                preds_tm.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.long)})
                            else:
                                preds_tm.append({
                                    "boxes": torch.as_tensor(boxes, dtype=torch.float32).to("cpu"),
                                    "scores": torch.as_tensor(scores, dtype=torch.float32).to("cpu"),
                                    "labels": torch.as_tensor(labels, dtype=torch.long).to("cpu"),
                                })
                        targets_tm: List[Dict[str, torch.Tensor]] = []
                        labels_list = batch_inputs.get("labels") if isinstance(batch_inputs, dict) else None
                        if isinstance(labels_list, list):
                            for t in labels_list[:len(preds_tm)]:
                                if not isinstance(t, dict):
                                    targets_tm.append({"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
                                    continue
                                tb = t.get("boxes")
                                tl = t.get("class_labels")
                                tb_t = tb if isinstance(tb, torch.Tensor) else torch.tensor(tb) if tb is not None else torch.zeros((0, 4))
                                tl_t = tl if isinstance(tl, torch.Tensor) else torch.tensor(tl, dtype=torch.long) if tl is not None else torch.zeros((0,), dtype=torch.long)
                                if tb_t.numel():
                                    cx, cy, w_, h_ = tb_t.unbind(-1)
                                    x1 = (cx - 0.5 * w_) * imgsz
                                    y1 = (cy - 0.5 * h_) * imgsz
                                    x2 = (cx + 0.5 * w_) * imgsz
                                    y2 = (cy + 0.5 * h_) * imgsz
                                    boxes_xyxy = torch.stack([x1, y1, x2, y2], -1)
                                else:
                                    boxes_xyxy = torch.zeros((0, 4))
                                targets_tm.append({
                                    "boxes": boxes_xyxy.to(dtype=torch.float32).to("cpu"),
                                    "labels": tl_t.to(dtype=torch.long).to("cpu"),
                                })
                        mp.update(preds=preds_tm, target=targets_tm)
                    except Exception:
                        pass

                # Visualization logic using Supervision (Detections + NMS + Annotators)
                if vis_max > 0 and saved_images < vis_max and box_annotator is not None and label_annotator is not None:
                    try:
                        B = batch_inputs["pixel_values"].shape[0]
                        # Post-process to image-sized predictions (imgsz x imgsz) to match reconstructed image
                        tgt_sizes_b = torch.tensor([(imgsz, imgsz)] * B, device=self.device)
                        post = self.image_processor.post_process_object_detection(
                            outputs,
                            threshold=score_thresh,
                            target_sizes=tgt_sizes_b,
                        )
                        id2label = getattr(self.model.config, "id2label", None)

                        for bi in range(B):
                            if saved_images >= vis_max:
                                break
                            # Reconstruct image (unnormalize) from pixel_values
                            pv = batch_inputs["pixel_values"][bi].detach().cpu()
                            try:
                                img_np = pv.clone()
                                if isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == img_np.shape[0]:
                                    for c in range(img_np.shape[0]):
                                        img_np[c] = img_np[c] * std[c] + mean[c]
                                img_np = (img_np * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                            except Exception:
                                img_np = (pv * 255.0).clamp(0, 255).byte().permute(1, 2, 0).numpy()

                            res = post[bi]
                            # Build Supervision detections
                            det = sv.Detections.from_transformers(
                                transformers_results=res,
                                id2label=id2label,
                            )
                            # Optional NMS
                            if with_nms:
                                try:
                                    det = det.with_nms(threshold=nms_iou)  # type: ignore[attr-defined]
                                except Exception:
                                    from supervision.detection.utils.iou_and_nms import box_non_max_suppression
                                    # Build predictions array for NMS
                                    if det.confidence is None:
                                        confs = np.ones(len(det.xyxy), dtype=float)
                                    else:
                                        confs = det.confidence
                                    if not nms_class_agnostic and det.class_id is not None:
                                        preds = np.c_[det.xyxy, confs, det.class_id]
                                    else:
                                        preds = np.c_[det.xyxy, confs]
                                    keep = box_non_max_suppression(preds, iou_threshold=nms_iou)
                                    # Slice detections
                                    data_sliced = {}
                                    if det.data:
                                        for k, v in det.data.items():
                                            try:
                                                if hasattr(v, "__len__") and len(v) == len(det):
                                                    data_sliced[k] = v[keep]
                                            except Exception:
                                                pass
                                    det = sv.Detections(
                                        xyxy=det.xyxy[keep],
                                        confidence=(det.confidence[keep] if det.confidence is not None else None),
                                        class_id=(det.class_id[keep] if det.class_id is not None else None),
                                        data=data_sliced,
                                    )

                            # Compose labels: class name + confidence
                            pred_class_names = None
                            if det.data and "class_name" in det.data:
                                try:
                                    pred_class_names = det.data["class_name"]
                                except Exception:
                                    pred_class_names = None
                            if pred_class_names is None and det.class_id is not None and id2label is not None:
                                pred_class_names = np.array([id2label.get(int(ci), str(int(ci))) for ci in det.class_id])
                            if det.confidence is not None and pred_class_names is not None:
                                labels = [f"{cn} {cf:.2f}" for cn, cf in zip(pred_class_names, det.confidence)]
                            elif pred_class_names is not None:
                                labels = [str(cn) for cn in pred_class_names]
                            else:
                                labels = None

                            # Annotate predictions
                            canvas = img_np.copy()
                            canvas = box_annotator.annotate(scene=canvas, detections=det)
                            if labels is not None and len(labels) == len(det):
                                canvas = label_annotator.annotate(scene=canvas, detections=det, labels=labels)

                            # Overlay Ground Truth in green
                            try:
                                labels_list = batch_inputs.get("labels") if isinstance(batch_inputs, dict) else None
                                if isinstance(labels_list, list) and bi < len(labels_list):
                                    gt_entry = labels_list[bi]
                                    if isinstance(gt_entry, dict):
                                        gtb = gt_entry.get("boxes")
                                        gtl = gt_entry.get("class_labels")
                                        if isinstance(gtb, torch.Tensor) and isinstance(gtl, torch.Tensor) and gtb.numel():
                                            cxg, cyg, wg, hg = gtb.unbind(-1)
                                            x1g = (cxg - 0.5 * wg) * imgsz
                                            y1g = (cyg - 0.5 * hg) * imgsz
                                            x2g = (cxg + 0.5 * wg) * imgsz
                                            y2g = (cyg + 0.5 * hg) * imgsz
                                            gt_xyxy = torch.stack([x1g, y1g, x2g, y2g], dim=-1).detach().cpu().tolist()
                                            # Draw with PIL to ensure consistent coloring irrespective of annotator API
                                            pil_canvas = Image.fromarray(canvas)
                                            draw = ImageDraw.Draw(pil_canvas)
                                            for gj, boxg in enumerate(gt_xyxy):
                                                cls_idg = int(gtl[gj].detach().cpu().item())
                                                cls_name_g = class_names[cls_idg] if class_names and cls_idg < len(class_names) else str(cls_idg)
                                                draw.rectangle(boxg, outline="green", width=2)
                                                try:
                                                    draw.text((boxg[0] + 2, boxg[1] + 2), f"G {cls_name_g}", fill="green")
                                                except Exception:
                                                    pass
                                            canvas = np.array(pil_canvas)
                            except Exception:
                                pass

                            # Save image
                            out_path = vis_dir / f"val_{saved_images:04d}.jpg"
                            Image.fromarray(canvas).save(out_path)
                            saved_images += 1
                    except Exception as viz_e:  # pragma: no cover
                        try:
                            print(f"[WARN] Visualization failure: {viz_e}")
                        except Exception:
                            pass
        avg_loss = sum(losses) / max(1, len(losses))
        map_vals: Dict[str, float] = {}
        if use_tm_map and mp is not None:
            try:
                res = mp.compute()
                map_vals = {
                    "map50_95": float(res.get("map", torch.tensor(0.0)).item()),
                    "map50": float(res.get("map_50", torch.tensor(0.0)).item()),
                    "map75": float(res.get("map_75", torch.tensor(0.0)).item()),
                }
            except Exception as ce:  # pragma: no cover
                try:
                    print(f"[WARN] MeanAveragePrecision compute failed: {ce}")
                except Exception:
                    pass
        metrics = {"val_loss": avg_loss, "samples": preds_collected, **map_vals}
        if vis_max > 0:
            metrics["visualizations"] = saved_images
        (out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        try:
            extra_vis = f" visualizations={saved_images} -> {vis_dir}" if vis_max > 0 else ""
            map_msg = ""
            if "map50_95" in metrics:
                map_msg = f" map50_95={metrics.get('map50_95', float('nan')):.4f} map50={metrics.get('map50', float('nan')):.4f} map75={metrics.get('map75', float('nan')):.4f}"
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
