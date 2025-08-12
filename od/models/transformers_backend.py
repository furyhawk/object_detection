from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore", message=r"for .*meta parameter")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
    """Dataset that reads a YOLO-style data.yaml and label txt files and produces COCO-like annotations.

    Expected label format per line: class x_center y_center width height (all normalized 0-1)
    The annotation dict produced follows the subset expected by HuggingFace image processors:
      {"image_id": int, "annotations": [{"bbox": [x_min, y_min, w, h], "category_id": int}, ...]}
    """

    def __init__(self, data_yaml: Path, split: str = "train"):
        yroot = _read_yaml(data_yaml)
        # Resolve split path(s)
        if split not in yroot:
            # Some YAMLs use 'val' vs 'validation'
            alt = "val" if split == "validation" else "validation"
            if alt in yroot:
                split_key = alt
            else:
                raise KeyError(f"Split '{split}' not found in data yaml: keys={list(yroot.keys())}")
        else:
            split_key = split
        split_entry = yroot[split_key]
        if isinstance(split_entry, str):
            # Could be path to a directory or a txt file listing images
            split_paths: List[Path]
            p = Path(split_entry)
            if p.suffix.lower() == ".txt" and p.exists():
                with p.open() as f:
                    split_paths = [Path(line.strip()) for line in f if line.strip()]
            else:
                # gather images under directory recursively
                if p.is_dir():
                    split_paths = [pp for pp in p.rglob("*") if pp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
                else:
                    raise FileNotFoundError(f"Split path not found: {p}")
        elif isinstance(split_entry, list):
            split_paths = []
            for entry in split_entry:
                ep = Path(entry)
                if ep.is_dir():
                    split_paths.extend([pp for pp in ep.rglob("*") if pp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
                else:
                    split_paths.append(ep)
        else:
            raise TypeError(f"Unsupported split entry type: {type(split_entry)}")

        names = yroot.get("names")
        if isinstance(names, dict):  # sometimes dict of id:name
            # Ensure order by id
            ordered = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
            self.class_names = ordered
        else:
            self.class_names = names or []

        self.images: List[Path] = split_paths
        self.data_yaml = data_yaml
        self.split = split

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, img_path: Path) -> Path:
        # Typical structure: images/... -> labels/... with same stem and .txt
        parts = list(img_path.parts)
        try:
            idx = parts.index("images")
            parts[idx] = "labels"
        except ValueError:
            # Fallback: just replace parent dir name if it endswith images
            if img_path.parent.name.startswith("image"):
                parts[-2] = "labels"
        return Path(*parts).with_suffix(".txt")

    def __getitem__(self, idx: int) -> _Sample:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        label_path = self._label_path(img_path)
        ann_list: List[Dict[str, Any]] = []
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
                    # Convert to top-left width height absolute
                    x_min = (fx - fw / 2.0) * w
                    y_min = (fy - fh / 2.0) * h
                    bw_abs = fw * w
                    bh_abs = fh * h
                    ann_list.append({
                        "bbox": [x_min, y_min, bw_abs, bh_abs],
                        "category_id": c,
                        "area": bw_abs * bh_abs,
                        "iscrowd": 0,
                    })
        target = {"image_id": idx, "annotations": ann_list, "width": w, "height": h}
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
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        data_yaml = Path(data_yaml)
        train_ds = YOLODetectionDataset(data_yaml, split="train")
        try:
            val_ds = YOLODetectionDataset(data_yaml, split="val")
        except Exception:
            try:
                val_ds = YOLODetectionDataset(data_yaml, split="validation")
            except Exception:
                val_ds = train_ds  # fallback

        class_names = train_ds.class_names
        if class_names:
            # Align model classifier if needed (resize num_classes inc background)
            num_classes = len(class_names)
            if hasattr(self.model, "class_labels_classifier") and self.model.config.num_labels != num_classes:  # type: ignore[attr-defined]
                # Adjust classifier output (add background token)
                in_features = getattr(self.model.class_labels_classifier, "in_features", None)  # type: ignore[attr-defined]
                if isinstance(in_features, int):
                    self.model.class_labels_classifier = nn.Linear(in_features, num_classes + 1)  # type: ignore[assignment]
                    self.model.config.num_labels = num_classes

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

        train_loader, val_loader, class_names = self._prepare_dataloaders(data, imgsz, batch)

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

        score_thresh = float((extra or {}).get("val_score_thresh", 0.05))

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, batch_inputs in enumerate(train_loader):
                # Move tensors (including labels) onto target device
                move_labels_inplace(batch_inputs)
                if epoch == 0 and batch_idx == 0:
                    # One-time verbose debug for first epoch / each batch
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

            # Validation: compute loss and (optionally) detection metrics (mAP)
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
                            # Extract predictions
                            logits = getattr(outputs, "logits", None)
                            pred_boxes = getattr(outputs, "pred_boxes", None)
                            if logits is None or pred_boxes is None:
                                continue
                            probs = logits.softmax(-1)[..., :-1]  # drop background
                            scores, labels_pred = probs.max(-1)
                            boxes_cxcywh = pred_boxes  # normalized cxcywh in 0..1
                            # Convert to xyxy absolute (after resize all imgs are imgsz x imgsz)
                            cx, cy, w_, h_ = boxes_cxcywh.unbind(-1)
                            x1 = (cx - 0.5 * w_) * imgsz
                            y1 = (cy - 0.5 * h_) * imgsz
                            x2 = (cx + 0.5 * w_) * imgsz
                            y2 = (cy + 0.5 * h_) * imgsz
                            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                            # Clamp
                            boxes_xyxy = boxes_xyxy.clamp(min=0, max=imgsz)

                            batch_preds = []
                            B, Q = scores.shape
                            for bi in range(B):
                                sc = scores[bi]
                                lb = labels_pred[bi]
                                bx = boxes_xyxy[bi]
                                keep = sc > score_thresh
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

                            # Targets: convert normalized cxcywh to xyxy absolute
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
                                            cx, cy, w_, h_ = tb.unbind(-1)
                                            x1 = (cx - 0.5 * w_) * imgsz
                                            y1 = (cy - 0.5 * h_) * imgsz
                                            x2 = (cx + 0.5 * w_) * imgsz
                                            y2 = (cy + 0.5 * h_) * imgsz
                                            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                                        else:
                                            boxes = torch.zeros((0, 4))
                                        tgt_list.append({"boxes": boxes.detach().cpu(), "labels": tl.detach().cpu()})
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
                        computed = {k: float(v) for k, v in map_metric.compute().items()}  # type: ignore[assignment]
                        # Standard keys: map, map_50, map_75
                        map_vals = {
                            "map": computed.get("map", float("nan")),
                            "map50": computed.get("map_50", float("nan")),
                            "map75": computed.get("map_75", float("nan")),
                        }
                    except Exception as ce:  # pragma: no cover
                        print(f"[WARN] mAP compute failed: {ce}")
            self.model.train()

            metrics = {"epoch": epoch + 1, "train_loss": avg_loss, "val_loss": val_loss, **map_vals}
            history.append(metrics)
            # Persist running log
            with (out_dir / "history.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")

            # Console progress output (one line per epoch)
            try:
                extra_msg = ""
                if "map" in metrics:
                    m = metrics
                    extra_msg = f" map={m.get('map', float('nan')):.4f} map50={m.get('map50', float('nan')):.4f} map75={m.get('map75', float('nan')):.4f}"
                print(
                    f"[INFO] Epoch {epoch + 1}/{epochs} | train_loss={avg_loss:.4f} val_loss={val_loss:.4f}{extra_msg}"
                )
            except Exception:
                pass

        # Save final model with fallbacks for shared tensor warning
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
                    # minimal config
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
        with torch.no_grad():
            for batch_inputs in val_loader:
                batch_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
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
        avg_loss = sum(losses) / max(1, len(losses))
        metrics = {"val_loss": avg_loss, "samples": preds_collected}
        (out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        try:
            print(f"[INFO] Validation finished. samples={preds_collected} val_loss={avg_loss:.4f} -> {out_dir}/val_metrics.json")
        except Exception:
            pass
        return metrics

    def export(self, format: str = "pt", **kwargs: Any) -> Any:  # noqa: D401
        """Export not fully implemented; returns path to saved directory."""
        return {"note": "Use model.save_pretrained(output_dir) after training.", **kwargs}
