from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

# Reuse the YOLO dataset reader to keep one source of truth for data.yaml handling
from .transformers_backend import YOLODetectionDataset  # noqa: F401


class TorchvisionRetinaNetBackend:
    """Backend wrapping torchvision RetinaNet for training/validation.

    Notes:
    - Uses torchvision.models.detection.retinanet_resnet50_fpn as the default arch.
    - Expects YOLO-format dataset (as used elsewhere in this repo).
    - Targets are converted to torchvision's expected format in the collate.
    """

    def __init__(self, arch: str = "retinanet_resnet50_fpn", device: Optional[str] = None):
        self.arch = arch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None  # lazy init after discovering num_classes

    # ---------------------- Data ----------------------
    def _build_dataloaders(
        self,
        data_yaml: str | Path,
        batch: int,
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        dy = Path(data_yaml)
        train_ds = YOLODetectionDataset(dy, split="train")
        try:
            val_ds = YOLODetectionDataset(dy, split="val")
        except Exception:
            try:
                val_ds = YOLODetectionDataset(dy, split="validation")
            except Exception:
                val_ds = train_ds

        class_names = train_ds.class_names or []

        def _to_tv_target(item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
            anns = item.get("annotations", [])
            boxes: List[List[float]] = []
            labels: List[int] = []
            for a in anns:
                bb = a.get("bbox")
                if not bb or len(bb) != 4:
                    continue
                x, y, w, h = map(float, bb)
                boxes.append([x, y, x + w, y + h])
                labels.append(int(a.get("category_id", 0)))
            if boxes:
                b = torch.tensor(boxes, dtype=torch.float32)
                l = torch.tensor(labels, dtype=torch.int64)
            else:
                b = torch.zeros((0, 4), dtype=torch.float32)
                l = torch.zeros((0,), dtype=torch.int64)
            return {"boxes": b, "labels": l}

        def collate_fn(samples: List[Any]):
            images = [to_tensor(s.image) for s in samples]
            targets = [_to_tv_target(s.target) for s in samples]
            return images, targets

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader, class_names

    # ---------------------- Model ----------------------
    def _init_model(self, num_classes: int) -> None:
        if self.model is not None:
            return
        from torchvision.models.detection import retinanet_resnet50_fpn
        # Prefer enum-based weights if available
        weights_arg: Any
        try:
            from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights  # type: ignore
            weights_arg = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        except Exception:
            weights_arg = "DEFAULT"
        # Start from COCO-pretrained then replace classification head to num_classes
        model = retinanet_resnet50_fpn(weights=weights_arg)
        try:
            # Replace classification head to match dataset classes
            ch = model.head.classification_head
            num_anchors = ch.num_anchors  # type: ignore[attr-defined]
            cls_logits = torch.nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
            # Bias init as in RetinaNet paper for rare positives
            prior = 0.01
            bias_value = -math.log((1 - prior) / prior)
            if cls_logits.bias is not None:
                torch.nn.init.constant_(cls_logits.bias, bias_value)
            torch.nn.init.normal_(cls_logits.weight, std=0.01)
            ch.cls_logits = cls_logits  # type: ignore[attr-defined]
            ch.num_classes = num_classes  # type: ignore[attr-defined]
        except Exception:
            # If head structure changes across versions, fall back to constructor with num_classes (no pretrained head)
            model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes)
        model.to(self.device)
        self.model = model

    # ---------------------- Train/Val ----------------------
    def train(
        self,
        data: str | Path,
        project: str = "runs/train",
        name: str = "exp",
        imgsz: int = 640,  # noqa: ARG002 - not used by torchvision backend (keeps native sizes)
        epochs: int = 20,
        batch: int = 4,
        lr: float = 1e-3,
        seed: int = 42,
        resume: bool = False,  # noqa: ARG002 - not implemented
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        out_dir = Path(project) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, class_names = self._build_dataloaders(data, batch)
        num_classes = max(1, len(class_names) or 1)
        self._init_model(num_classes)

        model = self.model
        assert model is not None, "Model not initialized"
        params = [p for p in model.parameters() if p.requires_grad]
        wd = float((extra or {}).get("weight_decay", 1e-4))
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)

        # Metric
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision  # type: ignore
            map_metric = MeanAveragePrecision()
        except Exception:
            map_metric = None

        history: List[Dict[str, float]] = []
        for ep in range(epochs):
            model.train()
            train_loss = 0.0
            for images, targets in train_loader:
                images = [im.to(self.device) for im in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                out = model(images, targets)  # type: ignore[operator]
                # Torchvision returns a dict of losses during training; if a list is returned,
                # it indicates predictions (e.g., when losses cannot be computed for the batch).
                if isinstance(out, dict):
                    loss = sum(v for v in out.values())
                else:
                    # Skip optimization for this batch (e.g., all-empty targets). Count as zero loss.
                    loss = torch.tensor(0.0, device=self.device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach().item())
            lr_sched.step()
            avg_train = train_loss / max(1, len(train_loader))

            # Validation
            model.eval()
            val_loss = 0.0
            if map_metric is not None:
                map_metric.reset()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [im.to(self.device) for im in images]
                    targets_dev = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    # Compute val loss
                    out = model(images, targets_dev)  # type: ignore[operator]
                    if isinstance(out, dict):
                        loss = sum(v for v in out.values())
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                    val_loss += float(loss.detach().item())
                    # mAP
                    if map_metric is not None:
                        preds = model(images)  # type: ignore[operator]
                        # Move to CPU for metric
                        preds_cpu = [
                            {"boxes": p["boxes"].detach().cpu(), "scores": p["scores"].detach().cpu(), "labels": p["labels"].detach().cpu()}  # type: ignore[index]
                            for p in preds
                        ]
                        tg_cpu = [
                            {"boxes": t["boxes"].detach().cpu(), "labels": t["labels"].detach().cpu()} for t in targets_dev
                        ]
                        try:
                            map_metric.update(preds_cpu, tg_cpu)  # type: ignore[arg-type]
                        except Exception:
                            pass
            avg_val = val_loss / max(1, len(val_loader))

            maps: Dict[str, float] = {}
            if map_metric is not None:
                try:
                    comp = map_metric.compute()
                    def _flt(x: Any) -> float:
                        try:
                            import torch as _t
                            if isinstance(x, _t.Tensor):
                                return float(x.item()) if x.ndim == 0 else float(x.mean().item())
                            return float(x)
                        except Exception:
                            return float("nan")
                    maps = {
                        "map": _flt(comp.get("map")),
                        "map50": _flt(comp.get("map_50")),
                        "map75": _flt(comp.get("map_75")),
                    }
                except Exception:
                    maps = {}

            rec = {"epoch": ep + 1, "train_loss": avg_train, "val_loss": avg_val, **maps}
            history.append(rec)
            (out_dir / "history.jsonl").open("a").write(json.dumps(rec) + "\n")
            try:
                extra_msg = (
                    f" map={maps.get('map', float('nan')):.4f} map50={maps.get('map50', float('nan')):.4f} map75={maps.get('map75', float('nan')):.4f}"
                    if maps else ""
                )
                print(f"[INFO] Epoch {ep + 1}/{epochs} | train_loss={avg_train:.4f} val_loss={avg_val:.4f}{extra_msg}")
            except Exception:
                pass

        # Save final weights and metadata
        try:
            torch.save(model.state_dict(), out_dir / "retinanet_state_dict.pt")
        except Exception as e:
            try:
                print(f"[WARN] Failed to save state_dict: {e}")
            except Exception:
                pass
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
        imgsz: int = 640,  # noqa: ARG002
        batch: int = 4,
        seed: int = 42,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        out_dir = Path(project) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        _val_only_loader, val_loader, class_names = self._build_dataloaders(data, batch)
        # Build model with dataset classes if not already
        num_classes = max(1, len(class_names) or 1)
        self._init_model(num_classes)
        model = self.model
        assert model is not None, "Model not initialized"

        # If a state_dict path is supplied via arch and exists, try loading
        arch_path = Path(self.arch)
        if arch_path.suffix == ".pt" and arch_path.is_file():
            try:
                state = torch.load(arch_path, map_location=self.device)
                model.load_state_dict(state)
                print(f"[INFO] Loaded weights from {arch_path}")
            except Exception as e:
                print(f"[WARN] Could not load weights '{arch_path}': {e}")

        # Metric
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision  # type: ignore
            map_metric = MeanAveragePrecision()
        except Exception:
            map_metric = None

        model.eval()
        losses: List[float] = []
        preds_cnt = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [im.to(self.device) for im in images]
                targets_dev = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                # Compute validation loss
                out = model(images, targets_dev)  # type: ignore[operator]
                if isinstance(out, dict):
                    loss = sum(v for v in out.values())
                else:
                    loss = torch.tensor(0.0, device=self.device)
                losses.append(float(loss.detach().item()))
                preds_cnt += len(images)
                # Compute predictions for mAP
                if map_metric is not None:
                    preds = model(images)  # type: ignore[operator]
                    preds_cpu = [
                        {"boxes": p["boxes"].detach().cpu(), "scores": p["scores"].detach().cpu(), "labels": p["labels"].detach().cpu()}  # type: ignore[index]
                        for p in preds
                    ]
                    tg_cpu = [
                        {"boxes": t["boxes"].detach().cpu(), "labels": t["labels"].detach().cpu()} for t in targets
                    ]
                    try:
                        map_metric.update(preds_cpu, tg_cpu)  # type: ignore[arg-type]
                    except Exception:
                        pass

        avg_loss = sum(losses) / max(1, len(losses))
        maps: Dict[str, float] = {}
        if map_metric is not None:
            try:
                comp = map_metric.compute()
                def _flt(x: Any) -> float:
                    try:
                        import torch as _t
                        if isinstance(x, _t.Tensor):
                            return float(x.item()) if x.ndim == 0 else float(x.mean().item())
                        return float(x)
                    except Exception:
                        return float("nan")
                maps = {"map": _flt(comp.get("map")), "map50": _flt(comp.get("map_50")), "map75": _flt(comp.get("map_75"))}
            except Exception:
                maps = {}

        metrics = {"val_loss": avg_loss, "samples": preds_cnt, **maps}
        (out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        try:
            m = maps
            msg = (
                f" map={m.get('map', float('nan')):.4f} map50={m.get('map50', float('nan')):.4f} map75={m.get('map75', float('nan')):.4f}"
                if m else ""
            )
            print(f"[INFO] Validation finished. samples={preds_cnt} val_loss={avg_loss:.4f}{msg} -> {out_dir}/val_metrics.json")
        except Exception:
            pass
        return metrics

    def export(self, format: str = "pt", **kwargs: Any) -> Any:  # noqa: D401
        """Export returns a simple note; use saved state_dict from training directory."""
        return {"note": "Use torch.save(model.state_dict(), path) to export weights.", **kwargs}
