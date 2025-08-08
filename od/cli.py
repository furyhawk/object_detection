from __future__ import annotations

import argparse
from typing import Any, Dict
from dotenv import load_dotenv
import os

from od.utils.config import Config
from od.trainers import run_train, run_validate
from od.data import ensure_roboflow_dataset


def build_base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Object Detection")
    p.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--override", "-o", action="append", default=[], help="Key=Value overrides (dotlist)")
    return p

def build_main_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Object Detection")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "validate", "download"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config file")
        sp.add_argument("--override", "-o", action="append", default=[], help="Key=Value overrides (dotlist)")
    return p


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in pairs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        # simple casting
        if v.lower() in {"true", "false"}:
            vv: Any = v.lower() == "true"
        else:
            try:
                vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    vv = v
        # support dot keys like model.lr
        cur = out
        parts = k.split(".")
        for px in parts[:-1]:
            if px not in cur or not isinstance(cur[px], dict):
                cur[px] = {}
            cur = cur[px]  # type: ignore[assignment]
        cur[parts[-1]] = vv
    return out


def main_train() -> None:
    load_dotenv()  # load env vars from .env if present
    args = build_base_parser().parse_args()
    cfg = Config.load(args.config, overrides=parse_overrides(args.override))
    run_train(cfg)


def main_validate() -> None:
    load_dotenv()
    args = build_base_parser().parse_args()
    cfg = Config.load(args.config, overrides=parse_overrides(args.override))
    run_validate(cfg)


def main_download() -> None:
    load_dotenv()
    args = build_base_parser().parse_args()
    cfg = Config.load(args.config, overrides=parse_overrides(args.override))
    path = ensure_roboflow_dataset(
        api_key=cfg.roboflow.api_key,
        workspace=cfg.roboflow.workspace,
        project=cfg.roboflow.project,
        version=cfg.roboflow.version,
        split=cfg.roboflow.split,
        location=cfg.roboflow.location,
        format=cfg.roboflow.format,
    )
    from od.data import find_data_yaml
    yaml_path = find_data_yaml(path)
    print(f"Dataset downloaded to: {path}")
    if yaml_path:
        print(f"Detected data.yaml: {yaml_path}")
    else:
        # If empty, exit with helpful message
        try:
            has_files = any(path.iterdir())
        except Exception:
            has_files = False
        if not has_files:
            print("Error: No files were created in the download directory.")
            print("Troubleshooting:")
            print("- Verify your ROBOFLOW_API_KEY has access to this dataset.")
            print("- Confirm the dataset export exists for the chosen format (yolov8) and you have accepted any Universe EULA.")
            print("- If this is a Universe dataset, open the dataset page in your browser and click Download once to generate the export.")
            if cfg.roboflow.workspace and cfg.roboflow.project and cfg.roboflow.version:
                print(
                    f"  URL: https://universe.roboflow.com/{cfg.roboflow.workspace}/{cfg.roboflow.project}/dataset/{cfg.roboflow.split}/{cfg.roboflow.version}"
                )
            print("- Then re-run: od-download -c", args.config)
            raise SystemExit(1)


if __name__ == "__main__":
    # Allow running as module: python -m od.cli train -c configs/example.yaml
    load_dotenv()
    parser = build_main_parser()
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    cfg = Config.load(args.config, overrides=overrides)
    if args.command == "train":
        run_train(cfg)
    elif args.command == "validate":
        run_validate(cfg)
    elif args.command == "download":
        path = ensure_roboflow_dataset(
            api_key=cfg.roboflow.api_key,
            workspace=cfg.roboflow.workspace,
            project=cfg.roboflow.project,
            version=cfg.roboflow.version,
            split=cfg.roboflow.split,
            location=cfg.roboflow.location,
            format=cfg.roboflow.format,
        )
        from od.data import find_data_yaml
        yaml_path = find_data_yaml(path)
        print(f"Dataset downloaded to: {path}")
        if yaml_path:
            print(f"Detected data.yaml: {yaml_path}")
