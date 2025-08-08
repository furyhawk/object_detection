# Object Detection - Config-Driven Training/Validation

This project now includes a small framework to train and validate object detection models using configuration files, with dataset downloads from Roboflow and an Ultralytics YOLO backend.

## Features

- Config (YAML) controls dataset (Roboflow), model, and run settings
- Roboflow integration: download datasets on demand
- Ultralytics backend for training/validation
- Simple CLI entry points

## Install

This repo uses `uv` with pyproject. Create an env and install:

```bash
uv sync
```

Ensure you have a working PyTorch install compatible with your GPU/CPU. The `pyproject.toml` includes indexes for Linux CUDA 12.8 wheels.

Environment variables from a `.env` file are loaded automatically. Create `.env` with:

```env
ROBOFLOW_API_KEY=your_key_here
```

## Configure

Copy and edit the example config:

```bash
cp configs/example.yaml configs/my.yaml
```

Set your dataset source and details in `configs/my.yaml`:
- `dataset.source`: `roboflow` (default) or `cvat`
	- If `roboflow`, set: `roboflow.workspace`, `roboflow.project`, `roboflow.version` and have `ROBOFLOW_API_KEY` in env or `roboflow.api_key` in config.
	- If `cvat`, provide either `cvat.zip_path` (a CVAT export zip) or `cvat.root` (an extracted directory). The tool will try to generate a `data.yaml` if missing.

For Roboflow, typical fields:

- `workspace`, `project`, `version`
- Optionally set `ROBOFLOW_API_KEY` env var, or put `roboflow.api_key` in the config.

## Usage

Train:

```bash
od-train -c configs/my.yaml
```

Validate:

```bash
od-validate -c configs/my.yaml
```

Download or prepare dataset only:

```bash
od-download -c configs/my.yaml
```

Override options at runtime (dotlist):

```bash
od-train -c configs/my.yaml -o model.epochs=100 -o model.arch=yolov8s.pt -o model.device=cuda:0
```

Outputs are written to `runs/train/...` and `runs/val/...` following Ultralytics conventions.

## Notes

- Currently only the Ultralytics backend is wired. The structure allows adding other backends (e.g., Hugging Face DETR) later.
- Roboflow export format should be `yolov8` to include a `data.yaml` compatible with Ultralytics.
- For CVAT, both YOLO and COCO exports are supported; YOLO is recommended for easiest setup. If no `data.yaml` exists, we'll try to create one based on the images/labels layout.

