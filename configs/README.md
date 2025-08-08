# Configs

Keys under `roboflow`:
- workspace, project, version: identify the dataset
- split: export format (use `yolov8` for Ultralytics)
- location: where to download
- data_yaml (optional): explicit path to `data.yaml` if auto-detection fails

Keys under `model`:
- backend: `ultralytics`
- arch: e.g., `yolov8n.pt` or a local checkpoint path
- device: `cpu`, `cuda:0`, or `0,1` for multi-GPU as a string

Train/val sections control run directories and names.Configs live here. Start from `example.yaml` and customize. Keys:

- roboflow.workspace, roboflow.project, roboflow.version: required to download
- roboflow.split: export format (yolov8 recommended)
- model.*: ultralytics settings like arch, imgsz, epochs, batch
- train.*, val.*: output dirs and run names
