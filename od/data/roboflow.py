from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Iterable

from roboflow import Roboflow
from dotenv import load_dotenv
import zipfile
import glob
import tempfile
import shutil
import requests


def _cleanup_partial_zip(location: str | os.PathLike[str]) -> None:
    loc = Path(location)
    loc.mkdir(parents=True, exist_ok=True)
    # remove any leftover roboflow*.zip files that may be corrupt/partial
    for p in glob.glob(str(loc / "roboflow*.zip")):
        try:
            Path(p).unlink()
        except Exception:
            pass


def _iter_dir(path: Path) -> Iterable[Path]:
    try:
        yield from path.iterdir()
    except FileNotFoundError:
        return


def _is_nonempty_dir(path: Path) -> bool:
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False
    except Exception:
        return False


def download_roboflow(
    api_key: Optional[str],
    workspace: str,
    project: str,
    version: int,
    split: str = "yolov8",
    location: str = "./data",
    format: Optional[str] = None,
) -> Path:
    """
    Download a dataset from Roboflow and return the local path to the dataset version folder.
    split: roboflow dataset export format name (e.g., 'yolov8', 'coco')
    """
    load_dotenv()
    key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Roboflow API key not provided. Set ROBOFLOW_API_KEY env or pass api_key.")

    rf = Roboflow(api_key=key)
    print("loading Roboflow workspace...")
    ws = rf.workspace(workspace)
    print("loading Roboflow project...")
    proj = ws.project(project)
    ver = proj.version(version)

    export_format = format or split
    _cleanup_partial_zip(location)
    dataset = ver.download(export_format, location=location)

    # Post-download detection and diagnostics
    loc_dir = Path(location).resolve()
    ds_path = None
    if hasattr(dataset, "location"):
        try:
            ds_path = Path(getattr(dataset, "location")).resolve()
        except Exception:
            ds_path = None
    if ds_path is None:
        # Some SDK versions return a string path
        try:
            ds_path = Path(str(dataset)).resolve()
        except Exception:
            ds_path = None

    # Prefer the directory that actually contains data.yaml
    yaml_path = find_data_yaml(loc_dir)
    if yaml_path:
        print(f"Detected data.yaml at: {yaml_path}")
        return yaml_path.parent

    # If the top-level location has no yaml, scan common subfolders created by Roboflow
    subdirs = [p for p in _iter_dir(loc_dir) if p.is_dir()]
    for sub in subdirs:
        yp = find_data_yaml(sub)
        if yp:
            print(f"Detected data.yaml at: {yp}")
            return yp.parent

    # Fallbacks: if ds_path is a directory and non-empty, return that; else continue
    if ds_path and ds_path.exists():
        if ds_path.is_dir() and _is_nonempty_dir(ds_path):
            print(f"Using dataset path reported by SDK: {ds_path}")
            return ds_path
        # If it's a file (zip), keep location as parent; SDK should have extracted already
        print(f"SDK returned a file path ({ds_path}); using download location: {loc_dir}")
        # fall through to additional checks

    # If top-level location is non-empty, return it even if no yaml (let caller inspect)
    if _is_nonempty_dir(loc_dir):
        print(f"No data.yaml found but download location is non-empty: {loc_dir}")
        return loc_dir

    # If we reach here, SDK call didn't raise but we couldn't find files; try manual HTTP fallback
    try:
        print("SDK download produced no files. Attempting manual HTTP download from Roboflow Universe...")
        dl_path = _manual_download_roboflow_zip(
            api_key=key,
            workspace=workspace,
            project=project,
            version=version,
            export_format=export_format,
            out_dir=loc_dir,
        )
        yp = find_data_yaml(dl_path)
        if yp:
            print(f"Detected data.yaml at: {yp}")
            return yp.parent
        return dl_path
    except Exception as e:
        print(f"Manual fallback failed: {e}")

    # Last resort: warn and return location for caller to inspect
    print("Warning: Could not auto-detect dataset folder with data.yaml. Returning download location.")
    # Quick listing to help users understand state
    try:
        listing = ", ".join(sorted(p.name for p in _iter_dir(loc_dir))) or "<empty>"
        print(f"Contents of {loc_dir}: {listing}")
    except Exception:
        pass
    return loc_dir


def _manual_download_roboflow_zip(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    export_format: str,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Try a few known endpoints; some may redirect or require download flag
    urls = [
        f"https://universe.roboflow.com/{workspace}/{project}/dataset/{export_format}/{version}?api_key={api_key}&download=true",
        f"https://universe.roboflow.com/{workspace}/{project}/dataset/{export_format}/{version}?api_key={api_key}",
        f"https://api.roboflow.com/dataset/{workspace}/{project}/{version}/{export_format}?api_key={api_key}",
    ]
    last_err: Exception | None = None
    with tempfile.TemporaryDirectory() as td:
        tmp_zip = Path(td) / "rf.zip"
        for url in urls:
            try:
                print(f"Downloading {url} ...")
                with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
                    ct = r.headers.get('Content-Type', '')
                    print(f"HTTP {r.status_code}, Content-Type: {ct}")
                    r.raise_for_status()
                    with open(tmp_zip, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                size = tmp_zip.stat().st_size
                print(f"Download complete: {size} bytes.")
                # Basic sanity: zip files start with 'PK' signature
                with open(tmp_zip, 'rb') as f:
                    sig = f.read(2)
                if sig != b'PK':
                    # Try to print a small snippet for debugging
                    try:
                        text = Path(tmp_zip).read_text(errors='ignore')
                        snippet = text[:200].replace('\n', ' ')
                        print(f"Downloaded file is not a ZIP. First 200 chars: {snippet}")
                    except Exception:
                        pass
                    print("Trying next URL...")
                    continue
                print(f"Extracting to {out_dir} ...")
                with zipfile.ZipFile(tmp_zip, 'r') as zf:
                    zf.extractall(out_dir)
                break
            except Exception as e:
                last_err = e
                continue
        else:
            # loop exhausted
            raise last_err or RuntimeError("Unable to download dataset ZIP via HTTP fallback.")

    # Roboflow zips typically contain a single top-level folder
    subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return out_dir


def find_data_yaml(dataset_dir: Path) -> Optional[Path]:
    # Ultralytics expects a data YAML; common names: data.yaml
    for name in ["data.yaml", "dataset.yaml", "roboflow.yaml"]:
        p = dataset_dir / name
        if p.exists():
            return p
    # search recursively
    for p in dataset_dir.rglob("*.yaml"):
        if p.name in {"data.yaml", "dataset.yaml", "roboflow.yaml"}:
            return p
    return None


def ensure_roboflow_dataset(
    api_key: Optional[str],
    workspace: Optional[str],
    project: Optional[str],
    version: Optional[int],
    split: str = "yolov8",
    location: str = "./data",
    format: Optional[str] = None,
) -> Path:
    if not (workspace and project and version):
        raise ValueError("Roboflow config must include workspace, project, and version (see configs/example.yaml)")
    if workspace == "your-workspace" or project == "your-project":
        raise ValueError("Please edit your config: set roboflow.workspace and roboflow.project to real values.")
    load_dotenv()
    key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Roboflow API key missing. Set ROBOFLOW_API_KEY or roboflow.api_key in your config.")
    try:
        path = download_roboflow(key, workspace, project, int(version), split, location, format)
    except zipfile.BadZipFile:
        # Provide actionable hint
        raise RuntimeError(
            "Roboflow download failed (bad zip). Check your workspace/project/version and API key. "
            "Also try deleting any existing files under the data location."
        )
    return path
