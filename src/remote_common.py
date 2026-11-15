"""Shared Modal image and GeoCalib row builder for Track A2 remote jobs.

Lives in ``src`` rather than ``remote`` because Modal mounts ``src`` into the container
(``add_local_python_source("src")``) but only the entrypoint file from ``remote`` — a
container cannot import a sibling ``remote`` module.

The image is the Gate 3 perception image verbatim (same pip pins, same weights handling as
``remote/a_geocalib_p3.py``) so an estimate here is comparable with an estimate there.
"""
from __future__ import annotations

import numpy as np

GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"
VOLUME_MOUNT = "/vol"
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
UNC_KEYS = ("gravity_uncertainty", "roll_uncertainty", "pitch_uncertainty")
CONF_KEYS = ("up_confidence", "latitude_confidence")


def make_image():
    """The Gate 3 perception image, plus the Track A2 gate module already in ``src``."""
    import modal

    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "libgl1", "libglib2.0-0")
        .pip_install("numpy<2", "scipy", "pandas", "pyyaml",
                     "opencv-python-headless==4.10.0.84",
                     "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
        .pip_install("kornia", "gdown", "huggingface_hub", "omegaconf", "poselib")
        .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
        .add_local_python_source("src")
    )


def ensure_weights() -> None:
    """Put the checkpoints where ``torch.hub`` looks for them (mirrored on the Volume)."""
    import shutil
    from pathlib import Path

    cache = Path("/root/.cache/torch/hub/geocalib")
    cache.mkdir(parents=True, exist_ok=True)
    src = Path(VOLUME_MOUNT) / "geocalib_weights"
    for name in ("pinhole.tar", "distorted.tar"):
        dst = cache / name
        if not dst.exists():
            shutil.copyfile(src / f"geocalib-{name}", dst)


def gate_row(res, extra: dict) -> dict:
    """Gravity direction plus GeoCalib's own self-assessment for one frame."""
    g = res["gravity"].vec3d.detach().cpu().numpy().ravel()
    row = dict(extra)
    row["gravity_up_cam"] = [float(v) for v in g]
    for k in UNC_KEYS:
        if k in res and res[k] is not None:
            v = res[k]
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            row[f"{k}_raw"] = float(np.asarray(v).ravel()[0])
    for k in CONF_KEYS:
        if k in res and res[k] is not None:
            v = res[k]
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            a = np.asarray(v, dtype=np.float64).ravel()
            row[f"{k}_mean"] = float(a.mean())
            row[f"{k}_p10"] = float(np.percentile(a, 10.0))
    return row
