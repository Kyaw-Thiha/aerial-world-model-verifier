"""Turn a generated ANWM clip directory into the shared `Clip` schema.

A clip written by `remote/b_anwm.py::generate` has one PNG per commanded step and a manifest with
the commanded positions and yaw. The `Clip` needs a per-frame attitude; ANWM depicts a **level**
camera by construction (roll = pitch = 0 in every released frame), so the attitude recorded here is
the level attitude with the commanded heading, and `meta["attitude_source"]` says so. That is the
honest truth value for a generated clip: the verifier's job is to compare it with what the pixels
depict, not to take it as ground truth for a real flight.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.datasets.clip import Clip

ANWM_K_512 = np.array([[256.0, 0.0, 256.0], [0.0, 256.0, 256.0], [0.0, 0.0, 1.0]])


def intrinsics_for(size: int = 224, k: np.ndarray = ANWM_K_512, native: int = 512) -> dict:
    """Intrinsics for the resized frame, scaled from the released 512x512 matrix."""
    s = size / native
    return {"fx": float(k[0, 0] * s), "fy": float(k[1, 1] * s),
            "cx": float(k[0, 2] * s), "cy": float(k[1, 2] * s),
            "width": int(size), "height": int(size), "native_size": int(native)}


def load_anwm_clip(clip_dir: str | Path, dt: float = 1.0) -> Clip:
    """Load `<regime>/<id>/` as a Clip. `dt` is the assumed step duration (unverified, see docs)."""
    d = Path(clip_dir)
    m = json.loads((d / "manifest.json").read_text())
    cp = m["commanded_path"]
    pos = np.asarray(cp["positions"], dtype=np.float64)
    yaw = np.asarray(cp["yaw"], dtype=np.float64)
    frames = sorted((d / "frames").glob("*.png"))
    if len(frames) != len(pos) - 1:
        raise ValueError(f"{d.name}: {len(frames)} frames for {len(pos) - 1} commanded steps")
    # frame i depicts the pose after applying action i, i.e. commanded row i+1
    quat = Rotation.from_euler("z", yaw[1:len(frames) + 1]).as_quat()  # xyzw, level camera
    return Clip(name=m["id"], t=np.arange(len(frames), dtype=np.float64) * dt,
                pos=pos[1:len(frames) + 1], quat=quat, quat_order="xyzw", fps=1.0 / dt,
                cam_uptilt_deg=0.0,
                frame_paths=[str(p) for p in frames],
                intrinsics=intrinsics_for(),
                meta={"generated": True, "regime": m["regime"],
                      "attitude_source": "commanded heading, level camera (ANWM depicts a level "
                                         "camera by construction)",
                      "seed": m.get("seed"), "sampling_steps": m.get("sampling_steps"),
                      "designed_required_tilt_deg": m.get("designed_required_tilt_deg"),
                      "dt_assumption_s": float(dt), "dt_verified": False,
                      "slim_checkpoint_sha256": m.get("slim_checkpoint_sha256"),
                      "code_commit": m.get("code_commit"),
                      "start_state": m.get("start_state"),
                      "commanded_path": cp})
