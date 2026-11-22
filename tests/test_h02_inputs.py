"""H2 unit checks for the SolarWM input construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.generation.attitude import body_z_from_poses
from src.perception.frames import camera_to_body_pitched

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h02_solarwm_inputs"


def test_input_manifest_and_pose_geometry():
    if not (INPUTS / "manifest.json").exists():
        return
    man = json.loads((INPUTS / "manifest.json").read_text())
    assert len(man["samples"]) == 110
    assert len(set(man["samples"])) == 110
    R_bc = camera_to_body_pitched(0.0)
    for sid in ("probe_p20", "probe_p60",
                "grid_302OLP89E75X7MFPRR58QG7CJPZCA7_processed_015_turn20_flatness",
                "grid_302OLP89E75X7MFPRR58QG7CJPZCA7_processed_015_turn20_level"):
        poses = np.load(INPUTS / sid / "poses.npy")
        assert poses.shape == (81, 4, 4)
        assert np.abs(np.linalg.det(poses[:, :3, :3]) - 1).max() < 1e-5
        assert np.linalg.norm(poses[:, :3, 3], axis=1).max() <= 16.0001
        z = body_z_from_poses(poses, R_bc)
        tilt = np.degrees(np.arccos(np.clip(z[:, 2], -1, 1)))
        if sid.endswith("_level"):
            assert tilt.max() < 1e-6
        if sid == "probe_p20":
            assert tilt[-1] > 15.0
