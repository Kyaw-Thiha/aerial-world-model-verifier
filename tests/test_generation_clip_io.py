import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.generation.clip_io import intrinsics_for, load_anwm_clip


def make_clip(root: Path, n_frames: int = 3) -> Path:
    d = root / "C" / "c0"
    (d / "frames").mkdir(parents=True)
    for i in range(n_frames):
        Image.new("RGB", (224, 224), (i * 20, 0, 0)).save(d / "frames" / f"{i:02d}.png")
    pos = np.cumsum(np.tile([5.0, 0.0, 0.0], (n_frames + 1, 1)), axis=0)
    yaw = np.linspace(0.0, 0.3, n_frames + 1)
    m = {"id": "c0", "regime": "C", "frames": n_frames, "seed": 0, "sampling_steps": 250,
         "designed_required_tilt_deg": 10.0,
         "commanded_path": {"positions": pos.tolist(), "yaw": yaw.tolist(),
                            "step_spacing_m": 5.0},
         "start_state": {"traj": "t", "time": 39},
         "slim_checkpoint_sha256": "x", "code_commit": "y"}
    (d / "manifest.json").write_text(json.dumps(m))
    return d


def test_clip_shape_and_attitude(tmp_path):
    d = make_clip(tmp_path, 3)
    c = load_anwm_clip(d, dt=0.5)
    assert c.fps == pytest.approx(2.0)
    assert len(c.t) == len(c.pos) == len(c.quat) == len(c.frame_paths) == 3
    assert c.t[1] - c.t[0] == pytest.approx(0.5)
    # level camera: each quat is a pure yaw about z
    from scipy.spatial.transform import Rotation
    eul = Rotation.from_quat(c.quat).as_euler("ZYX")
    np.testing.assert_allclose(eul[:, 1:], 0.0, atol=1e-12)
    assert c.meta["dt_verified"] is False
    assert "level camera" in c.meta["attitude_source"]


def test_intrinsics_scale_with_frame_size():
    a = intrinsics_for(224)
    assert a["cx"] == a["cy"] == pytest.approx(112.0)
    assert a["fx"] == pytest.approx(112.0)
    assert a["width"] == 224


def test_frame_count_must_match(tmp_path):
    d = make_clip(tmp_path, 3)
    (d / "frames" / "02.png").unlink()
    with pytest.raises(ValueError, match="frames"):
        load_anwm_clip(d)
