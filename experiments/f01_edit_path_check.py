"""Does a pixel-space rotation edit leave DPVO's translation unchanged?

One sequence only (``indoor_forward_5``), because these are the only edited-frame DPVO runs
Track C ever stored: the C09 pilot's ``indoor_forward_5_tilt_removed`` and
``indoor_forward_5_yaw_control`` paths, 150 frames each, alongside the real path. The F01
design reuses the real path for every edited condition; this quantifies how far that is from
the truth for the one case where the edited path exists.

    python experiments/f01_edit_path_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pathrec.align import umeyama

BASE = Path("data/pathrec/poses/dpvo")


def main() -> int:
    real = np.load(BASE / "indoor_forward_5.npz")["c2w"][:, :3, 3]
    out = {"sequence": "indoor_forward_5", "n_frames": int(len(real)),
           "source": "Track C DPVO poses on the edited frames (C09 pilot)",
           "real_path_len_units": float(np.linalg.norm(np.diff(real, axis=0), axis=1).sum()),
           "pairs": {}}
    for name in ("indoor_forward_5_tilt_removed", "indoor_forward_5_yaw_control"):
        other = np.load(BASE / f"{name}.npz")["c2w"][:, :3, 3]
        n = min(len(real), len(other))
        a, b = real[:n], other[:n]
        R, _, t = umeyama(b, a, with_scale=False)
        ba = b @ R.T + t
        ate = float(np.sqrt(np.mean(np.sum((ba - a) ** 2, axis=1))))
        sa, sb = np.diff(a, axis=0), np.diff(ba, axis=0)
        cos = np.clip((sa * sb).sum(1) /
                      (np.linalg.norm(sa, axis=1) * np.linalg.norm(sb, axis=1) + 1e-12),
                      -1, 1)
        out["pairs"][name] = {
            "rigid_ate_units": ate,
            "ate_fraction_of_path": ate / out["real_path_len_units"],
            "step_direction_median_deg": float(np.degrees(np.median(np.arccos(cos)))),
        }
    out["reading"] = ("a rotation about the optical centre leaves the camera's translation "
                      "in its own frame unchanged, so reusing the real path for rotation "
                      "edits is accurate to the estimator's re-render sensitivity; measured "
                      "here on one sequence only")
    path = Path("results/f01_stage0/edit_path_check.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
