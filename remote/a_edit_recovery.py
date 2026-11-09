"""Track A2, Task 3 positive control: can the estimator see tilt in generated frames?

"Reads about 0 deg" means nothing if the estimator is blind at 224 px. So before any ANWM
number is interpreted, known pure camera rotations are applied to generated frames and the
recovered tilt change is compared with the applied one.

Conditions: identity, roll and pitch of 5, 10 and 20 deg, applied with the **pinhole**
remap (``src.negatives.pinhole_warp``) because ANWM renders through a pinhole camera at a
90 deg field. The TII edits use the equidistant fisheye model; mixing the two would put a
lens change inside the rotation being measured.

    pixi run -e gpu python -m modal run remote/a_edit_recovery.py
"""
from __future__ import annotations

import modal

from src.remote_common import (GEOCALIB_COMMIT, VOLUME_MOUNT, ensure_weights,
                               gate_row as _gate_row, make_image)

app = modal.App("awmv-a-edit-recovery")
image = make_image()
volume = modal.Volume.from_name("awmv-a-data", create_if_missing=True)

SIZE = 224
F_PX = SIZE / 2.0          # 90 deg horizontal field, the Gate 3 224 px setting
ANGLES = (5.0, 10.0, 20.0)
AXES = ("roll", "pitch")


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: volume}, timeout=3 * 3600)
def edit_recovery(clip_ids: list[str], tag: str = "clips_v1",
                  out_tag: str = "gate_a2_edits") -> dict:
    import json
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib
    from scipy.spatial.transform import Rotation

    from src.negatives.pinhole_warp import applied_tilt_change_deg, pinhole_remap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_weights()
    model = GeoCalib(weights="pinhole").to(device)
    spec_doc = json.loads((Path(VOLUME_MOUNT) / "anwm_specs" / tag / "specs.json").read_text())
    clip_list = spec_doc["clips"] if isinstance(spec_doc, dict) else spec_doc
    by_id = {c["clip_id"]: c for c in clip_list}
    root = Path(VOLUME_MOUNT) / "anwm_clips"
    out_dir = Path(VOLUME_MOUNT) / "geocalib" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    conds = [("identity", None, 0.0)]
    for axis in AXES:
        for ang in ANGLES:
            conds.append((f"{axis}_{int(ang)}", axis, ang))

    maps: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for name, axis, ang in conds:
        if axis is None:
            M = np.eye(3)
        else:
            M = Rotation.from_euler("x" if axis == "roll" else "y", ang, degrees=True).as_matrix()
        mx, my = pinhole_remap(M, SIZE, SIZE, F_PX)
        maps[(axis or "none", ang)] = (mx, my)

    f_t = torch.tensor(float(F_PX), device=device)
    n_rows = 0
    t0 = time.time()
    with (out_dir / "edits.jsonl").open("w") as fh:
        for cid in clip_ids:
            c = by_id[cid]
            for fname in c["frames"]:
                p = root / c["dir"] / "frames" / fname
                img = cv2.imread(str(p))
                if img is None:
                    continue
                if img.shape[0] != SIZE:
                    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                for name, axis, ang in conds:
                    mx, my = maps[(axis or "none", ang)]
                    warped = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    valid = float(((mx >= 0) & (my >= 0)).mean())
                    feed = torch.from_numpy(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                    feed = feed.permute(2, 0, 1).float().to(device) / 255.0
                    with torch.no_grad():
                        res = model.calibrate(feed, camera_model="pinhole",
                                              priors={"focal": f_t})
                    row = _gate_row(res, {
                        "clip_id": cid, "frame": fname, "condition": name,
                        "axis": axis or "none", "applied_deg": float(ang),
                        "applied_tilt_change_deg": float(applied_tilt_change_deg(
                            np.eye(3) if axis is None else
                            Rotation.from_euler("x" if axis == "roll" else "y", ang,
                                                degrees=True).as_matrix())),
                        "valid_frac": valid, "f_prior": float(F_PX)})
                    fh.write(json.dumps(row) + "\n")
                    n_rows += 1
    info = {"tag": out_tag, "n_clips": len(clip_ids), "n_rows": n_rows,
            "conditions": [c[0] for c in conds], "size": SIZE, "f_px": F_PX,
            "seconds": time.time() - t0, "geocalib_commit": GEOCALIB_COMMIT}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    volume.commit()
    return info


@app.local_entrypoint()
def main(clip_ids: str = ""):
    import json
    ids = [c for c in clip_ids.split(",") if c]
    if not ids:
        # 7 cruise clips + their seed repeats + a few maneuvers: 12 distinct paths
        ids = ["C_cruise_ctl_course_0", "C_cruise_ctl_course_1", "C_cruise_ctl_course_2",
               "C_cruise_ctl_course_0_s1", "C_cruise_ctl_course_0_s2",
               "C_cruise_ctl_course_1_s1", "C_cruise_ctl_course_1_s2",
               "C_turn_10_fixed_0", "C_slalom_10_fixed_0", "C_accel_brake_10_course_0",
               "C_climb_turn_5_fixed_0", "C_turn_20_fixed_0"]
    print(json.dumps(edit_recovery.remote(ids), indent=2))
