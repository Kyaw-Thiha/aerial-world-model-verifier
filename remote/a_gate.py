"""Track A2: re-run GeoCalib and keep the confidence/uncertainty outputs (pre-reg amendment 2).

The Gate 3 rows (``remote/a_geocalib_p3.py``) stored only the gravity direction. Amendment 2
gates on what the estimator says about its own reliability, so the same frames are re-run
with the same model, weights, camera model and commit, and the extras are written next to
the gravity vector. Nothing about the Gate 3 artifacts is touched: every job here writes a
new directory (``geocalib/gate_a2_*``).

Three jobs, each just "decode, (optionally) warp, run GeoCalib, write JSONL":

* ``tii``   — the eight TII flights, all six conditions, exactly the Gate 3 warps.
* ``anwm``  — Track B's generated 224x224 clip frames (pixel side of Task 3).
* ``floor`` — the 186 released ANWM ground-truth frames (level) at 224 px, used for the
  amendment-2 transfer check and for the Task 3 positive controls.

The ``*_raw`` fields are GeoCalib's own numbers in its own units (the optimizer returns
``sqrt`` variances of its delta parameters); the gate only ever uses them as a monotone
score with a threshold fitted in the same units, so no unit conversion is needed and none
is applied.

    pixi run -e gpu python -m modal run remote/a_gate.py --job tii
    pixi run -e gpu python -m modal run remote/a_gate.py --job anwm
    pixi run -e gpu python -m modal run remote/a_gate.py --job floor
"""
from __future__ import annotations

import modal

from src.remote_common import (FLIGHTS, GEOCALIB_COMMIT, VOLUME_MOUNT, ensure_weights,
                               gate_row as _gate_row, make_image)

app = modal.App("awmv-a-gate")
image = make_image()
volume = modal.Volume.from_name("awmv-a-data", create_if_missing=True)

UNC_KEYS = ("gravity_uncertainty", "roll_uncertainty", "pitch_uncertainty")
CONF_KEYS = ("up_confidence", "latitude_confidence")


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: volume}, timeout=4 * 3600)
def tii_conditions(tag: str = "p3a_v1", conditions: list[str] | None = None,
                   flights: list[str] | None = None, out_tag: str = "gate_a2_tii") -> dict:
    """Re-run the Gate 3 warps, keeping uncertainty and confidence per frame."""
    import json
    import shutil
    import time
    import zipfile
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    from src.datasets.tii_video import flight_parts
    from src.negatives import equidistant_f, remap_arrays, warp_with_remap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_weights()
    model = GeoCalib(weights="distorted").to(device)
    out_dir = Path(VOLUME_MOUNT) / "geocalib" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {"tag": out_tag, "conditions": conditions, "flights": {},
                  "geocalib_commit": GEOCALIB_COMMIT, "row_builder": "a_gate._gate_row"}

    for flight in (flights or FLIGHTS):
        spec = np.load(Path(VOLUME_MOUNT) / "edits" / tag / f"{flight}.npz", allow_pickle=False)
        conds = conditions or [k[2:] for k in spec.files if k.startswith("M_")]
        control, _ = flight_parts(flight)
        zp = Path(VOLUME_MOUNT) / "tii_video" / control / f"camera_{flight}.zip"
        local = Path("/tmp") / zp.name
        if not local.exists():
            shutil.copyfile(zp, local)
        with zipfile.ZipFile(local) as z:
            have = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".jpg")}
            names = [str(n) for n in spec["names"]]
            missing = [n for n in names if n not in have]
            if missing:
                raise RuntimeError(f"{flight}: {len(missing)} frames missing, first {missing[0]}")
            size = tuple(int(v) for v in spec["image_size"])
            fpx = equidistant_f(size[0], size[1], 175.0)
            rows = {c: [] for c in conds}
            t0 = time.time()
            ident_maps = None
            for i, name in enumerate(names):
                buf = np.frombuffer(z.read(have[name]), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                for c in conds:
                    if c == "identity":
                        if ident_maps is None:
                            ident_maps = remap_arrays(spec["M_identity"][i].astype(np.float64),
                                                      size[0], size[1], fpx)
                        mx, my = ident_maps
                    else:
                        mx, my = remap_arrays(spec[f"M_{c}"][i].astype(np.float64),
                                              size[0], size[1], fpx)
                    warped, valid = warp_with_remap(img, mx, my)
                    feed = torch.from_numpy(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                    feed = feed.permute(2, 0, 1).float().to(device) / 255.0
                    with torch.no_grad():
                        res = model.calibrate(feed, camera_model="simple_divisional")
                    rows[c].append(_gate_row(res, {
                        "frame": name, "index": i,
                        "valid_frac": float(valid.mean()),
                        "angle_deg": float(spec[f"angle_{c}"][i]),
                    }))
            dt = time.time() - t0
            for c in conds:
                (out_dir / f"{flight}__{c}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in rows[c]))
            info["flights"][flight] = {"n_frames": len(names), "conditions": conds,
                                       "seconds": dt,
                                       "frames_per_s": len(names) * len(conds) / dt if dt else 0}
            volume.commit()
            print(f"{flight}: {len(names)}x{len(conds)} frames in {dt:.0f}s "
                  f"({len(names)*len(conds)/dt:.1f} f/s)", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    volume.commit()
    return info


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: volume}, timeout=2 * 3600)
def anwm_clips(tag: str = "clips_v1", limit: int = 0, size: int = 224,
               fov_deg: float = 90.0, out_tag: str = "gate_a2_anwm") -> dict:
    """Track B's generated 224x224 clip frames, with the same extras."""
    import json
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_weights()
    model = GeoCalib(weights="pinhole").to(device)
    spec_doc = json.loads((Path(VOLUME_MOUNT) / "anwm_specs" / tag / "specs.json").read_text())
    # the file is {"clips": [...], ...}; a bare list is accepted for older tags
    clip_list = spec_doc["clips"] if isinstance(spec_doc, dict) else spec_doc
    root = Path(VOLUME_MOUNT) / "anwm_clips"
    out_dir = Path(VOLUME_MOUNT) / "geocalib" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    f = (size / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    clips = clip_list[:limit] if limit else clip_list
    t0 = time.time()
    n_rows = 0
    with (out_dir / "depicted.jsonl").open("w") as fh:
        for c in clips:
            for name in c["frames"]:
                p = root / c["dir"] / "frames" / name
                img = cv2.imread(str(p))
                if img is None:
                    continue
                if img.shape[0] != size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                feed = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                feed = feed.float().to(device) / 255.0
                with torch.no_grad():
                    res = model.calibrate(feed, camera_model="pinhole",
                                          priors={"focal": torch.tensor(float(f), device=device)})
                fh.write(json.dumps(_gate_row(res, {
                    "frame": name, "clip_id": c["clip_id"], "frame_std": float(img.std()),
                    "f_prior": float(f)})) + "\n")
                n_rows += 1
    info = {"tag": out_tag, "n_clips": len(clips), "n_rows": n_rows, "size": size,
            "f_px": float(f), "seconds": time.time() - t0, "geocalib_commit": GEOCALIB_COMMIT}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    volume.commit()
    return info


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: volume}, timeout=3600)
def anwm_floor(size: int = 224, fov_deg: float = 90.0,
               out_tag: str = "gate_a2_floor") -> dict:
    """The released (level) ANWM frames at 224 px, with extras: transfer check + floor."""
    import json
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_weights()
    model = GeoCalib(weights="pinhole").to(device)
    src = Path(VOLUME_MOUNT) / "anwm_frames"
    out_dir = Path(VOLUME_MOUNT) / "geocalib" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("*/*.jpg"))
    f = (size / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    rows = []
    t0 = time.time()
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        feed = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        feed = feed.float().to(device) / 255.0
        with torch.no_grad():
            res = model.calibrate(feed, camera_model="pinhole",
                                  priors={"focal": torch.tensor(float(f), device=device)})
        rows.append(_gate_row(res, {"frame": p.name, "traj": p.parent.name, "size": size,
                                    "f_prior": float(f), "frame_std": float(img.std())}))
    (out_dir / f"floor_{size}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    info = {"tag": out_tag, "n": len(rows), "size": size, "f_px": float(f),
            "seconds": time.time() - t0, "geocalib_commit": GEOCALIB_COMMIT}
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    volume.commit()
    return info


@app.local_entrypoint()
def main(job: str = "tii", tag: str = "clips_v1", conditions: str = "", flights: str = "",
         limit: int = 0, size: int = 224, parallel: bool = True,
         out_tag: str = ""):
    import json

    if job == "anwm":
        print(json.dumps(anwm_clips.remote(tag, limit, size), indent=2))
        return
    if job == "floor":
        print(json.dumps(anwm_floor.remote(size), indent=2))
        return
    conds = [c for c in conditions.split(",") if c] or None
    fl = [f for f in flights.split(",") if f] or None
    # the TII edit spec is always p3a_v1; `tag` names the ANWM spec set instead
    tii_tag = "p3a_v1"
    ot = out_tag or "gate_a2_tii"
    if not parallel:
        print(json.dumps(tii_conditions.remote(tii_tag, conds, fl, ot), indent=2))
        return
    handles = [tii_conditions.spawn(tii_tag, conds, [f], ot) for f in (fl or FLIGHTS)]
    for f, h in zip((fl or FLIGHTS), handles):
        info = h.get()
        print(f"{f}: {info['flights'][f]['seconds']:.0f}s", flush=True)
