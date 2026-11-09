"""Phase 3a GPU work on Modal: GeoCalib on edited TII frames, and the ANWM noise floor.

Two jobs, both just "decode, (optionally) remap, run GeoCalib, write JSONL":

* ``anwm_floor`` — Task 1: the released ANWM frames are exactly level (roll = pitch = 0),
  so the estimated tilt on them is the false-alarm floor. Run at the native 512x512 with
  the declared 90 deg field (f = 256) and at the 224x224 size the model samples.
* ``run_conditions`` — Task 2: for each flight, for every edit condition in the spec
  produced by ``experiments/phase3a_make_specs.py``, warp each frame with its camera-frame
  rotation and estimate gravity. The real clip condition is the identity rotation, so it
  is also the null-edit control's geometric reference.

The spec (rotation matrices, frame names) is built locally and uploaded once, so the
physics and the clip definition never live in this image.

    pixi run -e gpu python -m modal run remote/a_geocalib_p3.py --job floor
    pixi run -e gpu python -m modal run remote/a_geocalib_p3.py --job conditions --conditions tilt_k0
"""
from __future__ import annotations

import modal

app = modal.App("awmv-a-geocalib-p3")

volume = modal.Volume.from_name("awmv-a-data", create_if_missing=True)
VOLUME_MOUNT = "/vol"
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("kornia", "gdown", "huggingface_hub", "omegaconf", "poselib")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .add_local_python_source("src")
)

def _ensure_weights() -> None:
    """Put the two checkpoints where ``torch.hub`` looks for them.

    geocalib fetches them from a GitHub release at model construction, which returned
    HTTP 504 twice in a row. They are downloaded once locally and mirrored on the
    Volume, so a run never depends on that host.
    """
    import shutil
    from pathlib import Path

    cache = Path("/root/.cache/torch/hub/geocalib")
    cache.mkdir(parents=True, exist_ok=True)
    src = Path(VOLUME_MOUNT) / "geocalib_weights"
    for name in ("pinhole.tar", "distorted.tar"):
        dst = cache / name
        if not dst.exists():
            shutil.copyfile(src / f"geocalib-{name}", dst)


FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]


def _geo_row(res, extra: dict) -> dict:
    """Flatten one GeoCalib result into a JSON row (gravity + whatever confidence exists)."""
    import numpy as np

    g = res["gravity"].vec3d.detach().cpu().numpy().ravel()
    row = dict(extra)
    row["gravity_up_cam"] = [float(v) for v in g]
    for name in ("covariance", "log_probs", "sigmas"):
        if hasattr(res["gravity"], name):
            val = getattr(res["gravity"], name)
            if val is not None:
                arr = np.asarray(val.detach().cpu().numpy() if hasattr(val, "detach") else val)
                row[name] = arr.ravel().tolist()[:9]
    return row


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: volume}, timeout=3600)
def anwm_floor(sizes: list[int] = [512, 224], tag: str = "anwm_floor_v1",
               fov_deg: float = 90.0, limit: int = 0) -> dict:
    """GeoCalib pinhole on the released (level) ANWM frames at each requested size."""
    import json
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_weights()
    model = GeoCalib(weights="pinhole").to(device)
    src = Path(VOLUME_MOUNT) / "anwm_frames"
    out_dir = Path(VOLUME_MOUNT) / "geocalib" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("*/*.jpg"))
    if limit:
        step = max(1, len(paths) // limit)
        paths = paths[::step]
    info = {"tag": tag, "n_frames": len(paths), "fov_deg": fov_deg,
            "geocalib_commit": GEOCALIB_COMMIT, "sizes": {}, "device": device}
    for size in sizes:
        rows = []
        t0 = time.time()
        f = (size / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
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
            rows.append(_geo_row(res, {"frame": p.name, "traj": p.parent.name, "size": size,
                                       "f_prior": float(f)}))
        dt = time.time() - t0
        (out_dir / f"floor_{size}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        info["sizes"][str(size)] = {"n": len(rows), "seconds": dt,
                                    "fps": len(rows) / dt if dt else 0.0, "f_px": float(f)}
        volume.commit()
        print(f"{size}px: {len(rows)} frames in {dt:.1f}s ({len(rows)/dt:.1f} fps)", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    volume.commit()
    return info


@app.function(image=image, gpu="A10G", volumes={VOLUME_MOUNT: volume}, timeout=4 * 3600)
def run_conditions(tag: str = "p3a_v1", conditions: list[str] | None = None,
                   flights: list[str] | None = None, out_tag: str | None = None) -> dict:
    """Warp every frame under each condition and estimate gravity on the result."""
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
    _ensure_weights()
    model = GeoCalib(weights="distorted").to(device)
    out_tag = out_tag or tag
    out_dir = Path(VOLUME_MOUNT) / "geocalib" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {"tag": out_tag, "conditions": conditions, "flights": {},
                  "geocalib_commit": GEOCALIB_COMMIT}
    metrics = json.loads((Path(VOLUME_MOUNT) / "edits" / tag / "clips.json").read_text())

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
                    M = spec[f"M_{c}"][i].astype(np.float64)
                    if c == "identity":
                        if ident_maps is None:
                            ident_maps = remap_arrays(M, size[0], size[1], fpx)
                        mx, my = ident_maps
                    else:
                        mx, my = remap_arrays(M, size[0], size[1], fpx)
                    warped, valid = warp_with_remap(img, mx, my)
                    feed = torch.from_numpy(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                    feed = feed.permute(2, 0, 1).float().to(device) / 255.0
                    with torch.no_grad():
                        res = model.calibrate(feed, camera_model="simple_divisional")
                    rows[c].append(_geo_row(res, {
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
                                       "frames_per_s": len(names) * len(conds) / dt if dt else 0,
                                       "valid_frac_median": {c: float(np.median(
                                           [r["valid_frac"] for r in rows[c]])) for c in conds}}
            volume.commit()
            print(f"{flight}: {len(names)} frames x {len(conds)} conditions in {dt:.0f}s "
                  f"({len(names)*len(conds)/dt:.1f} frames/s)", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2))
    volume.commit()
    return info


@app.local_entrypoint()
def main(job: str = "floor", tag: str = "p3a_v1", conditions: str = "",
         flights: str = "", out_tag: str = "", sizes: str = "512,224", limit: int = 0):
    import json

    if job == "floor":
        print(json.dumps(anwm_floor.remote([int(s) for s in sizes.split(",")], tag, 90.0, limit),
                         indent=2))
        return
    print(json.dumps(run_conditions.remote(
        tag,
        [c for c in conditions.split(",") if c] or None,
        [f for f in flights.split(",") if f] or None,
        out_tag or None), indent=2))
