"""Track M, Stage 2: run two independent tilt readers on the Gate B frame plan.

Both readers see the same feed -- square centre crop, 224 x 224, focal prior 112 px (H03's
validated generated-frame feed):

* ``R_pf``   Perspective Fields, Paramnet-360Cities-edina-centered (commit d54be73).
             Adobe Research License, noncommercial research use only; the weights are
             downloaded at run time and never committed or re-hosted.
* ``R_gc224`` GeoCalib (commit 1dedb29), pinhole model, scalar focal prior 112 px.

The verdict reader (``R_frozen``: GeoCalib simple_divisional on the raw fisheye) is already
stored in the schema clips and is not re-run here.

Output: one JSONL per (shard, source) with one row per frame and condition, carrying both
readers' gravity directions and their raw self-assessments.  The lean, the disagreement and
the ensemble are computed in the analysis, from the gravity vectors and each clip's mount, so
the clip-side geometry never lives in this image.

    pixi run modal run remote/m_readers.py --mode probe
    pixi run modal run remote/m_readers.py --mode read --sources sim
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

VOL_M = modal.Volume.from_name("awmv-m-data", create_if_missing=True)
VOL_A = modal.Volume.from_name("awmv-a-data")
VOL = "/vol"
VA = "/va"
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"
PF_COMMIT = "d54be737d6eacfb9d39a2b7079a494924b45bb6c"
F_PX = 112.0

app = modal.App("awmv-m-readers")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml",
                 "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1")
    .pip_install("kornia", "omegaconf", "poselib", "huggingface_hub", "gdown")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .pip_install("albumentations==1.3.1", "pyequilib==0.3.0", "timm", "yacs",
                 "scikit-learn", "opencv-contrib-python-headless==4.10.0.84")
    .run_commands(f"pip install --no-cache-dir git+https://github.com/jinlinyi/"
                  f"PerspectiveFields@{PF_COMMIT}")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("src")
)


def _prep(img):
    """Square centre crop then 224 px -- exactly H03's feed."""
    import cv2
    h, w = img.shape[:2]
    if h != w:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


def _roll(u) -> float:
    import numpy as np
    u = np.asarray(u, dtype=float).ravel()
    return float(np.degrees(np.arctan2(u[0], -u[1])))


def _read_pair(gc, pf, feed, device, f_px: float = F_PX):
    """Both readers on one prepared BGR image.  Returns a dict of raw quantities."""
    import cv2
    import numpy as np
    import torch
    t = torch.from_numpy(cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    t = t.float().to(device) / 255.0
    with torch.no_grad():
        res = gc.calibrate(t, camera_model="pinhole",
                           priors={"focal": torch.tensor(float(f_px), device=device)})
        pred = pf.inference(img_bgr=feed)
    g = res["gravity"].vec3d.detach().cpu().numpy().ravel()
    row = {"gc_gravity": [float(v) for v in g], "gc_roll_deg": _roll(g)}
    for key in ("gravity_uncertainty", "roll_uncertainty", "pitch_uncertainty"):
        v = res.get(key)
        if v is not None:
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            row[f"gc_{key}"] = float(np.asarray(v).ravel()[0])
    for key in ("up_confidence", "latitude_confidence"):
        v = res.get(key)
        if v is not None:
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            row[f"gc_{key}_mean"] = float(np.asarray(v, dtype=float).mean())
    up = pred["pred_gravity_original"].detach().cpu().numpy()
    if up.ndim == 3 and up.shape[0] == 2:
        up = up.transpose(1, 2, 0)
    ux, uy = up[..., 0], up[..., 1]            # field y is up-positive
    ok = np.isfinite(ux) & np.isfinite(uy)
    mean = np.array([float(ux[ok].mean()), float(uy[ok].mean())])
    mean = mean / max(np.linalg.norm(mean), 1e-9)
    pf_roll = float(pred["pred_roll"].item())
    pf_pitch = float(pred["pred_pitch"].item())
    r_, p_ = np.radians(pf_roll), np.radians(pf_pitch)
    # the 3D up vector in camera coordinates from PF's own angles; camera convention
    # x right, y down, z forward.  roll = atan2(u_x, -u_y) by construction.
    row.update({
        "pf_roll_deg": pf_roll,
        "pf_pitch_deg": pf_pitch,
        "pf_vfov_deg": float(pred["pred_general_vfov"].item()),
        "pf_field_mean": [float(v) for v in mean],
        "pf_gravity": [float(np.sin(r_) * np.cos(p_)),
                       float(-np.cos(r_) * np.cos(p_)),
                       float(np.sin(p_))],
    })
    return row


def _video_frames(path, stride):
    import cv2
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            out.append(frame)
        i += 1
    cap.release()
    return out


@app.function(image=image, gpu="A10", volumes={VOL: VOL_M, VA: VOL_A},
              timeout=6 * 3600, env={"HF_HOME": "/vol/hf"})
def read(sources: str = "sim,tii,h04,h01", tag: str = "v1", shard: int = 0,
         n_shards: int = 1, limit: int = 0, feed: str = "crop224") -> dict:
    import shutil
    import time
    import zipfile

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib
    from perspective2d import PerspectiveFields

    from src.datasets.tii_video import flight_parts, load_calibration
    from src.negatives import equidistant_f, remap_arrays, warp_with_remap
    from src.perception import undistort as U

    vc_cache = {}

    def _pinhole(flight, img):
        """Undistort a TII fisheye frame to the project's 100 deg virtual pinhole view."""
        if flight not in vc_cache:
            control, track = flight_parts(flight)
            cal = load_calibration(Path(VOL, "m02/inputs/tii_calib"), control, track)
            vc_cache[flight] = U.virtual_camera(cal["K"], cal["dist"], (640, 480),
                                                fov_deg=100.0)
        vc = vc_cache[flight]
        return U.undistort(img, vc), float(vc.K[0, 0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gc = GeoCalib(weights="pinhole").to(device)
    pf = PerspectiveFields("Paramnet-360Cities-edina-centered").eval().to(device)
    plan = json.loads(Path(VOL, "m02/inputs/plan.json").read_text())
    out_dir = Path(VOL, "m02/scores", tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = [s.strip() for s in sources.split(",") if s.strip()]
    t0 = time.time()
    stats = {}

    def emit(fh, row):
        fh.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------ TII
    if "tii" in want:
        items = [r for i, r in enumerate(plan["tii"]) if i % n_shards == shard]
        spec_cache = {}
        zips: dict[str, zipfile.ZipFile] = {}
        for it in items:
            flight = it["flight"]
            zp = Path(VA, "tii_video", it["zip"])
            if flight not in zips:
                local = Path("/tmp") / zp.name
                if not local.exists():
                    shutil.copyfile(zp, local)
                zips[flight] = zipfile.ZipFile(local)
                have = {Path(n).name: n for n in zips[flight].namelist()
                        if n.lower().endswith(".jpg")}
                spec_cache[flight] = (have, np.load(Path(VA, "edits/p3a_v1",
                                                         f"{flight}.npz")))
            have, spec = spec_cache[flight]
            size = tuple(int(v) for v in spec["image_size"])
            fpx = equidistant_f(size[0], size[1], 175.0)
            lo, hi = it["index"]
            names = [str(n) for n in spec["names"]][lo:hi]
            maps = {}
            with (out_dir / f"tii_shard{shard}.jsonl").open("a") as fh:
                for k, name in enumerate(names):
                    if k % it["stride"]:
                        continue
                    buf = np.frombuffer(zips[flight].read(have[name]), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    for cond in it["conditions"]:
                        if cond == "identity":
                            if "identity" not in maps:
                                maps["identity"] = remap_arrays(
                                    spec["M_identity"][lo + k].astype(np.float64),
                                    size[0], size[1], fpx)
                            mx, my = maps["identity"]
                        else:
                            mx, my = remap_arrays(spec[f"M_{cond}"][lo + k].astype(np.float64),
                                                  size[0], size[1], fpx)
                        warped, _ = warp_with_remap(img, mx, my)
                        if feed == "pinhole100":
                            feed, fpx_use = _pinhole(flight, warped)
                            row = _read_pair(gc, pf, feed, device, fpx_use)
                            row["feed"] = "pinhole100"
                        else:
                            row = _read_pair(gc, pf, _prep(warped), device,
                                             float(it.get("f_px", F_PX)))
                            row["feed"] = "crop224"
                        row.update({"source": "tii", "clip_id": it["clip_id"],
                                    "flight": flight, "cond": cond,
                                    "frame_index": k, "frame_name": name})
                        emit(fh, row)
            stats.setdefault("tii", 0)
            stats["tii"] += 1
            print(f"tii {it['clip_id']} {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ sim
    if "sim" in want:
        items = [r for i, r in enumerate(plan["sim"]) if i % n_shards == shard]
        for it in items if not limit else items[:limit]:
            vid = Path(VOL, "m02/inputs/sim/videos", f"{it['clip_id']}.mp4")
            frames = _video_frames(vid, it["stride"]) if vid.exists() else []
            if not frames:
                continue
            with (out_dir / f"sim_shard{shard}.jsonl").open("a") as fh:
                for k, frame in enumerate(frames):
                    row = _read_pair(gc, pf, _prep(frame), device,
                                     float(it.get("f_px", F_PX)))
                    row.update({"source": "sim", "clip_id": it["clip_id"],
                                "family": it["family"], "uptilt_deg": it["uptilt_deg"],
                                "defect_family": it["defect_family"],
                                "frame_index": k * it["stride"]})
                    emit(fh, row)
            stats["sim"] = stats.get("sim", 0) + 1
            print(f"sim {it['clip_id']} {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ h04
    if "h04" in want:
        items = [r for i, r in enumerate(plan["h04"]) if i % n_shards == shard]
        for it in items:
            vid = Path(VOL, "m02/inputs/h04", f"{it['clip_id']}.mp4")
            frames = _video_frames(vid, it["stride"]) if vid.exists() else []
            if not frames:
                continue
            with (out_dir / f"h04_shard{shard}.jsonl").open("a") as fh:
                for k, frame in enumerate(frames):
                    row = _read_pair(gc, pf, _prep(frame), device,
                                     float(it.get("f_px", F_PX)))
                    row.update({"source": "h04", "clip_id": it["clip_id"],
                                "frame_index": k * it["stride"]})
                    emit(fh, row)
            stats["h04"] = stats.get("h04", 0) + 1
            print(f"h04 {it['clip_id']} {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ h01
    if "h01" in want:
        items = [r for i, r in enumerate(plan["h01"]) if i % n_shards == shard]
        for it in items:
            paths = [Path(VOL, "m02/inputs/h01", it["clip_id"], f"{i:02d}.png")
                     for i in range(len(it["frames"]))]
            if not all(p.exists() for p in paths):
                continue
            with (out_dir / f"h01_shard{shard}.jsonl").open("a") as fh:
                for k, p in enumerate(paths):
                    frame = cv2.imread(str(p))
                    if frame is None:
                        continue
                    row = _read_pair(gc, pf, _prep(frame), device,
                                     float(it.get("f_px", F_PX)))
                    row.update({"source": "h01", "clip_id": it["clip_id"],
                                "f_px": float(it.get("f_px", F_PX)), "frame_index": k})
                    emit(fh, row)
            stats["h01"] = stats.get("h01", 0) + 1
            print(f"h01 {it['clip_id']} {time.time() - t0:.0f}s", flush=True)

    VOL_M.commit()
    info = {"tag": tag, "shard": shard, "n_shards": n_shards, "sources": want,
            "clips": stats, "seconds": time.time() - t0,
            "seconds_per_clip": {k: v for k, v in stats.items()},
            "geocalib_commit": GEOCALIB_COMMIT, "pf_commit": PF_COMMIT,
            "feed": f"centre square crop -> 224 px, f={F_PX}"}
    (out_dir / f"summary_shard{shard}_{int(t0)}.json").write_text(json.dumps(info, indent=1))
    VOL_M.commit()
    return info


@app.function(image=image, gpu="A10", volumes={VOL: VOL_M}, timeout=3600,
              env={"HF_HOME": "/vol/hf"})
def probe(n_frames: int = 6) -> dict:
    """Sign convention: a pure image rotation is a camera roll; both readers must follow it."""
    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib
    from perspective2d import PerspectiveFields

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gc = GeoCalib(weights="pinhole").to(device)
    pf = PerspectiveFields("Paramnet-360Cities-edina-centered").eval().to(device)
    frames = []
    simdir = Path(VOL, "m02/inputs/sim/videos")
    for p in sorted(simdir.glob("*.mp4"))[:2]:
        frames += _video_frames(p, 60)[:n_frames // 2 + 1]
    h04 = sorted(Path(VOL, "m02/inputs/h04").glob("*.mp4"))[:2]
    for p in h04:
        frames += _video_frames(p, 40)[:n_frames // 2 + 1]
    rows = []
    for i, frame in enumerate(frames):
        feed = _prep(frame)
        for deg in (-10.0, 0.0, 10.0):
            M = cv2.getRotationMatrix2D((112.0, 112.0), deg, 1.0)
            rot = cv2.warpAffine(feed, M, (224, 224), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            row = _read_pair(gc, pf, rot, device)
            row.update({"frame": i, "rotation_deg": deg})
            rows.append(row)
    out = Path(VOL, "m02")
    (out / "rotation_probe.json").write_text(json.dumps(rows, indent=1))
    VOL_M.commit()
    return {"n": len(rows)}


@app.local_entrypoint()
def main(mode: str = "read", sources: str = "sim,tii,h04,h01", tag: str = "v1",
         shard: int = 0, n_shards: int = 1, limit: int = 0, feed: str = "crop224"):
    if mode == "probe":
        print(json.dumps(probe.remote(), indent=1))
    else:
        print(json.dumps(read.remote(sources=sources, tag=tag, shard=shard,
                                     n_shards=n_shards, limit=limit, feed=feed),
                         indent=1))
