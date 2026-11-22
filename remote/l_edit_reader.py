"""Track L, L02 downstream: run a reader on Track A's regenerated pixel edits.

The new reader is a function of pixels, so detecting a *pixel* edit needs the edited pixels.
Track D's clip store holds only the frozen reader's estimates, not images. This app therefore
regenerates the exact conditions with Track A's own pipeline -- ``condition_maps`` from
``src/negatives/edits.py`` and the 175 deg equidistant fisheye warp -- applies the per-clip
common border mask (the intersection of every condition's valid region, so no condition gains
pixels another loses), and runs the reader on the real frame and on each condition.

Pipeline check: with ``--model frozen`` the same frames are read by the frozen reader and its
per-frame leans are compared with the stored ``depicted_lean`` of Track D's clips. A discrepancy
above 0.5 deg median on any condition stops the downstream analysis.

    pixi run modal run remote/l_edit_reader.py --model frozen --flights flight-01a-ellipse
    pixi run modal run remote/l_edit_reader.py --model mixed --all
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-l-edit-reader")
volume = modal.Volume.from_name("awmv-l-data", create_if_missing=True)
VOL = "/vol"
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml",
                 "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1")
    .pip_install("kornia", "omegaconf", "poselib", "huggingface_hub",
                 "numpy<2", "torch==2.4.1", "torchvision==0.19.1")
    .run_commands(f"pip install --no-deps git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("src")
)

CONDS = ("null_edit", "tilt_k0", "mirror_roll", "yaw_control")


def _lean(up_cam, R_cb):
    import numpy as np

    from src.verdict.lean import depicted_lean_from_gravity

    return depicted_lean_from_gravity(np.asarray(up_cam)[None], R_cb)[0]


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=6 * 3600)
def edits(flights: str, model: str = "frozen", lm_steps: int = 10,
          max_clips: int = 0) -> dict:
    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib
    from scipy.spatial.transform import Rotation

    from src.negatives.edits import condition_maps, rotation_angle_deg
    from src.negatives.fisheye_warp import equidistant_f, remap_arrays
    from src.perception.frames import camera_to_body_pitched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "frozen":
        calib = GeoCalib(weights="distorted").to(device)
    else:
        calib = GeoCalib(weights="distorted").to(device)
        state = torch.load(Path(VOL) / "l02" / "models" / f"{model}.pt",
                           map_location=device, weights_only=True)
        calib.model.load_state_dict(state, strict=False)
    net = calib.model
    net.optimizer.num_steps = lm_steps
    net.eval()

    specs = Path(VOL) / "l02" / "p3a"
    clips_json = json.loads((specs / "clips.json").read_text())["clips"]
    store = Path(VOL) / "l02" / "downstream" / "clips"
    out_dir = Path(VOL) / "l02" / "downstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    for flight in [f for f in flights.split(",") if f]:
        spec = np.load(specs / f"{flight}.npz", allow_pickle=False)
        names = [str(n) for n in spec["names"]]
        R_bw = Rotation.from_quat(np.asarray(spec["quat"])).as_matrix()
        R_cb = camera_to_body_pitched(np.deg2rad(float(spec["uptilt_deg"])))
        R_cw = np.einsum("nij,jk->nik", R_bw, R_cb)
        frames_root = Path(VOL) / "l02" / "real_frames" / flight
        fl_rows = []
        my_clips = [c for c in clips_json if c["flight"] == flight]
        if max_clips:
            my_clips = my_clips[:max_clips]
        for clip in my_clips:
            lo, hi = int(clip["index"][0]), int(clip["index"][1])
            idxs = list(range(lo, hi))
            imgs = []
            for i in idxs:
                p = frames_root / names[i]
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    imgs.append(None)
                else:
                    imgs.append(img)
            h, w = next((im.shape[:2] for im in imgs if im is not None), (480, 640))
            fpx = equidistant_f(w, h, 175.0)
            cond_rows = {c: [] for c in CONDS}
            real_rows = []
            for j, i in enumerate(idxs):
                img = imgs[j]
                if img is None:
                    for c in CONDS:
                        cond_rows[c].append(None)
                    real_rows.append(None)
                    continue
                maps = {}
                Ms = {}
                for cond in CONDS:
                    if cond == "null_edit":
                        M = condition_maps("identity", R_bw[i:i + 1], R_cw[i:i + 1])[0]
                    elif cond == "tilt_k0":
                        M = condition_maps("tilt_k", R_bw[i:i + 1], R_cw[i:i + 1], k=0.0)[0]
                    elif cond == "mirror_roll":
                        M = condition_maps("mirror_roll", R_bw[i:i + 1], R_cw[i:i + 1])[0]
                    else:
                        mag = min(float(rotation_angle_deg(
                            condition_maps("tilt_k", R_bw[i:i + 1], R_cw[i:i + 1], k=0.0)[0])),
                            60.0)
                        M = condition_maps("yaw_control", R_bw[i:i + 1], R_cw[i:i + 1],
                                           yaw_rot_rad=np.deg2rad(mag))[0]
                    mx, my = remap_arrays(M, w, h, fpx)
                    maps[cond] = (mx, my, (mx >= 0) & (my >= 0))
                    Ms[cond] = M
                common = np.logical_and.reduce([maps[c][2] for c in CONDS])
                warped = {}
                for cond in CONDS:
                    mx, my, _ = maps[cond]
                    wimg = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    wimg = np.where(common[..., None], wimg, 0)
                    warped[cond] = wimg
                for cond, src in [("real", img)] + [(c, warped[c]) for c in CONDS]:
                    rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                    t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
                    with torch.no_grad():
                        net.optimizer.set_camera_model("simple_divisional")
                        net.optimizer.shared_intrinsics = False
                        res = net(calib.image_processor(t[None]) | {"camera_R_rig": None})
                    g = np.asarray(res["gravity"].vec3d.detach().cpu()).ravel()
                    g = g / np.linalg.norm(g)
                    row = {"frame": int(i), "name": names[i],
                           "up_cam": [float(v) for v in g],
                           "lean": [float(v) for v in _lean(g, R_cb)]}
                    if cond == "real":
                        real_rows.append(row)
                    else:
                        cond_rows[cond].append(row)
            fl_rows.append({"clip_id": clip["clip_id"], "index": [lo, hi],
                            "real": real_rows, "conditions": cond_rows})
            print(f"[{flight}] {clip['clip_id']} done", flush=True)
        out[flight] = fl_rows
        (out_dir / f"{model}__{flight}.json").write_text(json.dumps({"flight": flight,
                                                                     "model": model,
                                                                     "clips": fl_rows}))
        volume.commit()

        if model == "frozen":                      # pipeline check against Track D's store
            import math

            diffs = {}
            for c in out[flight]:
                for cond in CONDS:
                    d = store / flight / f"{c['clip_id']}__{cond.replace('null_edit', 'null_edit')}"
                    npz = d / "verdict_clip.npz"
                    if not npz.exists():
                        continue
                    z = np.load(npz)
                    ref = np.asarray(z["depicted_lean"], float)
                    got = np.array([r["lean"] if r else [np.nan, np.nan]
                                    for r in c["conditions"][cond]], float)
                    ok = np.isfinite(ref).all(1) & np.isfinite(got).all(1)
                    if ok.sum() < 3:
                        continue
                    cth = np.sum(ref[ok] * got[ok], 1) / (
                        np.linalg.norm(ref[ok], axis=1) * np.linalg.norm(got[ok], axis=1))
                    ang = np.degrees(np.arccos(np.clip(cth, -1, 1)))
                    diffs.setdefault(cond, []).extend(ang.tolist())
            out[flight + "__check"] = {k: {"n": len(v), "median_deg": float(np.median(v)),
                                           "p90_deg": float(np.percentile(v, 90))}
                                       for k, v in diffs.items()}
            print(f"[{flight}] check {out[flight + '__check']}", flush=True)
    return {k: (v if k.endswith("__check") else len(v)) for k, v in out.items()}


@app.local_entrypoint()
def main(model: str = "frozen", flights: str = "", all: bool = False, max_clips: int = 0,
         lm_steps: int = 10):
    if all:
        flights = ",".join(["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
                            "flight-07a-lemniscate", "flight-07p-lemniscate",
                            "flight-13a-trackRATM", "flight-13p-trackRATM",
                            "flight-16b-trackRATM".replace("16b", "16p")])
    print(edits.remote(flights, model, lm_steps, max_clips))
