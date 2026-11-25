"""Track V, Part 0: Pi3X multi-view reconstruction + ground-plane extraction (Modal, GPU).

Frozen protocol: ``configs/v00_feasibility_prereg.yaml`` + amendments 1 and 2.

Each unit is reconstructed in overlapping 8 s chunks (60 frames at 7.5 Hz, stride 4 s, the last
chunk shifted to the clip end). Per chunk the function writes one ``<unit>__c<k>.npz`` with the
camera-to-world poses, the per-frame decimated point cloud (2,000 points per frame, deterministic
even stride), the confidence field summary and a chunk-level plane fit. Per analysis window it
fits the ground plane from that chunk's points whose frame times lie in ``[t1-0.5, t2+0.5]``
(amendment 1, A2) and records the winning plane. The window table goes to ``<unit>.json``.

    modal run remote/v_recon.py --probe i02_dev_084
    modal run remote/v_recon.py --units i02_dev_084,outdoor_forward_1
    modal run remote/v_recon.py --all
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-v-recon")

v_vol = modal.Volume.from_name("awmv-v-data", create_if_missing=True)
m_vol = modal.Volume.from_name("awmv-f2-models", create_if_missing=True)
V_DATA = "/v-data"
MODELS = "/models"

CHUNK_S = 8.0
CHUNK_STRIDE_S = 4.0
WINDOW_S = 3.0
WINDOW_STRIDE_S = 0.5
PLANE_PAD_S = 0.5
POINTS_PER_FRAME = 2000

pi3_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ffmpeg")
    .pip_install("torch==2.5.1", "torchvision==0.20.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==1.26.4", "wheel", "pip>=24.2", "setuptools>=77", "ninja",
                 "packaging", "huggingface_hub", "safetensors", "pillow",
                 "opencv-python-headless", "plyfile", "einops", "timm")
    .run_commands(
        "git clone --depth 1 https://github.com/yyfz/Pi3 /opt/Pi3",
        "cd /opt/Pi3 && pip install --no-build-isolation -e .",
    )
    .env({"HF_HOME": f"{MODELS}/hf", "PYTHONUNBUFFERED": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("src")
)


def chunk_index(t: list[float]) -> list[list[int]]:
    """(start, stop) frame-index pairs: 8 s chunks, 4 s stride, last shifted to the end."""
    import numpy as np

    t = np.asarray(t, dtype=float)
    n = len(t)
    dur = float(t[-1] - t[0])
    if dur <= CHUNK_S + 1e-9:
        return [[0, n]]
    starts, s = [], 0.0
    while s + CHUNK_S < dur - 1e-9:
        starts.append(s)
        s += CHUNK_STRIDE_S
    starts.append(max(0.0, dur - CHUNK_S))
    out, seen = [], set()
    for s in starts:
        i = int(np.searchsorted(t, t[0] + s - 1e-9))
        while i > 0 and t[i - 1] >= t[0] + s - 1e-9:
            i -= 1
        j = int(np.searchsorted(t, t[0] + s + CHUNK_S + 1e-9, side="right"))
        j = min(j, i + 64, n)
        if j - i >= 8 and (i, j) not in seen:
            seen.add((i, j))
            out.append([i, j])
    return out


def window_list(t: list[float]) -> list[dict]:
    import numpy as np

    t = np.asarray(t, dtype=float)
    out, start = [], float(t[0])
    while start + WINDOW_S <= float(t[-1]) + 1e-9:
        inside = np.flatnonzero((t >= start - 1e-9) & (t <= start + WINDOW_S + 1e-9))
        if len(inside) >= 6:
            out.append({"t1": start, "t2": start + WINDOW_S, "idx": inside.tolist()})
        start += WINDOW_STRIDE_S
    return out


def _assign_windows(windows: list[dict], chunk_t: list[list[float]]) -> list[int]:
    centres = [0.5 * (c[0] + c[-1]) for c in chunk_t]
    return [int(min(range(len(chunk_t)),
                    key=lambda k: abs(centres[k] - 0.5 * (w["t1"] + w["t2"]))))
            for w in windows]


@app.function(image=pi3_image, gpu="L40S", volumes={V_DATA: v_vol, MODELS: m_vol},
              timeout=3600, cpu=8, memory=32768)
def recon(specs: list[dict], force: bool = False, probe: bool = False) -> str:
    import time

    import numpy as np
    import torch
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_multimodal_data

    from src.vertical.plane import fit_ground_plane, subsample

    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to("cuda")
    out_dir = Path(V_DATA) / "recon"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for spec in specs:
        unit = spec["unit"]
        if (out_dir / f"{unit}.json").exists() and not force and not probe:
            recs.append({"unit": unit, "skipped": True})
            continue
        frames = sorted((Path(V_DATA) / "v_frames" / spec["frames"]).glob("*.png"))
        t = np.asarray(spec["t"], dtype=float)
        assert len(frames) == len(t), (unit, len(frames), len(t))
        K = spec["K"]
        K_np = np.array([[K["fx"], 0.0, K["cx"]], [0.0, K["fy"], K["cy"]],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
        spans = chunk_index(spec["t"])
        chunk_meta, chunk_files = [], []
        t0 = time.time()
        for ci, (i0, i1) in enumerate(spans):
            stage = Path("/tmp") / f"chunk_{unit}_{ci}"
            if stage.exists():
                import shutil
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            for k, p in enumerate(frames[i0:i1]):
                (stage / f"{k:04d}.png").symlink_to(p)
            n = i1 - i0
            cond = {"intrinsics": np.tile(K_np, (n, 1, 1)), "poses": None, "depths": None}
            imgs, cond2 = load_multimodal_data(str(stage), cond, interval=1, device="cuda")
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    res = model(imgs=imgs, **cond2)
            if probe:
                recs.append({"unit": unit, "probe": True, "chunk": ci,
                             "keys": sorted(str(k) for k in res.keys()),
                             "shapes": {str(k): list(np.asarray(
                                 (res[k].detach().float().cpu() if hasattr(res[k], "detach")
                                  else res[k])).shape) for k in res.keys()},
                             "imgs_shape": list(imgs.shape),
                             "wall_s": round(time.time() - t0, 1)})
                continue
            poses = res.get("camera_poses", res.get("poses"))
            if poses is None:
                raise RuntimeError(f"no poses in Pi3X output for {unit}: {list(res.keys())}")
            poses = np.asarray(poses[0].detach().float().cpu(), dtype=np.float64)
            pts = res["points"][0].detach().float().cpu().numpy()
            conf = None
            for key in ("conf", "confidence", "confidences"):
                if key in res:
                    conf = res[key][0].detach().float().cpu().numpy()
                    break
            cams = poses[:, :3, 3]
            down = -poses[:, :3, 1]
            per_frame, labels = [], []
            for s in range(pts.shape[0]):
                p = pts[s].reshape(-1, 3).astype(np.float64)
                ok = np.isfinite(p).all(axis=1)
                if conf is not None:
                    ok &= conf[s].reshape(-1) >= 0.5
                p = p[ok]
                if len(p) > POINTS_PER_FRAME:
                    p = p[:: int(np.ceil(len(p) / POINTS_PER_FRAME))][:POINTS_PER_FRAME]
                per_frame.append(p)
                labels.append(np.full(len(p), s, dtype=np.int32))
            pts_all = np.concatenate(per_frame, axis=0) if per_frame else np.zeros((0, 3))
            lab_all = np.concatenate(labels, axis=0) if labels else np.zeros(0, np.int32)
            clip_plane = fit_ground_plane(subsample(pts_all, 200_000), cams, down)
            name = out_dir / f"{unit}__c{ci}.npz"
            np.savez_compressed(
                name, unit=unit, chunk=ci, i0=i0, i1=i1, t=t[i0:i1], c2w=poses,
                K=np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]]),
                pts=pts_all, pts_frame=lab_all,
                clip_plane=json.dumps({k: (v.as_dict() if k == "plane" else v)
                                       for k, v in clip_plane.items()}),
                conf_available=conf is not None)
            chunk_files.append(name.name)
            chunk_meta.append({"chunk": ci, "i0": i0, "i1": i1,
                               "t_range": [float(t[i0]), float(t[i1 - 1])],
                               "n_points": int(len(pts_all)),
                               "clip_plane": {k: (v.as_dict() if k == "plane" else v)
                                              for k, v in clip_plane.items()}})
        if probe:
            continue
        wins = window_list(spec["t"])
        which = _assign_windows(wins, [c["t_range"] for c in chunk_meta])
        win_rows = []
        for w, k in zip(wins, which):
            d = np.load(out_dir / chunk_files[k])
            ct = d["t"]
            m = (ct >= w["t1"] - PLANE_PAD_S - 1e-9) & (ct <= w["t2"] + PLANE_PAD_S + 1e-9)
            local = np.flatnonzero(m)
            c2w = d["c2w"]
            cams = c2w[local, :3, 3]
            down = -c2w[local, :3, 1]
            sel = np.isin(d["pts_frame"], local)
            pts = d["pts"][sel]
            fit = fit_ground_plane(pts, cams, down)
            win_rows.append({"t1": w["t1"], "t2": w["t2"], "chunk": int(k),
                             "i0": int(chunk_meta[k]["i0"]),
                             "plane": fit["plane"].as_dict() if fit.get("plane") else None,
                             "fit": {kk: vv for kk, vv in fit.items() if kk != "plane"}})
        (out_dir / f"{unit}.json").write_text(json.dumps(
            {"unit": unit, "kind": spec["kind"], "n_frames": len(t),
             "wall_s": round(time.time() - t0, 1), "chunks": chunk_meta,
             "windows": win_rows}, indent=1))
        v_vol.commit()
        recs.append({"unit": unit, "n_chunks": len(chunk_meta), "n_windows": len(win_rows),
                     "wall_s": round(time.time() - t0, 1),
                     "coverage": float(np.mean([r["plane"] is not None
                                                for r in win_rows]))})
    return json.dumps(recs, indent=1)


@app.local_entrypoint()
def main(units: str = "", all_units: bool = False, force: bool = False,
         probe: bool = False, limit: int = 0):
    table = json.loads((Path("data/v/units.json")).read_text())["units"]
    if units:
        want = {u for u in units.split(",") if u}
        table = [u for u in table if u["unit"] in want]
    if all_units:
        table = [u for u in table if u["kind"] in ("sim", "uzh")]
    if limit:
        table = table[:limit]
    specs = [{"unit": u["unit"], "kind": u["kind"], "frames": u["frames"], "t": u["t"],
              "K": u["K"]} for u in table]
    print(f"{len(specs)} units")
    print(recon.remote(specs, force, probe))
