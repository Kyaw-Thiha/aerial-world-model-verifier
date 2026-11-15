"""Track E: GeoCalib on UZH-FPV frames, three pre-registered variants, with self-assessment.

Variants (fixed in ``configs/e_prereg.yaml`` before any run):
  a  pinhole frames + Kalibr focal prior,  camera_model="pinhole",      weights "pinhole"
  b  raw KB4 frames,                       camera_model="simple_divisional", weights "distorted"
  c  pinhole frames, focal free,           camera_model="pinhole",      weights "pinhole"

Per frame the row carries the gravity direction in **camera** coordinates plus GeoCalib's own
confidence maps, aggregated the same way Track A2's gate row does it
(``latitude_confidence_mean``, ``up_confidence_mean``, ``gravity_uncertainty_raw``), so the
frozen S3 gate can be applied to these rows unchanged. The row builder is re-implemented here
rather than imported: ``src/remote_common.py`` is Track A2's file and Track E does not edit or
depend on it.

One container per sequence (``--parallel``), so a failure costs one sequence.

    pixi run modal run remote/e_geocalib.py --parallel --seqs indoor_forward_5,indoor_45_9
"""
from __future__ import annotations

import modal

app = modal.App("awmv-e-geocalib")

e_data = modal.Volume.from_name("awmv-e-data", create_if_missing=True)
a_data = modal.Volume.from_name("awmv-a-data", create_if_missing=True)

E_DATA = "/e-data"
A_DATA = "/a-data"
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("kornia", "gdown", "huggingface_hub", "omegaconf", "poselib")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
)

VARIANTS = ("a", "b", "c")


def _ensure_weights() -> None:
    """Use the checkpoints Track A2 already mirrored on its Volume; fetch only if absent."""
    import shutil
    from pathlib import Path

    cache = Path("/root/.cache/torch/hub/geocalib")
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("pinhole.tar", "distorted.tar"):
        dst = cache / name
        src = Path(A_DATA) / "geocalib_weights" / f"geocalib-{name}"
        if not dst.exists() and src.exists():
            shutil.copyfile(src, dst)


def _row(res, extra: dict) -> dict:
    import numpy as np

    g = res["gravity"].vec3d.detach().cpu().numpy().ravel()
    row = dict(extra)
    row["gravity_up_cam"] = [float(v) for v in g]
    for k in ("gravity_uncertainty", "roll_uncertainty", "pitch_uncertainty"):
        if k in res and res[k] is not None:
            v = res[k]
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            row[f"{k}_raw"] = float(np.asarray(v).ravel()[0])
    for k in ("up_confidence", "latitude_confidence"):
        if k in res and res[k] is not None:
            v = res[k]
            v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            a = np.asarray(v, dtype=np.float64).ravel()
            row[f"{k}_mean"] = float(a.mean())
            row[f"{k}_p10"] = float(np.percentile(a, 10.0))
    cam = res["camera"]
    for k in ("f", "k1"):
        if hasattr(cam, k):
            row[k] = np.asarray(getattr(cam, k).detach().cpu()).ravel().tolist()
    return row


@app.function(image=image, gpu="A10G", volumes={E_DATA: e_data, A_DATA: a_data},
              timeout=3 * 3600)
def sequence(seq: str, variants: str = "abc", max_frames: int = 450) -> str:
    import json
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_weights()
    clip = Path(E_DATA) / "uzh" / seq
    meta = json.loads((clip / "meta.json").read_text())
    fx = float(meta["intrinsics_pinhole"]["fx"])
    raw_files = sorted((clip / "frames_raw").glob("*.png"))[:max_frames]
    pin_files = sorted((clip / "frames_pinhole").glob("*.png"))[:max_frames]
    out_dir = Path(E_DATA) / "geocalib"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    want = [v for v in variants]
    if any(v == "b" for v in want):
        models["b"] = GeoCalib(weights="distorted").to(device)
    if any(v in ("a", "c") for v in want):
        models["pin"] = GeoCalib(weights="pinhole").to(device)

    rows = {v: [] for v in want}
    t0 = time.time()
    for i, (pf, rf) in enumerate(zip(pin_files, raw_files)):
        pin = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)
        blur_pin = float(cv2.Laplacian(pin, cv2.CV_64F).var())
        blur_raw = float(cv2.Laplacian(raw, cv2.CV_64F).var())
        base = {"index": i, "frame_raw": rf.name, "frame_pinhole": pf.name,
                "blur_var_pinhole": blur_pin, "blur_var_raw": blur_raw}
        for v in want:
            img = raw if v == "b" else pin
            feed = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            feed = feed.permute(2, 0, 1).float().to(device) / 255.0
            if v == "b":
                priors, cm, model = None, "simple_divisional", models["b"]
            elif v == "a":
                priors, cm, model = {"focal": torch.tensor(float(fx), device=device)}, "pinhole", models["pin"]
            else:
                priors, cm, model = None, "pinhole", models["pin"]
            with torch.no_grad():
                res = model.calibrate(feed, camera_model=cm, priors=priors)
            rows[v].append(_row(res, {**base, "variant": v}))
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    info = {"seq": seq, "n_frames": len(pin_files), "variants": want, "seconds": dt,
            "estimates_per_s": len(pin_files) * len(want) / dt if dt else 0.0,
            "peak_gpu_gb": peak, "geocalib_commit": GEOCALIB_COMMIT,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "focal_prior_px": fx, "blur_metric": "variance of the Laplacian, 8-bit grey"}
    for v in want:
        (out_dir / f"{seq}__{v}.jsonl").write_text("\n".join(json.dumps(r) for r in rows[v]))
    (out_dir / f"{seq}__summary.json").write_text(json.dumps(info, indent=1))
    e_data.commit()
    print(json.dumps(info), flush=True)
    return json.dumps(info)


@app.local_entrypoint()
def main(seqs: str = "", variants: str = "abc", max_frames: int = 450,
         parallel: bool = False) -> None:
    import json

    from src.depicted.uzh import all_sequences

    wanted = [s for s in seqs.split(",") if s] or all_sequences()
    if parallel:
        handles = [(s, sequence.spawn(s, variants, max_frames)) for s in wanted]
        for s, h in handles:
            try:
                print(json.loads(h.get()), flush=True)
            except Exception as exc:  # one failed sequence must not lose the others
                print(f"FAILED {s}: {exc}", flush=True)
    else:
        for s in wanted:
            print(json.loads(sequence.remote(s, variants, max_frames)), flush=True)
