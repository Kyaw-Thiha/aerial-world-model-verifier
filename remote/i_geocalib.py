"""GeoCalib on simulated frames (Track I spike, and the Step 3 sim-to-real check), on Modal.

The tilt reader was developed on real images, so before any detection number can mean
anything on simulated flights we have to report its error on simulated frames against exact
ground truth. This app measures, per frame:

* ``up_cam`` from GeoCalib with the project's existing settings (pinhole model, known focal
  prior, gravity vector in camera coordinates);
* ground truth ``up_cam = R_wc^T e3`` straight from the render poses (the renderer was
  *given* the camera pose, so this is not an inference);
* derived: the angular error of the up vector; the depicted body tilt through the mount
  rotation against the commanded tilt from flatness; and the depicted lean vector against
  both the ground-truth lean and the lean the path requires.

The last comparison is the spike's end-to-end check: on the ``feasible`` clip the depicted
lean should follow the required lean (slope ~1), on the ``level`` clip it should not
(slope ~0), which is exactly the verifier's gain statistic read early.

    pixi run modal run remote/i_geocalib.py::score --tag i01_feasible
    pixi run modal run remote/i_geocalib.py::score --tag i01_level
"""
from __future__ import annotations

import modal

app = modal.App("awmv-i-geocalib")
volume = modal.Volume.from_name("awmv-i-data", create_if_missing=True)

GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "pandas", "pyyaml", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("kornia", "omegaconf", "poselib", "huggingface_hub")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .add_local_python_source("src")
)


def _summ(x) -> dict:
    import numpy as np

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {"n": int(x.size), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}


@app.function(image=image, gpu="A10G", volumes={"/vol": volume}, timeout=3600)
def score(tag: str = "i01_feasible", camera_model: str = "pinhole",
          focal_prior: bool = True) -> dict:
    import json
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    from src.verdict.lean import depicted_lean_from_gravity

    job = json.loads(Path(f"/vol/spike/{tag}_job.json").read_text())
    job["poses"] = f"/vol/spike/{tag}_poses.npz"     # the job was written for a local run
    npz = np.load(job["poses"])
    poses = npz["poses"]
    R_cb = np.asarray(job["mount_R_cb"], dtype=float)
    width, height, hfov = job["width"], job["height"], job["hfov_deg"]
    f = 0.5 * width / np.tan(np.radians(hfov) / 2.0)

    e3 = np.array([0.0, 0.0, 1.0])
    up_gt = np.einsum("nji,j->ni", poses[:, :3, :3], e3)      # R_wc^T e3
    tilt_gt = np.degrees(np.arccos(np.clip((up_gt @ R_cb.T) @ e3, -1, 1)))
    lean_gt = depicted_lean_from_gravity(up_gt, R_cb)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights="pinhole" if camera_model == "pinhole" else "distorted").to(device)

    fdir = Path(f"/vol/frames/{tag}")
    names = sorted(fdir.glob("*.png"))
    if not names:
        return {"tag": tag, "error": f"no frames under {fdir}"}
    rows = []
    t0 = time.time()
    for p in names:
        i = int(p.stem)
        feed = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(feed).permute(2, 0, 1).float().to(device) / 255.0
        priors = {"focal": torch.tensor(float(f), device=device)} if focal_prior else None
        with torch.no_grad():
            res = model.calibrate(t, camera_model=camera_model, priors=priors)
        g = np.asarray(res["gravity"].vec3d.detach().cpu()).ravel()
        g = g / np.linalg.norm(g)
        up_body = R_cb @ g
        tilt_est = float(np.degrees(np.arccos(np.clip(up_body @ e3, -1, 1))))
        lean_est = depicted_lean_from_gravity(g[None], R_cb)[0]
        rows.append({
            "frame": p.name, "i": i,
            "up_cam": g.tolist(), "up_error_deg": float(np.degrees(np.arccos(np.clip(g @ up_gt[i], -1, 1)))),
            "tilt_est_deg": tilt_est, "tilt_gt_deg": float(tilt_gt[i]),
            "tilt_error_deg": tilt_est - float(tilt_gt[i]),
            "lean_est": [float(v) for v in lean_est],
            "lean_gt": [float(v) for v in lean_gt[i]],
            "lean_req": [float(v) for v in npz["lean_req"][i]],
            "f_mle": float(np.asarray(res["camera"].f.detach().cpu()).ravel()[0])
            if hasattr(res["camera"], "f") else None,
        })
    dt = time.time() - t0
    le = np.array([r["lean_est"] for r in rows])
    lg = np.array([r["lean_gt"] for r in rows])
    lr = np.array([r["lean_req"] for r in rows])
    terr = np.array([r["tilt_error_deg"] for r in rows])
    out = {
        "tag": tag, "n_frames": len(rows), "seconds": round(dt, 2),
        "fps": round(len(rows) / dt, 2) if dt else None,
        "camera_model": camera_model, "focal_prior": focal_prior, "focal_used_px": f,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "up_error_deg": _summ([r["up_error_deg"] for r in rows]),
        "tilt_error_deg": {**_summ(np.abs(terr)), "bias": _summ(terr).get("median")},
        # required lean is (cos(roll) sin(pitch), -sin(roll)); its magnitude is sin(tilt),
        # so this is the median tilt the path demands over the scored frames
        "tilt_req_median_deg": float(np.degrees(np.arcsin(np.clip(
            np.linalg.norm(lr, axis=1), 0.0, 1.0))).mean()),
        "lean_slope_est_on_gt": _slope(le, lg),
        "lean_slope_est_on_req": _slope(le, lr),
        "lean_slope_gt_on_req": _slope(lg, lr),
        "lean_corr_est_req": _corr(le, lr),
        "rows": rows,
    }
    Path("/vol/geocalib").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/geocalib/{tag}_{camera_model}{'_prior' if focal_prior else ''}.json").write_text(
        json.dumps(out, indent=2))
    volume.commit()
    return {k: v for k, v in out.items() if k != "rows"} | {"rows_head": rows[:3]}


def _slope(est, req) -> dict:
    """Per-axis least-squares slope of estimate on reference (1.0 = faithful, 0 = no tilt)."""
    import numpy as np

    out = {}
    for k, name in enumerate(("forward", "left")):
        x = np.asarray(req, dtype=float)[:, k]
        y = np.asarray(est, dtype=float)[:, k]
        ok = np.isfinite(x) & np.isfinite(y)
        denom = float((x[ok] ** 2).sum())
        out[name] = float((x[ok] * y[ok]).sum() / denom) if denom > 1e-12 else None
    return out


def _corr(a, b) -> dict:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = {}
    for k, name in enumerate(("forward", "left")):
        x, y = b[:, k], a[:, k]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9:
            out[name] = None
        else:
            out[name] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return out
