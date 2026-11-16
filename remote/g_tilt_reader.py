"""Fine-tune the depicted-tilt reader (GeoCalib) on mocap-labelled attitude frames, on Modal.

Report G03. Pre-registration ``configs/g03_tilt_reader_prereg.yaml``.

Two datasets, two frozen reader variants (they are not interchangeable: the camera models
differ, so no fold mixes them):

* ``tii`` - raw TII fisheye at 640x480, the adopted verdict route (``weights="distorted"``,
  camera model ``simple_divisional``);
* ``uzh`` - the UZH-FPV forward-rig pinhole export, ``weights="pinhole"``, camera model
  ``pinhole``.

Folds are leave-one-flight (TII, 8) and leave-one-sequence (UZH, 3). A held-out unit's frames
never enter training, augmentation or the confidence weighting of its fold. Training uses the
network output after GeoCalib's own LM refinement, with the LM step count reduced (the LM is
the expensive part; ``--lm-steps`` records it), and a cosine loss on the up direction.

    modal run remote/g_tilt_reader.py --mode smoke
    modal run --detach remote/g_tilt_reader.py --mode run --epochs 2
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-g-tilt-reader")

volume = modal.Volume.from_name("awmv-g-data", create_if_missing=True)
VOL = "/vol"
GEOCALIB_COMMIT = "1dedb29f94fc11b896c5f572005978aeebccd67f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("numpy<2", "scipy", "opencv-python-headless==4.10.0.84",
                 "torch==2.4.1", "torchvision==0.19.1", "matplotlib")
    .pip_install("kornia", "omegaconf", "poselib", "huggingface_hub")
    .run_commands(f"pip install git+https://github.com/cvg/GeoCalib@{GEOCALIB_COMMIT}")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("data/g03_frames", "/frames/tii")
    .add_local_dir("data/g03_uzh", "/frames/uzh")
)

DATASETS = {"tii": {"weights": "distorted", "camera_model": "simple_divisional",
                    "root": "/frames/tii", "local": "data/g03_frames"},
            "uzh": {"weights": "pinhole", "camera_model": "pinhole",
                    "root": "/frames/uzh", "local": "data/g03_uzh"}}


def _units(dataset: str) -> list[str]:
    root = Path(DATASETS[dataset]["root"])
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "labels.json").exists())


def _load_unit(dataset: str, unit: str) -> list[dict]:
    root = Path(DATASETS[dataset]["root"]) / unit
    labels = json.loads((root / "labels.json").read_text())["frames"]
    for row in labels:
        row["path"] = str(root / row["frame"])
    return labels


@app.function(image=image, gpu="A10G", timeout=1800)
def smoke(dataset: str = "tii", n: int = 2, lm_steps: int = 10) -> dict:
    """One forward+backward with gradients: does the LM keep the graph alive?"""
    import cv2
    import numpy as np
    import torch
    from geocalib import GeoCalib

    cfg = DATASETS[dataset]
    torch.manual_seed(0)
    calib = GeoCalib(weights=cfg["weights"]).to("cuda")
    net = calib.model
    net.optimizer.num_steps = lm_steps
    unit = _units(dataset)[0]
    rows = _load_unit(dataset, unit)[:n]
    imgs, labels = [], []
    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        labels.append(torch.tensor(r["up_cam"], dtype=torch.float32))
    x = torch.stack(imgs).to("cuda")
    y = torch.nn.functional.normalize(torch.stack(labels).to("cuda"), dim=-1)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-6)
    data = calib.image_processor(x)
    net.optimizer.set_camera_model(cfg["camera_model"])
    net.optimizer.shared_intrinsics = False
    out = net(data | {"camera_R_rig": None})
    up = torch.nn.functional.normalize(out["gravity"].vec3d, dim=-1)
    loss = (1.0 - (up * y).sum(-1)).mean()
    loss.backward()
    gnorm = float(sum((p.grad.detach().norm() ** 2).item() for p in net.parameters()
                      if p.grad is not None) ** 0.5)
    # a few real optimisation steps, so the per-fold cost can be estimated before a long run
    import time
    t0 = time.time()
    n_steps = 5
    for _ in range(n_steps):
        data = calib.image_processor(x)
        net.optimizer.set_camera_model(cfg["camera_model"])
        out = net(data | {"camera_R_rig": None})
        up = torch.nn.functional.normalize(out["gravity"].vec3d, dim=-1)
        step_loss = (1.0 - (up * y).sum(-1)).mean()
        opt.zero_grad(set_to_none=True)
        step_loss.backward()
        opt.step()
    sec_per_step = (time.time() - t0) / n_steps
    with torch.no_grad():
        err = torch.rad2deg(torch.arccos(torch.clamp((up * y).sum(-1), -1, 1)))
    return {"dataset": dataset, "unit": unit, "n": n, "lm_steps": lm_steps, "batch": n,
            "sec_per_step_batch": sec_per_step,
            "loss": float(loss), "grad_norm": gnorm,
            "per_frame_err_deg": [float(v) for v in err],
            "n_tensors_with_grad": sum(1 for p in net.parameters() if p.grad is not None),
            "n_params": sum(1 for _ in net.parameters())}


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=6 * 3600)
def run_fold(dataset: str, holdout: str, epochs: int = 2, lr: float = 2e-5,
             batch: int = 2, lm_steps: int = 10, seed: int = 1,
             max_frames_per_unit: int = 0) -> dict:
    """Fine-tune on every unit except ``holdout``, then score ``holdout`` and the fit units."""
    import math
    import time

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from geocalib import GeoCalib

    cfg = DATASETS[dataset]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    calib = GeoCalib(weights=cfg["weights"]).to("cuda")
    net = calib.model
    net.optimizer.num_steps = lm_steps
    units = _units(dataset)
    train_units = [u for u in units if u != holdout]

    def frames(unit_list, cap=0):
        rows = []
        for u in unit_list:
            rs = _load_unit(dataset, u)
            if cap and len(rs) > cap:
                idx = np.linspace(0, len(rs) - 1, cap).astype(int)
                rs = [rs[i] for i in idx]
            rows.extend(rs)
        return rows

    train_rows = frames(train_units, max_frames_per_unit)
    print(f"[{dataset}/{holdout}] train {len(train_rows)} frames from {len(train_units)} units",
          flush=True)

    def to_tensor(row):
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.tensor(row["up_cam"], dtype=torch.float32)
        return x, y

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr,
                            weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_rows) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    net.train()
    t0 = time.time()
    hist = []
    for epoch in range(epochs):
        order = rng.permutation(len(train_rows))
        run, nb = 0.0, 0
        for i in range(0, len(order) - batch + 1, batch):
            xs, ys = [], []
            for j in order[i:i + batch]:
                x, y = to_tensor(train_rows[j])
                if x is None:
                    continue
                xs.append(x)
                ys.append(y)
            if not xs:
                continue
            xb = torch.stack(xs).to("cuda")
            yb = F.normalize(torch.stack(ys).to("cuda"), dim=-1)
            data = calib.image_processor(xb)
            net.optimizer.set_camera_model(cfg["camera_model"])
            net.optimizer.shared_intrinsics = False
            out = net(data | {"camera_R_rig": None})
            up = F.normalize(out["gravity"].vec3d, dim=-1)
            loss = (1.0 - (up * yb).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss)
            nb += 1
        hist.append({"epoch": epoch, "loss": run / max(nb, 1), "steps": nb,
                     "seconds": time.time() - t0})
        torch.cuda.empty_cache()
        print(f"[{dataset}/{holdout}] epoch {epoch} loss {run / max(nb,1):.4f}", flush=True)

    @torch.no_grad()
    def evaluate(rows):
        net.eval()
        out_rows = []
        ev_batch = 2   # the LM keeps a graph-sized working set even under no_grad
        for k in range(0, len(rows), ev_batch):
            chunk = rows[k:k + ev_batch]
            xs, keep = [], []
            for r in chunk:
                x, _ = to_tensor(r)
                if x is not None:
                    xs.append(x)
                    keep.append(r)
            if not xs:
                continue
            xb = torch.stack(xs).to("cuda")
            data = calib.image_processor(xb)
            net.optimizer.set_camera_model(cfg["camera_model"])
            net.optimizer.shared_intrinsics = False
            out = net(data | {"camera_R_rig": None})
            up = F.normalize(out["gravity"].vec3d, dim=-1).cpu().numpy()
            for r, u in zip(keep, up):
                out_rows.append({"unit": r.get("unit", holdout), "frame": r["frame"],
                                 "up_cam_ft": [float(v) for v in u],
                                 "up_cam": r["up_cam"],
                                 "up_cam_base": r.get("up_cam_base"),
                                 "conf_base": r.get("conf_base")})
        net.train()
        return out_rows

    hold_rows = _load_unit(dataset, holdout)
    eval_hold = evaluate(hold_rows)
    # a fit-unit sample, to show the fine-tune is not just moving the held-out unit
    fit_sample = frames(train_units, 40)
    for r in fit_sample:
        r["unit"] = r["path"].split("/")[-2]
    eval_fit = evaluate(fit_sample)

    rec = {"dataset": dataset, "holdout": holdout, "epochs": epochs, "lr": lr,
           "batch": batch, "lm_steps": lm_steps, "seed": seed,
           "n_train_frames": len(train_rows), "history": hist,
           "held_out": eval_hold, "fit_sample": eval_fit}
    vol.mkdir(exist_ok=True)
    out_dir = Path(VOL) / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{dataset}__{holdout}.json").write_text(json.dumps(rec))
    volume.commit()
    print(f"[{dataset}/{holdout}] wrote {out_dir / (dataset + '__' + holdout + '.json')}",
          flush=True)
    return {"dataset": dataset, "holdout": holdout, "n": len(eval_hold),
            "seconds": time.time() - t0, "loss_last": hist[-1]["loss"] if hist else None}


@app.local_entrypoint()
def main(mode: str = "smoke", dataset: str = "tii", holdout: str = "",
         epochs: int = 2, lr: float = 2e-5, batch: int = 2, lm_steps: int = 10,
         max_frames: int = 0):
    if mode == "smoke":
        print(smoke.remote(dataset))
        return
    if holdout:
        units = [holdout]
    else:  # the container path does not exist locally; enumerate from the prepared tree
        root = Path(DATASETS[dataset]["local"])
        units = sorted(p.name for p in root.iterdir()
                       if p.is_dir() and (p / "labels.json").exists())
    calls = [run_fold.spawn(dataset, u, epochs, lr, batch, lm_steps, 1 + i, max_frames)
             for i, u in enumerate(units)]
    for c in calls:
        print(c.get())
