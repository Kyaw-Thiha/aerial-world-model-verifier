"""Track L, L02: fine-tune the depicted-tilt reader on simulated and real frames, on Modal.

One function per fold, one fold per container, resumable by writing each fold to the volume.
The reader, the loss, the optimiser and the LM-in-the-loop forward are Track G03's
(``remote/g_tilt_reader.py``, track-g-segments2 ``c189727``) with three changes that the Track L
pre-registration fixes:

* the training pool is sampled to exactly ``frames_per_epoch`` frames per epoch, so every arm
  and every fold takes the same number of gradient steps -- the budget is matched by steps, not
  by dataset size;
* the simulated domain is added: frames from ``/vol/l02/frames`` with exact ``up_cam`` labels;
* the mixed arm applies the pre-registered domain randomisation to simulated frames only
  (photometric, camera roll, and a synthetic ``simple_divisional`` fisheye warp). The gravity
  direction is invariant to a change of camera model, so these warps leave the label exact.

Arms (``--arm``): ``real`` (TII only, distorted route), ``sim`` (simulated only, pinhole route),
``mixed`` (both, distorted route, DR on the simulated side).

    pixi run modal run remote/l_tilt_reader.py --mode smoke
    pixi run modal run remote/l_tilt_reader.py --mode fold --arm sim --holdout forest_ridge
    pixi run modal run remote/l_tilt_reader.py --mode final --arm mixed
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("awmv-l-tilt-reader")
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

SIM_ROOT = Path(VOL) / "l02" / "frames"
REAL_ROOT = Path(VOL) / "l02" / "real_frames"
DEV_FLIGHTS = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
               "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELD_FLIGHTS = ["flight-13p-trackRATM", "flight-07a-lemniscate"]
SIM_FAMILIES = ["urban_checker", "forest_ridge", "industrial"]


# --------------------------------------------------------------------------- data
def _sim_units() -> dict[str, list[dict]]:
    units: dict[str, list[dict]] = {}
    for clip_dir in sorted(SIM_ROOT.iterdir()):
        if not clip_dir.is_dir():
            continue
        lab = json.loads((clip_dir / "labels.json").read_text())
        key = f"{lab['family']}::{lab['scene_seed']}"
        for fr in lab["frames"]:
            fr["path"] = str(clip_dir / f"{fr['frame']:05d}.jpg")
            fr["clip_id"] = lab["clip_id"]
            fr["family"] = lab["family"]
        units.setdefault(key, []).extend(lab["frames"])
    return units


def _real_units() -> dict[str, list[dict]]:
    units: dict[str, list[dict]] = {}
    for flight_dir in sorted(REAL_ROOT.iterdir()):
        if not flight_dir.is_dir():
            continue
        lab = json.loads((flight_dir / "labels.json").read_text())
        for fr in lab["frames"]:
            fr["path"] = str(flight_dir / fr["frame"])
            fr["flight"] = flight_dir.name
        units[flight_dir.name] = lab["frames"]
    return units


# --------------------------------------------------------------------------- domain randomisation
def _roll(x, deg: float):
    import cv2
    import numpy as np

    h, w = x.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    img = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return img, Rz


def _photometric(x, rng):
    import cv2
    import numpy as np

    out = x.astype(np.float32)
    gain = float(rng.uniform(0.75, 1.25))
    bias = float(rng.uniform(-12.0, 12.0))
    out = out * gain + bias
    if rng.random() < 0.5:
        k = int(rng.integers(3, 8)) | 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.3:                       # mild sensor noise, not a texture swap
        out = out + rng.normal(0.0, 2.0, out.shape)
    return np.clip(out, 0, 255)


def _fisheye_warp(x, f_in: float, s: float, k1: float, cache: dict):
    """Pinhole image -> synthetic ``simple_divisional`` image with the same optical axis.

    GeoCalib's simple-divisional model maps a normalised undistorted point ``p`` to
    ``p * (1 - sqrt(1 - 4 k1 r^2)) / (2 k1 r^2)``; the output pixel of a ray is therefore the
    *distorted* position. We invert that map (its undistortion is ``p / (1 + k1 r^2)``) to build
    a backward map from the output grid to the pinhole source. The gravity direction in camera
    coordinates is unchanged, so the label needs no correction.
    """
    import cv2
    import numpy as np

    h, w = x.shape[:2]
    key = (round(s, 3), round(k1, 4))
    if key not in cache:
        f_out = f_in * s
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        px = (u - w / 2.0) / f_out
        py = (v - h / 2.0) / f_out
        r2 = px ** 2 + py ** 2
        denom = 1.0 + k1 * r2
        ux, uy = px / denom, py / denom
        mapx = (ux * f_in + w / 2.0).astype(np.float32)
        mapy = (uy * f_in + h / 2.0).astype(np.float32)
        cache[key] = (mapx, mapy)
    mapx, mapy = cache[key]
    return cv2.remap(x, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


# --------------------------------------------------------------------------- training core
def _train_and_eval(arm: str, holdout: str, seed: int, epochs: int, lr: float, batch: int,
                    lm_steps: int, frames_per_epoch: int, eval_rows: str = "heldout",
                    dr: bool = True) -> dict:
    import time

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from geocalib import GeoCalib

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    sim_units = _sim_units()
    real_units = _real_units()

    sim_train_keys, real_train_keys = [], []
    if arm in ("sim", "mixed"):
        sim_train_keys = [k for k in sim_units if k.split("::")[0] != holdout]
    if arm in ("real", "mixed"):
        real_train_keys = [k for k in DEV_FLIGHTS if k != holdout]
    sim_pool = [fr for k in sim_train_keys for fr in sim_units[k]]
    real_pool = [fr for k in real_train_keys for fr in real_units[k]]
    if arm == "real":
        pool = real_pool
    elif arm == "sim":
        pool = sim_pool
    else:
        pool = [("real", r) for r in real_pool] + [("sim", r) for r in sim_pool]
    if not pool:
        raise SystemExit(f"empty pool for arm {arm} holdout {holdout}")

    weights = "pinhole" if arm == "sim" else "distorted"
    calib = GeoCalib(weights=weights).to("cuda")
    net = calib.model
    net.optimizer.num_steps = lm_steps

    def load(fr, tag):
        img = cv2.imread(fr["path"], cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        up = np.asarray(fr["up_cam"], dtype=np.float64)
        cm = "pinhole"
        if tag == "real":
            cm = "simple_divisional"
        elif arm == "mixed" and dr:
            cm = "simple_divisional"
            s = float(rng.uniform(0.85, 1.15))
            k1 = float(rng.uniform(0.05, 0.30))
            img = _fisheye_warp(img, f_in=320.0, s=s, k1=k1, cache=warp_cache)
        if dr and arm == "mixed" and tag == "sim":
            img = _photometric(img, rng)
            if rng.random() < 0.7:
                deg = float(rng.uniform(-15.0, 15.0))
                img, Rz = _roll(img, deg)
                up = Rz @ up
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return x, torch.tensor(up, dtype=torch.float32), cm

    warp_cache: dict = {}

    def batches(n_frames):
        """Yield batches that never mix domains (and therefore never mix camera models).

        Mixed arms alternate one batch from each domain, half the epoch each: the pre-registered
        budget is matched in steps, and G03's rule (no fold mixes two camera models) is met at the
        batch level as well as the fold level.
        """
        if arm != "mixed":
            tag = "real" if arm == "real" else "sim"
            sub = pool
            idx = rng.permutation(len(sub))[:n_frames] if n_frames <= len(sub) else \
                rng.choice(len(sub), size=n_frames, replace=True)
            buf, ys, cms, shape = [], [], [], None
            for i in idx:
                got = load(sub[i], tag)
                if got is None:
                    continue
                x, y, cm = got
                if shape is not None and tuple(x.shape) != shape:
                    yield torch.stack(buf), torch.stack(ys), cms
                    buf, ys, cms = [], [], []
                shape = tuple(x.shape)
                buf.append(x)
                ys.append(y)
                cms.append(cm)
                if len(buf) == batch:
                    yield torch.stack(buf), torch.stack(ys), cms
                    buf, ys, cms, shape = [], [], [], None
            if buf:
                yield torch.stack(buf), torch.stack(ys), cms
            return

        # mixed: separate streams, one batch from each in turn
        streams = {}
        for tag in ("real", "sim"):
            sub = real_pool if tag == "real" else sim_pool
            n = n_frames // 2 if tag == "real" else n_frames - n_frames // 2
            idx = (rng.permutation(len(sub))[:n] if n <= len(sub)
                   else rng.choice(len(sub), size=n, replace=True))
            streams[tag] = idx

        def stream(tag):
            buf, ys, cms = [], [], []
            for i in streams[tag]:
                got = load(real_pool[i] if tag == "real" else sim_pool[i], tag)
                if got is None:
                    continue
                x, y, cm = got
                buf.append(x)
                ys.append(y)
                cms.append(cm)
                if len(buf) == batch:
                    yield torch.stack(buf), torch.stack(ys), cms
                    buf, ys, cms = [], [], []
            if buf:
                yield torch.stack(buf), torch.stack(ys), cms

        it_r, it_s = stream("real"), stream("sim")
        while True:
            got_any = False
            for it in (it_r, it_s):
                try:
                    yield next(it)
                    got_any = True
                except StopIteration:
                    pass
            if not got_any:
                break

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr,
                            weight_decay=1e-4)
    steps_per_epoch = max(1, frames_per_epoch // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    net.train()
    t0 = time.time()
    hist = []
    for epoch in range(epochs):
        run, nb = 0.0, 0
        for xb, yb, cms in batches(frames_per_epoch):
            if len(set(cms)) == 1:                     # one camera model: true batching
                net.optimizer.set_camera_model(cms[0])
                net.optimizer.shared_intrinsics = False
                data = calib.image_processor(xb.to("cuda"))
                o = net(data | {"camera_R_rig": None})
                up = F.normalize(o["gravity"].vec3d, dim=-1)
            else:                                      # mixed batch: one forward per sample
                out = []
                for j in range(len(xb)):
                    net.optimizer.set_camera_model(cms[j])
                    net.optimizer.shared_intrinsics = False
                    o = net(calib.image_processor(xb[j:j + 1].to("cuda")) |
                            {"camera_R_rig": None})
                    out.append(F.normalize(o["gravity"].vec3d, dim=-1))
                up = torch.cat(out, 0)
            loss = (1.0 - (up * F.normalize(yb.to("cuda"), dim=-1)).sum(-1)).mean()
            if not loss.requires_grad:
                raise RuntimeError(
                    f"loss has no grad: arm={arm} holdout={holdout} cms={set(cms)} "
                    f"n_requires_grad={sum(p.requires_grad for p in net.parameters())} "
                    f"training={net.training}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss)
            nb += 1
        hist.append({"epoch": epoch, "loss": run / max(nb, 1), "steps": nb,
                     "seconds": time.time() - t0})
        print(f"[{arm}/{holdout}] epoch {epoch} loss {run / max(nb, 1):.4f}", flush=True)

    @torch.no_grad()
    def evaluate(rows, tag):
        net.eval()
        out_rows = []
        for k in range(0, len(rows), 2):
            chunk = rows[k:k + 2]
            xs, keep, cms = [], [], []
            for r in chunk:
                img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cm = "simple_divisional" if tag == "real" else "pinhole"
                if tag == "sim" and arm == "mixed":
                    cm = "simple_divisional"
                    img = _fisheye_warp(img, f_in=320.0, s=1.0, k1=0.15, cache=warp_cache)
                xs.append(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
                          .float() / 255.0)
                keep.append(r)
                cms.append(cm)
            if not xs:
                continue
            for j, r in enumerate(keep):
                net.optimizer.set_camera_model(cms[j])
                net.optimizer.shared_intrinsics = False
                o = net(calib.image_processor(xs[j][None].to("cuda")) |
                        {"camera_R_rig": None})
                u = F.normalize(o["gravity"].vec3d, dim=-1).cpu().numpy()[0]
                out_rows.append({"frame": r["frame"], "clip_id": r.get("clip_id"),
                                 "flight": r.get("flight"),
                                 "up_cam_ft": [float(v) for v in u],
                                 "up_cam": r["up_cam"],
                                 "up_cam_base": r.get("up_cam_base")})
        net.train()
        return out_rows

    rec = {"arm": arm, "holdout": holdout, "seed": seed, "epochs": epochs, "lr": lr,
           "batch": batch, "lm_steps": lm_steps, "frames_per_epoch": frames_per_epoch,
           "dr": bool(dr and arm == "mixed"), "weights": weights,
           "n_train_pool": len(pool), "n_real_pool": len(real_pool),
           "n_sim_pool": len(sim_pool), "history": hist}

    if eval_rows == "heldout":
        if arm in ("real", "mixed"):
            rec["held_out_real"] = evaluate(real_units.get(holdout, []), "real")
        if arm in ("sim", "mixed"):
            keys = [k for k in sim_units if k.split("::")[0] == holdout]
            rec["held_out_sim"] = evaluate([fr for k in keys for fr in sim_units[k]], "sim")
    elif eval_rows == "all_real":
        rec["all_real"] = evaluate([fr for k in real_units for fr in real_units[k]], "real")
    elif eval_rows == "held_real":
        rec["held_real"] = evaluate([fr for k in HELD_FLIGHTS for fr in real_units[k]], "real")
    elif eval_rows == "all_sim":
        rec["all_sim"] = evaluate([fr for k in sim_units for fr in sim_units[k]], "sim")

    if holdout == "none":                       # final model: keep the weights for downstream
        mdir = Path(VOL) / "l02" / "models"
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), mdir / f"{arm}.pt")
        rec["checkpoint"] = f"/l02/models/{arm}.pt"

    key = f"{arm}__{holdout}"
    out_dir = Path(VOL) / "l02" / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.json").write_text(json.dumps(rec))
    volume.commit()
    print(f"[{key}] wrote {len(rec.get('held_out_real', []) or rec.get('all_real', []))} real rows, "
          f"{len(rec.get('held_out_sim', []) or rec.get('all_sim', []))} sim rows, "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"arm": arm, "holdout": holdout, "key": key, "seconds": time.time() - t0,
            "n_train_pool": len(pool), "history": hist}


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=6 * 3600)
def fold(arm: str, holdout: str, seed: int = 1, epochs: int = 2, lr: float = 2e-5,
         batch: int = 2, lm_steps: int = 10, frames_per_epoch: int = 1600) -> dict:
    return _train_and_eval(arm, holdout, seed, epochs, lr, batch, lm_steps,
                           frames_per_epoch, "heldout")


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=6 * 3600)
def final(arm: str, seed: int = 101, epochs: int = 2, lr: float = 2e-5, batch: int = 2,
          lm_steps: int = 10, frames_per_epoch: int = 1600) -> dict:
    """Train on every development unit the arm may use, then score all real frames once."""
    holdout = "none"
    return _train_and_eval(arm, holdout, seed, epochs, lr, batch, lm_steps,
                           frames_per_epoch, "held_real")


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=1800)
def smoke(arm: str = "mixed", steps: int = 6) -> dict:
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from geocalib import GeoCalib

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    calib = GeoCalib(weights="pinhole" if arm == "sim" else "distorted").to("cuda")
    net = calib.model
    net.optimizer.num_steps = 10
    import cv2

    sim = _sim_units()
    rows = [fr for k in sorted(sim) for fr in sim[k]][:64]
    cache: dict = {}
    xb = []
    ys = []
    for r in rows[:16]:
        img = cv2.cvtColor(cv2.imread(r["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = _photometric(img, rng)
        img = _fisheye_warp(img, 320.0, 0.95, 0.15, cache)
        xb.append(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0)
        ys.append(torch.tensor(r["up_cam"], dtype=torch.float32))
    xb = torch.stack(xb)
    ys = torch.stack(ys)
    net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2e-5)
    t0 = time.time()
    for _ in range(steps):
        for j in range(4):
            net.optimizer.set_camera_model("simple_divisional")
            net.optimizer.shared_intrinsics = False
            o = net(calib.image_processor(xb[j:j + 1].to("cuda")) | {"camera_R_rig": None})
            up = F.normalize(o["gravity"].vec3d, dim=-1)
            loss = (1.0 - (up * F.normalize(ys[j:j + 1].to("cuda"), dim=-1)).sum(-1)).mean()
            if not loss.requires_grad:
                raise RuntimeError(
                    f"loss has no grad: arm={arm} holdout={holdout} cms={set(cms)} "
                    f"n_requires_grad={sum(p.requires_grad for p in net.parameters())} "
                    f"training={net.training}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    sec = (time.time() - t0) / (steps * 4)
    return {"arm": arm, "sec_per_frame": sec, "frames": steps * 4,
            "gpu": torch.cuda.get_device_name(0)}


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=1800)
def check() -> dict:
    """Two steps on two simulated and two real frames, then one frame per domain scored."""
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from geocalib import GeoCalib

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    calib = GeoCalib(weights="distorted").to("cuda")
    net = calib.model
    net.optimizer.num_steps = 10
    sim = _sim_units()
    real = _real_units()
    sim_rows = [fr for k in sorted(sim) for fr in sim[k]][:2]
    real_rows = [fr for k in DEV_FLIGHTS[:1] for fr in real[k]][:2]
    cache: dict = {}
    items = []
    for r in sim_rows:
        img = cv2.cvtColor(cv2.imread(r["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = _photometric(img, rng)
        img = _fisheye_warp(img, 320.0, 0.95, 0.15, cache)
        items.append((img, r["up_cam"], "simple_divisional"))
    for r in real_rows:
        img = cv2.cvtColor(cv2.imread(r["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        items.append((img, r["up_cam"], "simple_divisional"))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2e-5)
    net.train()
    for _ in range(2):
        for img, y, cm in items:
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            net.optimizer.set_camera_model(cm)
            net.optimizer.shared_intrinsics = False
            o = net(calib.image_processor(x[None].to("cuda")) | {"camera_R_rig": None})
            up = F.normalize(o["gravity"].vec3d, dim=-1)
            yb = F.normalize(torch.tensor(y, dtype=torch.float32)[None].to("cuda"), dim=-1)
            loss = (1.0 - (up * yb).sum(-1)).mean()
            if not loss.requires_grad:
                raise RuntimeError(
                    f"loss has no grad: arm={arm} holdout={holdout} cms={set(cms)} "
                    f"n_requires_grad={sum(p.requires_grad for p in net.parameters())} "
                    f"training={net.training}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    errs = []
    with torch.no_grad():
        for img, y, cm in items:
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            net.optimizer.set_camera_model(cm)
            o = net(calib.image_processor(x[None].to("cuda")) | {"camera_R_rig": None})
            up = F.normalize(o["gravity"].vec3d, dim=-1).cpu().numpy()[0]
            errs.append(float(np.degrees(np.arccos(np.clip(up @ np.asarray(y), -1, 1)))))
    return {"ok": True, "n_sim_pool": sum(len(v) for v in sim.values()),
            "n_real_pool": sum(len(v) for v in real.values()),
            "eval_errors_deg": errs, "gpu": torch.cuda.get_device_name(0)}


@app.local_entrypoint()
def main(mode: str = "smoke", arm: str = "mixed", holdout: str = "", seed: int = 1,
         epochs: int = 2, lr: float = 2e-5, batch: int = 2, lm_steps: int = 10,
         frames_per_epoch: int = 1600, steps: int = 6, jobs: str = ""):
    if mode == "smoke":
        print(smoke.remote(arm, steps))
        return
    if mode == "check":
        print(check.remote())
        return
    if mode == "fold":
        print(fold.remote(arm, holdout, seed, epochs, lr, batch, lm_steps, frames_per_epoch))
        return
    if mode == "final":
        print(final.remote(arm, seed, epochs, lr, batch, lm_steps, frames_per_epoch))
        return
    if mode == "finals":
        calls = [final.spawn(a, 101, epochs, lr, batch, lm_steps, frames_per_epoch)
                 for a in ("real", "sim", "mixed")]
        for c in calls:
            print(c.get())
        return
    if mode == "batch":
        # jobs: "arm:holdout:seed,..." -- spawn one container per fold, wait in order
        specs = [s.split(":") for s in jobs.split(",") if s]
        calls = [fold.spawn(a, h, int(sd), epochs, lr, batch, lm_steps, frames_per_epoch)
                 for a, h, sd in specs]
        for c in calls:
            print(c.get())
        return
    raise SystemExit(f"unknown mode {mode}")
