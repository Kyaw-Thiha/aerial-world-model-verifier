"""Track E: export verdict clips for UZH-FPV in the shared schema (Tasks 4-5).

Per sequence: build non-overlapping 4 s clips from the prepared frames, keep the testable
ones, warp the six edit conditions as exact pure rotations about the camera centre, run the
depicted-tilt estimator on the real and every edited clip, and write one
``verdict_clip.npz`` + ``meta.json`` per clip and condition under ``/e-data/clips/<seq>/``.

Two jobs:

* ``validate_edits`` — the pre-registered edit check (5/10/20 deg roll and pitch must be
  recovered, the null edit must reproduce the real clip). Run and read before scoring clips.
* ``export_sequence`` — the export itself.

Both write small JSONL summaries so the local side never has to touch the volume's frames.

    pixi run modal run remote/e_export.py --validate --seqs indoor_forward_5
    pixi run modal run remote/e_export.py --seqs indoor_forward_5 --max-clips 4
"""
from __future__ import annotations

import modal

app = modal.App("awmv-e-export")

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
    .add_local_python_source("src")
)

CONDITIONS = ("real", "null_edit", "tilt_removal", "scale_0.5", "scale_1.5",
              "mirror_roll", "yaw_control")
GEOMETRY = {"a": "pinhole", "c": "pinhole", "b": "kb4"}


def _ensure_weights() -> None:
    import shutil
    from pathlib import Path

    cache = Path("/root/.cache/torch/hub/geocalib")
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("pinhole.tar", "distorted.tar"):
        dst = cache / name
        src = Path(A_DATA) / "geocalib_weights" / f"geocalib-{name}"
        if not dst.exists() and src.exists():
            shutil.copyfile(src, dst)


def _estimator(variant: str):
    from geocalib import GeoCalib

    if variant == "b":
        return GeoCalib(weights="distorted"), "simple_divisional"
    return GeoCalib(weights="pinhole"), "pinhole"


def _run(model, camera_model, img, device, focal=None):
    import cv2
    import numpy as np
    import torch

    feed = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    feed = feed.permute(2, 0, 1).float().to(device) / 255.0
    priors = ({"focal": torch.tensor(float(focal), device=device)}
              if focal is not None else None)
    with torch.no_grad():
        res = model.calibrate(feed, camera_model=camera_model, priors=priors)
    g = res["gravity"].vec3d.detach().cpu().numpy().ravel()
    conf = res.get("latitude_confidence")
    if conf is not None:
        conf = conf.detach().cpu().numpy() if hasattr(conf, "detach") else conf
        conf = float(np.asarray(conf, dtype=np.float64).mean())
    return g, conf


def maximal_rectangle(mask):
    """Largest axis-aligned all-true rectangle in a boolean mask (histogram/stack method).

    A *centred* rectangle is not always available: intersecting the valid region of seven
    conditions over a clip that manoeuvres hard can leave a valid area that is off-centre, and
    `indoor_forward_7#01` had no centred rectangle above a quarter of the frame. The largest
    rectangle anywhere is just as fixed per clip, and it is the one the estimator deserves.
    """
    import numpy as np

    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best = (0, 0, 0, 0)  # area, x0, y0, x1, y1 through the tuple below
    best_area = 0
    best_rect = (0, 0, 0, 0)
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best_area:
                    best_area = area
                    best_rect = (sx, y - sh + 1, x - 1, y)
                start = sx
            if cur > 0:
                stack.append((start, cur))
    x0, y0, x1, y1 = best_rect
    # y1 is the last row of the rectangle and x1 the last column: both slices are inclusive
    sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    return sl, float((y1 + 1 - y0) / h)


def centred_crop(mask, min_fraction: float = 0.25):
    """Largest centred rectangle (same aspect) whose *actual slice* is fully inside ``mask``.

    The fraction is found by scanning downwards, then the slice is shrunk one pixel at a time
    until ``mask[sl]`` is true everywhere: rounding a fraction to a half-extent can otherwise
    put the corners one pixel outside, which failed the export on four aggressive clips.
    """
    import numpy as np

    h, w = mask.shape
    cy, cx = int(round((h - 1) / 2.0)), int(round((w - 1) / 2.0))
    f = 1.0
    while f > min_fraction:
        hh, ww = int(round(f * h)) // 2, int(round(f * w)) // 2
        sl = (slice(cy - hh, cy + hh), slice(cx - ww, cx + ww))
        if mask[sl].all():
            return sl, float((sl[0].stop - sl[0].start) / h)
        f -= 0.02
    hh = ww = 2
    sl = (slice(cy - hh, cy + hh), slice(cx - ww, cx + ww))
    return sl, float((sl[0].stop - sl[0].start) / h)


def centred_crop_fraction(mask, min_fraction: float = 0.3, step: float = 0.02) -> float:
    """Largest centred rectangle (same aspect as the frame) that lies fully inside ``mask``.

    Cropping beats black fill here: a black border over 20-50% of the frame shifts GeoCalib's
    gravity by about 13 deg (measured through the null-edit control), which is larger than the
    violations being tested. The crop leaves only real pixels and is applied identically to
    every condition of the frame, including the null edit.
    """
    import numpy as np

    h, w = mask.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    f = 1.0
    while f > min_fraction:
        hh, ww = int(round(f * h)) // 2, int(round(f * w)) // 2
        y0, y1 = int(round(cy)) - hh, int(round(cy)) + hh
        x0, x1 = int(round(cx)) - ww, int(round(cx)) + ww
        if y0 >= 0 and x0 >= 0 and y1 < h and x1 < w and mask[y0:y1 + 1, x0:x1 + 1].all():
            return float((y1 - y0 + 1) / h)
        f -= step
    return float(min_fraction)


def _condition_rotations(name: str, R_bw, R_cb, yaw_deg: float):
    import numpy as np

    from src.depicted.edits import Condition, camera_maps

    if name == "real" or name == "null_edit":
        return np.tile(np.eye(3), (len(R_bw), 1, 1))
    if name == "tilt_removal":
        return camera_maps(Condition("tilt_k", k=0.0), R_bw, R_cb)
    if name == "scale_0.5":
        return camera_maps(Condition("tilt_k", k=0.5), R_bw, R_cb)
    if name == "scale_1.5":
        return camera_maps(Condition("tilt_k", k=1.5), R_bw, R_cb)
    if name == "mirror_roll":
        return camera_maps(Condition("mirror_roll"), R_bw, R_cb)
    if name == "yaw_control":
        return camera_maps(Condition("yaw_control", yaw_deg=yaw_deg), R_bw, R_cb)
    raise ValueError(name)


@app.function(image=image, gpu="A10G", volumes={E_DATA: e_data, A_DATA: a_data},
              timeout=3 * 3600)
def validate_edits(seq: str, variant: str = "b", angles: str = "5,10,20",
                   n_frames: int = 12) -> str:
    """Apply known rotations and check that the estimator's gravity follows them."""
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    from src.depicted.edits import kb4_from_meta, pinhole_K, warp
    from src.depicted.uzh import load_truth

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_weights()
    clip = Path(E_DATA) / "uzh" / seq
    meta = json.loads((clip / "meta.json").read_text())
    geometry = GEOMETRY[variant]
    calib = kb4_from_meta(meta) if geometry == "kb4" else pinhole_K(meta)
    folder = "frames_raw" if geometry == "kb4" else "frames_pinhole"
    files = sorted((clip / folder).glob("*.png"))
    truth = load_truth(clip.parent, clip.name)
    step = max(1, len(files) // n_frames)
    model, cm = _estimator(variant)
    model = model.to(device)
    focal = float(meta["intrinsics_pinhole"]["fx"]) if variant == "a" else None

    rows = []
    for i in list(range(0, len(files), step))[:n_frames]:
        img = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        g0, _ = _run(model, cm, img, device, focal)
        g0 = g0 / np.linalg.norm(g0)
        for axis_name in ("roll", "pitch"):
            for ang in [float(a) for a in angles.split(",") if a]:
                ax = {"roll": [1.0, 0.0, 0.0], "pitch": [0.0, 1.0, 0.0]}[axis_name]
                M = Rotation.from_rotvec(np.deg2rad(ang) * np.array(ax)).as_matrix()
                warped, valid = warp(img, M, geometry, calib)
                g1, _ = _run(model, cm, warped, device, focal)
                g1 = g1 / np.linalg.norm(g1)
                # the image was rotated by M, so the same world direction reads as M^-1 g0
                pred = np.linalg.inv(M) @ g0
                pred = pred / np.linalg.norm(pred)
                err = float(np.degrees(np.arccos(np.clip(pred @ g1, -1, 1))))
                rows.append({"seq": seq, "variant": variant, "frame": files[i].name,
                             "axis": axis_name, "angle_deg": ang, "recovery_err_deg": err,
                             "valid_frac": float(valid.mean()),
                             "gravity_before": g0.tolist(),
                             "gravity_after": g1.tolist()})
        # null edit must reproduce the real estimate
        warped, valid = warp(img, np.eye(3), geometry, calib)
        gn, _ = _run(model, cm, warped, device, focal)
        gn = gn / np.linalg.norm(gn)
        rows.append({"seq": seq, "variant": variant, "frame": files[i].name, "axis": "null",
                     "angle_deg": 0.0,
                     "recovery_err_deg": float(np.degrees(np.arccos(np.clip(gn @ g0, -1, 1)))),
                     "valid_frac": float(valid.mean()),
                     "gravity_before": g0.tolist(), "gravity_after": gn.tolist()})

    out = Path(E_DATA) / "edits"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"validate_{seq}__{variant}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    e_data.commit()
    rec = {"seq": seq, "variant": variant, "n": len(rows),
           "median_err_by_angle": {str(a): float(np.median([r["recovery_err_deg"] for r in rows
                                                            if r["angle_deg"] == a]))
                                   for a in sorted({r["angle_deg"] for r in rows})},
           "median_err_null": float(np.median([r["recovery_err_deg"] for r in rows
                                               if r["axis"] == "null"]))}
    print(json.dumps(rec), flush=True)
    return json.dumps(rec)


@app.function(image=image, gpu="A10G", volumes={E_DATA: e_data, A_DATA: a_data},
              timeout=3 * 3600)
def export_sequence(seq: str, variant: str = "b", clip_seconds: float = 4.0,
                    max_clips: int = 4, yaw_deg: float = 20.0,
                    crop_mode: str = "fixed") -> str:
    """Clip, warp, estimate and write one npz per clip and condition.

    Border handling, v3 (`crop_mode="fixed"`, the pre-registered default): **one crop per clip**,
    the largest centred rectangle inside the valid region of every frame and every condition,
    applied to all seven conditions including the source crop for `real`. A per-frame crop
    (`crop_mode="per_frame"`, the E02 behaviour kept for comparison) left the field of view
    changing frame to frame and made the null edit's lean gain a third below the real clip's on
    identical pixels; see reports/E02 and reports/E03.
    """
    import json
    import shutil
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from src.depicted.clips import (clip_bounds, lean_from_up_body, path_side_with_gt,
                                    signal_mask)
    from src.depicted.edits import (kb4_from_meta, kb4_source_coords, pinhole_K,
                                    pinhole_source_coords, warp)
    from src.depicted.uzh import load_truth

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_weights()
    clip_dir = Path(E_DATA) / "uzh" / seq
    meta = json.loads((clip_dir / "meta.json").read_text())
    geometry = GEOMETRY[variant]
    calib = kb4_from_meta(meta) if geometry == "kb4" else pinhole_K(meta)
    folder = "frames_raw" if geometry == "kb4" else "frames_pinhole"
    files = sorted((clip_dir / folder).glob("*.png"))
    truth = load_truth(clip_dir.parent, clip_dir.name)
    gt = np.load(clip_dir / "truth.npz")["gt"]
    path = path_side_with_gt(truth, gt, k_d=0.0)
    R_cb = truth.R_cb()
    yaw = truth.yaw()
    focal = float(meta["intrinsics_pinhole"]["fx"]) if variant == "a" else None
    model, cm = _estimator(variant)
    model = model.to(device)
    source_coords = pinhole_source_coords if geometry == "pinhole" else kb4_source_coords

    fps = float(1.0 / np.median(np.diff(truth.t)))
    bounds = clip_bounds(len(files), clip_seconds, fps)
    out_root = Path(E_DATA) / "clips" / seq
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    index = []
    kept = 0
    for ci, (lo, hi) in enumerate(bounds):
        if kept >= max_clips:
            break
        sig = signal_mask(path["required_tilt_deg"][lo:hi])
        if not sig["testable"]:
            index.append({"clip": f"{seq}#{ci:02d}", "index": [lo, hi], "testable": False,
                          "reasons": sig["abstain_reasons"]})
            continue
        kept += 1
        R_bw_clip = truth.R_bw[lo:hi]
        t_clip = truth.t[lo:hi] - truth.t[lo]
        per_cond = {c: _condition_rotations(c, R_bw_clip, R_cb, yaw_deg) for c in CONDITIONS}
        h, w = cv2.imread(str(files[lo]), cv2.IMREAD_GRAYSCALE).shape
        frames_out = {}
        for cond in CONDITIONS:
            d = out_root / f"{seq}_{ci:02d}_{cond}" / "frames"
            d.mkdir(parents=True, exist_ok=True)
            frames_out[cond] = d

        def inside_mask(M):
            xs, ys, ok = source_coords(M, calib, w, h)
            return ok & (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)

        # Fixed crop first. If the manoeuvre is so large that no single window is valid in
        # every frame and condition (measured: indoor_forward_7), fall back to a fixed *size*
        # with a per-frame offset — all conditions of a frame still share one window, and the
        # field of view no longer jitters in scale, which was E02's defect.
        fallback = False
        if crop_mode == "fixed":
            probe = None
            for cond in CONDITIONS:
                if cond == "real":
                    continue
                for j in range(len(t_clip)):
                    m = inside_mask(per_cond[cond][j])
                    probe = m if probe is None else (probe & m)
            probe_frac = 0.0
            if probe is not None:
                probe_sl, probe_frac = maximal_rectangle(probe)
            if probe is None or probe_frac < 0.15:
                crop_mode = "fixed_size"
                fallback = True
                print(f"{seq}#{ci:02d}: no window is valid in every frame; "
                      f"using a fixed size with a per-frame offset", flush=True)

        if crop_mode == "per_frame":
            crops = None
            mask_all = None
            crop_frac = None
        elif crop_mode == "fixed_size":
            # fixed size for the clip, offset per frame: intersect the conditions of each frame,
            # take the smallest maximal rectangle over the frames, then centre it on the mask
            sizes = []
            for j in range(len(t_clip)):
                m = None
                for cond in CONDITIONS:
                    if cond == "real":
                        continue
                    mm = inside_mask(per_cond[cond][j])
                    m = mm if m is None else (m & mm)
                sl, frac = maximal_rectangle(m)
                sizes.append((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start, m))
            fh = min(x[0] for x in sizes)
            fw = min(x[1] for x in sizes)
            per_frame_windows = []
            for j, (_, _, m) in enumerate(sizes):
                ys, xs = np.nonzero(m)
                cy = int(np.median(ys))
                cx = int(np.median(xs))
                y0 = min(max(0, cy - fh // 2), h - fh)
                x0 = min(max(0, cx - fw // 2), w - fw)
                per_frame_windows.append((slice(y0, y0 + fh), slice(x0, x0 + fw)))
            crops = None
            crop_frac = float(fh / h)
        else:
            # one crop for the whole clip: intersection over every frame and every condition
            mask_all = None
            for cond in CONDITIONS:
                if cond == "real":
                    continue
                for j in range(len(t_clip)):
                    m = inside_mask(per_cond[cond][j])
                    mask_all = m if mask_all is None else (mask_all & m)
            crops, crop_h_frac = maximal_rectangle(mask_all)
            if not mask_all[crops].all():
                # cannot happen for a correct maximal rectangle; fall back rather than abort
                crop_mode = "fixed_size"
                fallback = True
            else:
                crop_frac = float((crops[1].stop - crops[1].start) / w)

        ups = {c: np.full((len(t_clip), 3), np.nan) for c in CONDITIONS}
        confs = {c: np.full(len(t_clip), np.nan) for c in CONDITIONS}
        valids = {c: np.full(len(t_clip), np.nan) for c in CONDITIONS}
        for j, fi in enumerate(range(lo, hi)):
            img = cv2.imread(str(files[fi]), cv2.IMREAD_GRAYSCALE)
            # Fixed crop first. If the manoeuvre is so large that no single window is valid in
        # every frame and condition (measured: indoor_forward_7), fall back to a fixed *size*
        # with a per-frame offset — all conditions of a frame still share one window, and the
        # field of view no longer jitters in scale, which was E02's defect.
        fallback = False
        if crop_mode == "fixed":
            probe = None
            for cond in CONDITIONS:
                if cond == "real":
                    continue
                for j in range(len(t_clip)):
                    m = inside_mask(per_cond[cond][j])
                    probe = m if probe is None else (probe & m)
            probe_frac = 0.0
            if probe is not None:
                probe_sl, probe_frac = maximal_rectangle(probe)
            if probe is None or probe_frac < 0.15:
                crop_mode = "fixed_size"
                fallback = True
                print(f"{seq}#{ci:02d}: no window is valid in every frame; "
                      f"using a fixed size with a per-frame offset", flush=True)

            if crop_mode == "fixed_size":
                crop, frame_crop_frac = per_frame_windows[j], crop_frac
            elif crop_mode == "per_frame":
                inside = [inside_mask(per_cond[cond][j]) for cond in CONDITIONS
                          if cond != "real"]
                mask = np.logical_and.reduce(inside)
                width_frac = centred_crop_fraction(mask)
                hh = int(round(width_frac * h)) // 2
                ww = int(round(width_frac * w)) // 2
                cy, cx = int(round((h - 1) / 2.0)), int(round((w - 1) / 2.0))
                crop = (slice(cy - hh, cy + hh), slice(cx - ww, cx + ww))
                frame_crop_frac = width_frac
            else:
                crop, frame_crop_frac = crops, crop_frac
            for cond in CONDITIONS:
                if cond == "real":
                    out_img, valid = img, np.ones(img.shape, bool)
                else:
                    out_img, valid = warp(img, per_cond[cond][j], geometry, calib)
                out_img = out_img[crop]
                valid = valid[crop]
                cv2.imwrite(str(frames_out[cond] / f"{j:05d}.png"), out_img)
                valids[cond][j] = float(valid.mean())
                if valid.mean() < 0.5:
                    continue
                g, c = _run(model, cm, out_img, device)
                if g is None or not np.all(np.isfinite(g)) or np.linalg.norm(g) < 1e-6:
                    continue
                ups[cond][j] = g / np.linalg.norm(g)
                confs[cond][j] = c

        for cond in CONDITIONS:
            up_b = np.einsum("ij,nj->ni", R_cb, ups[cond])
            name = f"{seq}_{ci:02d}_{cond}"
            np.savez_compressed(out_root / f"{name}.npz", t=t_clip,
                                pos=truth.pos[lo:hi], yaw=yaw[lo:hi],
                                depicted_lean=lean_from_up_body(up_b),
                                depicted_conf=confs[cond], valid_frac=valids[cond])
            (out_root / f"{name}.meta.json").write_text(json.dumps({
                "dataset": "uzh_fpv", "sequence": seq, "clip_id": name, "condition": cond,
                "index": [lo, hi], "fps": fps, "n_frames": int(hi - lo),
                "mount_source": "published",
                "mount_rotation_R_cb": R_cb.tolist(),
                "mount_rotation_T_cam_imu": meta["kalibr_imu_cam0"]["T_cam_imu"],
                "estimator": "geocalib", "estimator_commit": GEOCALIB_COMMIT,
                "estimator_variant": variant, "estimator_camera_model": cm,
                "depicted_conf_meaning": "mean of GeoCalib's latitude_confidence map",
                "truth_frames": ("Leica MS60 total station, IMU frame of the Snapdragon "
                                 "Flight in a gravity-aligned world frame, nearest sample at "
                                 "the frame timestamp"),
                "clock_offset_applied_s": truth.offset_s, "k_d_path": 0.0,
                "yaw_control_deg": yaw_deg if cond == "yaw_control" else None,
                "testability": sig,
                "border": ("one fixed centred crop per clip, inside the valid region of every "
                           "frame and every condition, applied to all conditions including real; "
                           "no black fill" if crop_mode == "fixed" else
                           "per-frame centred crop (E02 behaviour, kept for comparison)"),
                "crop_mode": crop_mode,
                "crop_fallback_no_shared_window": bool(fallback),
                "crop_fraction": float(crop_frac) if crop_mode in ("fixed", "fixed_size") else None,
                      "crop_fallback_no_shared_window": bool(fallback),
                "frames_dir": str(frames_out[cond])}, indent=1))
        index.append({"clip": f"{seq}#{ci:02d}", "index": [lo, hi], "testable": True,
                      "valid_frac_median": float(np.nanmedian(valids["tilt_removal"])),
                      "crop_fraction": float(crop_frac) if crop_mode in ("fixed", "fixed_size") else None,
                      "crop_fallback_no_shared_window": bool(fallback),
                      "valid_frac_real_median": float(np.nanmedian(valids["real"])),
                      "conditions": list(CONDITIONS)})
        e_data.commit()

    (out_root / "index.json").write_text(json.dumps(
        {"sequence": seq, "variant": variant, "fps": fps, "clip_seconds": clip_seconds,
         "clips": index}, indent=1))
    e_data.commit()
    rec = {"seq": seq, "variant": variant, "n_bounds": len(bounds), "n_exported": kept,
           "valid_frac_median": float(np.nanmedian([c.get("valid_frac_median", np.nan)
                                                    for c in index]))}
    print(json.dumps(rec), flush=True)
    return json.dumps(rec)



@app.local_entrypoint()
def main(seqs: str = "", variant: str = "b", max_clips: int = 4, validate: bool = False,
         clip_seconds: float = 4.0, parallel: bool = True,
         crop_mode: str = "fixed") -> None:
    import json

    from src.depicted.uzh import all_sequences

    wanted = [s for s in seqs.split(",") if s] or all_sequences()
    fn = validate_edits if validate else export_sequence
    args = (variant,) if validate else (variant, clip_seconds, max_clips, 20.0, crop_mode)
    if parallel:
        handles = [(s, fn.spawn(s, *args)) for s in wanted]
        for s, h in handles:
            try:
                print(json.loads(h.get()), flush=True)
            except Exception as exc:
                print(f"FAILED {s}: {exc}", flush=True)
    else:
        for s in wanted:
            print(json.loads(fn.remote(s, *args)), flush=True)
