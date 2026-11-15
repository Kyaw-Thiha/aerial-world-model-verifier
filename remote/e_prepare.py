"""Track E: prepare UZH-FPV sequences for the depicted-tilt study (Task 1-2).

Why this exists rather than reusing `remote/c_prepare.py`: Track C saved only the
undistorted pinhole view, capped at 150 frames, and kept neither the raw fisheye frames
(needed for the `simple_divisional` variant that is best on TII) nor the per-frame
timestamps or the full-resolution ground-truth table (needed for the Task 1 audit and the
time-offset estimate). Nothing here changes Track C's artifacts; this writes its own tree
under `track-e/` on a separate volume.

Per sequence it writes, under ``/e-data/uzh/<seq>/``:
  frames_raw/xxxxx.png      raw KB4 fisheye, full 640x480, 8-bit grey   (variant b input)
  frames_pinhole/xxxxx.png  cv2.fisheye.undistortImage, same focal, 640x480 (variants a,c)
  truth.npz                 t (s, relative), pos (N,3), quat (N,4 xyzw), t_unix (N,)
                            and the *raw* gt table `gt` (M,8) + its clock
  meta.json                 intrinsics (KB4 + pinhole), stride, frame names, timestamps,
                            the Kalibr cam-IMU chain, and provenance

Truth semantics (Task 1 audit, established from the ICRA19 paper Sec. III-E): the Leica
MS60 tracks a prism; the released pose is the state (q, p, v) of the **Snapdragon IMU
frame** in the Leica world frame, which the paper describes as gravity-aligned. The audit
in `experiments/e_uzh_audit.py` checks the z direction and the quaternion convention
empirically instead of trusting this note.

    pixi run modal run remote/e_prepare.py --seqs indoor_forward_5 --max-frames 450
"""
from __future__ import annotations

import modal

app = modal.App("awmv-e-prepare")

e_data = modal.Volume.from_name("awmv-e-data", create_if_missing=True)     # ours, writable
c_data = modal.Volume.from_name("awmv-c-data", create_if_missing=True)     # Track C, read-only use

E_DATA = "/e-data"
C_DATA = "/c-data"

UZH_BASE = "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv-newer-versions/v3"
UZH_CALIB = "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv/calib"

# name -> (zip stem, rig, environment, published vmax m/s, notes)
SEQUENCES = {
    "indoor_forward_3": ("indoor_forward_3_snapdragon_with_gt", "indoor_forward", "indoor", 9.50, "easy"),
    "indoor_forward_5": ("indoor_forward_5_snapdragon_with_gt", "indoor_forward", "indoor", 4.87, "medium"),
    "indoor_forward_6": ("indoor_forward_6_snapdragon_with_gt", "indoor_forward", "indoor", 12.52, "medium"),
    "indoor_forward_7": ("indoor_forward_7_snapdragon_with_gt", "indoor_forward", "indoor", 12.78, "hard"),
    "indoor_forward_9": ("indoor_forward_9_snapdragon_with_gt", "indoor_forward", "indoor", 11.42, "easy"),
    "indoor_forward_10": ("indoor_forward_10_snapdragon_with_gt", "indoor_forward", "indoor", 9.49, "easy"),
    "outdoor_forward_1": ("outdoor_forward_1_snapdragon_with_gt", "outdoor_forward", "outdoor", 8.55, "easy"),
    "outdoor_forward_3": ("outdoor_forward_3_snapdragon_with_gt", "outdoor_forward", "outdoor", 14.04, "medium"),
    "outdoor_forward_5": ("outdoor_forward_5_snapdragon_with_gt", "outdoor_forward", "outdoor", 20.73, "hard"),
    "indoor_45_2": ("indoor_45_2_snapdragon_with_gt", "indoor_45", "indoor", 6.97, "easy"),
    "indoor_45_4": ("indoor_45_4_snapdragon_with_gt", "indoor_45", "indoor", 6.55, "easy"),
    "indoor_45_9": ("indoor_45_9_snapdragon_with_gt", "indoor_45", "indoor", 11.23, "medium"),
    "indoor_45_13": ("indoor_45_13_snapdragon_with_gt", "indoor_45", "indoor", 7.92, "medium"),
    "outdoor_45_1": ("outdoor_45_1_snapdragon_with_gt", "outdoor_45", "outdoor", 15.62, "medium"),
}

CALIB_FOR_RIG = {
    "indoor_forward": "indoor_forward_calib_snapdragon.zip",
    "indoor_45": "indoor_45_calib_snapdragon.zip",
    "outdoor_forward": "outdoor_forward_calib_snapdragon.zip",
    "outdoor_45": "outdoor_45_calib_snapdragon.zip",
}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "scipy", "opencv-python-headless", "pyyaml")
)


def _fetch(url: str, dst) -> None:
    import urllib.request
    from pathlib import Path

    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dst)


def _local_or_fetch(zip_name: str, remote_url: str, cache):
    """Return a container-local path to ``zip_name``.

    Random reads inside a 1-2 GB zip on a Volume mount are ~10x slower than sequential reads
    on local disk (Track A measured this on TII), so the zip is always used from /tmp. The
    persistent cache still gets a copy of anything newly downloaded, and Track C's volume is
    used in place of a download. Volume-to-volume copies are avoided: they cost minutes.
    """
    import shutil
    import urllib.request
    from pathlib import Path

    local = Path("/tmp") / zip_name
    if local.exists() and local.stat().st_size > 0:
        return local
    for cand in (Path(cache), Path(C_DATA) / "zips"):
        p = cand / zip_name
        if p.exists() and p.stat().st_size > 0:
            shutil.copyfile(p, local)
            return local
    tmp = Path(cache) / (zip_name + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(remote_url, timeout=900) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(Path(cache) / zip_name)
    shutil.copyfile(Path(cache) / zip_name, local)
    return local


def _read_yaml_from_zip(path, needle: str, exclude: str = "") -> dict:
    import zipfile

    import yaml

    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist()
                    if needle in n and n.endswith((".yaml", ".yml"))
                    and (not exclude or exclude not in n))
        return yaml.safe_load(z.read(name)), name


def _gt_rows(text: str):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= 8:
            rows.append(v[:8])
    return rows


@app.function(image=image, volumes={E_DATA: e_data, C_DATA: c_data}, timeout=3600, cpu=4,
              memory=8192)
def prepare(seq: str, stride: int = 2, max_frames: int = 450,
            offset_s: float = 0.0) -> str:
    """Extract raw + pinhole frames inside the truth interval, with truth and provenance."""
    import json
    import zipfile
    from pathlib import Path

    import cv2
    import numpy as np

    stem, rig, env, vmax, note = SEQUENCES[seq]
    zdir = Path(E_DATA) / "zips"
    zpath = _local_or_fetch(f"{stem}.zip", f"{UZH_BASE}/{stem}.zip", zdir)
    cpath = _local_or_fetch(CALIB_FOR_RIG[rig], f"{UZH_CALIB}/{CALIB_FOR_RIG[rig]}", zdir)

    out = Path(E_DATA) / "uzh" / seq
    (out / "frames_raw").mkdir(parents=True, exist_ok=True)
    (out / "frames_pinhole").mkdir(parents=True, exist_ok=True)

    cam_chain, cam_name = _read_yaml_from_zip(cpath, "camchain-", exclude="imucam")
    imu_chain, imu_name = _read_yaml_from_zip(cpath, "camchain-imucam")
    cam0 = cam_chain["cam0"]
    fx, fy, cx, cy = (float(v) for v in cam0["intrinsics"])
    dist = [float(v) for v in cam0["distortion_coeffs"]]
    w_src, h_src = (int(v) for v in cam0["resolution"])

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        gt_text = z.read("groundtruth.txt").decode()
        gt = np.asarray(_gt_rows(gt_text), dtype=np.float64)
        gt = gt[np.argsort(gt[:, 0])]
        img_list = next(n for n in names if Path(n).name == "left_images.txt")
        rows = []
        for line in z.read(img_list).decode().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 3:
                rows.append((float(p[1]), p[2]))
        rows.sort(key=lambda r: r[0])
        lo, hi = gt[0, 0], gt[-1, 0]
        # frame time on the truth clock = image timestamp + measured clock offset
        rows = [r for r in rows if lo <= r[0] + offset_s <= hi]
        n_in_truth = len(rows)
        rows = rows[::stride][:max_frames]
        if len(rows) < 3:
            raise ValueError(f"{seq}: only {len(rows)} frames inside the truth interval")

        ts_unix = np.array([r[0] for r in rows], dtype=np.float64)
        t = ts_unix - ts_unix[0]
        K_in = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        K_out = np.array([[fx, 0, w_src / 2.0], [0, fy, h_src / 2.0], [0, 0, 1.0]])
        d4 = np.asarray(dist, dtype=np.float64)
        names_out, shapes = [], None
        for i, (ts, name) in enumerate(rows):
            img = cv2.imdecode(np.frombuffer(z.read(name), dtype=np.uint8),
                               cv2.IMREAD_GRAYSCALE)
            if shapes is None:
                shapes = list(img.shape)
            cv2.imwrite(str(out / "frames_raw" / f"{i:05d}.png"), img)
            und = cv2.fisheye.undistortImage(img, K_in, d4, Knew=K_out,
                                            new_size=(w_src, h_src))
            cv2.imwrite(str(out / "frames_pinhole" / f"{i:05d}.png"), und)
            names_out.append(name)

    # truth sampled at the frame timestamps (nearest; the tracker is far faster than 30 Hz)
    t_gt = gt[:, 0] - (ts_unix[0] + offset_s)
    idx = np.searchsorted(t_gt, t).clip(1, len(t_gt) - 1)
    left = np.abs(t_gt[idx - 1] - t) < np.abs(t_gt[idx] - t)
    idx = np.where(left, idx - 1, idx)
    pos = gt[idx, 1:4]
    quat = gt[idx, 4:8]
    dt_gt = float(np.median(np.diff(gt[:, 0])))

    np.savez_compressed(out / "truth.npz", t=t, pos=pos, quat=quat, t_unix=ts_unix,
                        gt=gt, gt_t0=float(ts_unix[0]))
    meta = {
        "sequence": seq, "zip_stem": stem, "rig": rig, "environment": env,
        "published_vmax_mps": vmax, "published_difficulty": note,
        "n": int(len(t)), "stride": stride, "max_frames": max_frames,
        "n_frames_inside_truth": int(n_in_truth),
        "fps_measured": float(1.0 / np.median(np.diff(t))),
        "image_hw": shapes, "image_dtype": "uint8 grey",
        "intrinsics_kb4": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "k": dist,
                           "model": "pinhole+equidistant(KB4)", "w": w_src, "h": h_src},
        "intrinsics_pinhole": {"fx": fx, "fy": fy, "cx": w_src / 2.0, "cy": h_src / 2.0,
                               "w": w_src, "h": h_src, "model": "pinhole",
                               "note": "cv2.fisheye.undistortImage, same focal, centre crop"},
        "kalibr_cam0": cam0,
        "kalibr_imu_cam0": imu_chain["cam0"],
        "kalibr_files": {"cam": cam_name, "imucam": imu_name},
        "clock_offset_applied_s": float(offset_s),
        "gt_format": "TUM: timestamp tx ty tz qx qy qz qw (xyzw), Leica MS60 world frame",
        "gt_rows": int(len(gt)), "gt_dt_s": dt_gt, "gt_span_s": float(gt[-1, 0] - gt[0, 0]),
        "frame_names": names_out, "t_unix": ts_unix.tolist(),
        "license": "CC BY-NC-SA 3.0 (UZH-FPV dataset page)",
    }
    (out / "meta.json").write_text(json.dumps(meta))
    e_data.commit()
    rec = {k: meta[k] for k in ("sequence", "n", "fps_measured", "rig", "gt_rows",
                                "gt_dt_s", "gt_span_s", "n_frames_inside_truth")}
    print(json.dumps(rec), flush=True)
    return json.dumps(rec)


@app.function(image=image, volumes={E_DATA: e_data, C_DATA: c_data}, timeout=900, cpu=2)
def probe(seq: str) -> str:
    """Report the structure of one sequence zip without extracting frames."""
    import json
    import zipfile
    from pathlib import Path

    import numpy as np

    stem = SEQUENCES[seq][0]
    zdir = Path(E_DATA) / "zips"
    zpath = _local_or_fetch(f"{stem}.zip", f"{UZH_BASE}/{stem}.zip", zdir)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        gt_text = z.read("groundtruth.txt").decode()
        gt = np.asarray(_gt_rows(gt_text), dtype=np.float64)
        list_name = next(n for n in names if Path(n).name == "left_images.txt")
        lines = [l for l in z.read(list_name).decode().splitlines() if l.strip()]
        imgs = [n for n in names if n.lower().endswith((".png", ".jpg"))]
        rec = {
            "seq": seq, "zip": str(zpath), "n_members": len(names),
            "members_top": sorted({n.split("/")[0] for n in names})[:10],
            "n_image_members": len(imgs), "first_image_member": imgs[0] if imgs else None,
            "image_list_file": list_name, "image_list_head": lines[:4],
            "n_image_list_rows": len(lines),
            "gt_head": gt_text.splitlines()[:4], "n_gt_rows": int(len(gt)),
            "gt_dt_median_s": float(np.median(np.diff(np.sort(gt[:, 0])))),
            "gt_span_s": float(gt[:, 0].max() - gt[:, 0].min()),
            "image_list_span_s": None,
        }
        ts = []
        for line in lines:
            p = line.split()
            if not line.startswith("#") and len(p) >= 3:
                try:
                    ts.append(float(p[1]))
                except ValueError:
                    pass
        if ts:
            rec["image_list_span_s"] = float(max(ts) - min(ts))
            rec["image_list_first_unix"] = min(ts)
        rec["gt_first_unix"] = float(gt[:, 0].min())
        rec["gt_last_unix"] = float(gt[:, 0].max())
        rec["image_list_last_unix"] = max(ts) if ts else None
    Path(E_DATA, "probe").mkdir(parents=True, exist_ok=True)
    (Path(E_DATA) / "probe" / f"{seq}.json").write_text(json.dumps(rec, indent=1))
    e_data.commit()
    return json.dumps(rec)


@app.local_entrypoint()
def main(seqs: str = "", stride: int = 2, max_frames: int = 450, do_probe: bool = False,
         offsets: str = "", parallel: bool = True) -> None:
    """``offsets`` is ``seq=seconds,...`` for sequences whose clock offset is measured."""
    import json

    want = {kv.split("=")[0]: float(kv.split("=")[1])
            for kv in offsets.split(",") if "=" in kv}
    wanted = [s for s in seqs.split(",") if s] or list(SEQUENCES)
    if do_probe:
        for s in wanted:
            print("probe", probe.remote(s), flush=True)
        return
    if parallel:
        handles = [(s, prepare.spawn(s, stride, max_frames, want.get(s, 0.0))) for s in wanted]
        for s, h in handles:
            try:
                print("prepare", json.loads(h.get()), flush=True)
            except Exception as exc:
                print(f"FAILED {s}: {exc}", flush=True)
    else:
        for s in wanted:
            print("prepare", prepare.remote(s, stride, max_frames, want.get(s, 0.0)), flush=True)
