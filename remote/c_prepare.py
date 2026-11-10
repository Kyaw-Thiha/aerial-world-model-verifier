"""Track C, Task 2: prepare the known-truth testbeds inside Modal.

Frames and undistortion need OpenCV, which the local analysis environment does not have,
so preparation runs here and only the small artifacts (truth arrays, metrics) come back.

Volumes
-------
``awmv-c-data``   prepared clips under ``/data/uzh/<seq>/`` and the raw zips under
                  ``/data/zips/``.
``awmv-a-data``   Track A's data, read-only: TII video zips and ANWM clips.

    pixi run modal run remote/c_prepare.py --seq indoor_forward_6 --stride 2 --max-frames 150
    pixi run modal run remote/c_prepare.py --all
"""
from __future__ import annotations

import modal

app = modal.App("awmv-c-prepare")

c_data = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
a_data = modal.Volume.from_name("awmv-a-data")

C_DATA = "/data"
A_DATA = "/a-data"
UZH_BASE = "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv-newer-versions/v3"
UZH_CALIB_URL = "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv/calib"
SEQUENCES = {
    # name: (zip stem, published vmax m/s, ground truth public)
    "indoor_forward_3": ("indoor_forward_3_snapdragon_with_gt", 9.5, True),
    "indoor_forward_6": ("indoor_forward_6_snapdragon_with_gt", 12.52, True),
    "indoor_forward_5": ("indoor_forward_5_snapdragon_with_gt", 4.87, True),
    "indoor_45_9": ("indoor_45_9_snapdragon_with_gt", 11.23, True),
    "outdoor_forward_5": ("outdoor_forward_5_snapdragon_with_gt", 20.73, True),
    "indoor_forward_7": ("indoor_forward_7_snapdragon_with_gt", 12.78, True),
    "indoor_forward_9": ("indoor_forward_9_snapdragon_with_gt", 11.42, True),
    "indoor_forward_10": ("indoor_forward_10_snapdragon_with_gt", 9.49, True),
    "outdoor_forward_1": ("outdoor_forward_1_snapdragon_with_gt", 8.55, True),
    "outdoor_forward_3": ("outdoor_forward_3_snapdragon_with_gt", 14.04, True),
}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "scipy", "opencv-python-headless", "pyyaml", "pandas")
    .add_local_python_source("src")
)


def _fetch(url: str, dst) -> None:
    import urllib.request
    from pathlib import Path
    dst = Path(dst)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dst)


@app.function(image=image, volumes={C_DATA: c_data, A_DATA: a_data}, timeout=3600, cpu=4,
              memory=8192)
def prepare_uzh(seq: str, stride: int = 2, max_frames: int = 150, keep_zip: bool = True) -> str:
    """Undistort one UZH sequence, save frames + truth, and report its kinematics."""
    import json
    import shutil
    from pathlib import Path

    import numpy as np

    from src.pathrec.lean import differentiate, lean_xy, tilt_deg
    from src.pathrec.testbeds import prepare_uzh as prepare

    stem, vmax_pub, _ = SEQUENCES[seq]
    zdir = Path(C_DATA) / "zips"
    zpath = zdir / f"{stem}.zip"
    _fetch(f"{UZH_BASE}/{stem}.zip", zpath)
    calib = zdir / "indoor_forward_calib_snapdragon.zip"
    _fetch(f"{UZH_CALIB_URL}/indoor_forward_calib_snapdragon.zip", calib)

    out = Path(C_DATA) / "uzh" / seq
    if out.exists():
        shutil.rmtree(out)
    clip = prepare(zpath, calib, out, stride=stride, max_frames=max_frames)

    # truth kinematics on the same time base the extractors will see
    t, v, a = differentiate(clip.pos, dt=1.0 / clip.fps, window_s=0.2)
    speed = np.linalg.norm(v, axis=1)
    acc_h = np.linalg.norm(a[:, :2], axis=1)
    tilt = tilt_deg(a, v, k_d=0.295)
    lean = lean_xy(a)
    rec = {
        "seq": seq, "n": len(clip.t), "fps": clip.fps, "duration_s": clip.duration_s,
        "stride": stride, "intrinsics": clip.intrinsics,
        "vmax_published_mps": vmax_pub,
        "speed_median_mps": float(np.median(speed)), "speed_p95_mps": float(np.percentile(speed, 95)),
        "speed_max_mps": float(speed.max()),
        "accel_h_median_mps2": float(np.median(acc_h)), "accel_h_p95_mps2": float(np.percentile(acc_h, 95)),
        "tilt_median_deg": float(np.median(tilt)), "tilt_p90_deg": float(np.percentile(tilt, 90)),
        "frac_tilt_ge_10deg": float((tilt >= 10).mean()),
        "frac_tilt_ge_25deg": float((tilt >= 25).mean()),
        "lean_magnitude_median": float(np.nanmedian(np.linalg.norm(lean, axis=1))),
        "pose_semantics": "Leica ground truth (camera frame as published), TUM xyzw",
        "world_axes": "as published (not re-oriented here); orientation checked downstream",
    }
    (out / "summary.json").write_text(json.dumps(rec, indent=1))
    c_data.commit()
    if not keep_zip:
        zpath.unlink(missing_ok=True)
        c_data.commit()
    print(json.dumps(rec, indent=1), flush=True)
    return json.dumps(rec)


@app.function(image=image, volumes={C_DATA: c_data, A_DATA: a_data}, timeout=3600, cpu=2)
def copy_anwm() -> str:
    """Mirror the 82 ANWM clips (frames, gt, manifest) into the Track C volume."""
    import json
    import shutil
    from pathlib import Path

    src = Path(A_DATA) / "anwm_clips"
    dst = Path(C_DATA) / "anwm"
    n = 0
    for man in sorted(src.rglob("manifest.json")):
        rel = man.parent.relative_to(src)
        target = dst / rel
        target.mkdir(parents=True, exist_ok=True)
        for sub in ("frames", "gt"):
            if (man.parent / sub).is_dir():
                if (target / sub).exists():
                    shutil.rmtree(target / sub)
                shutil.copytree(man.parent / sub, target / sub)
        shutil.copy2(man, target / "manifest.json")
        n += 1
    c_data.commit()
    return json.dumps({"clips": n, "dest": str(dst)})


@app.local_entrypoint()
def main(seq: str = "", all: bool = False, stride: int = 2, max_frames: int = 150,
         anwm: bool = False) -> None:
    import json as _json

    if anwm:
        print(copy_anwm.remote())
    todo = list(SEQUENCES) if all else ([seq] if seq else [])
    for s in todo:
        out = prepare_uzh.remote(s, stride=stride, max_frames=max_frames)
        rec = _json.loads(out)
        print(f"== {s}: n={rec['n']} fps={rec['fps']:.1f} "
              f"v_med={rec['speed_median_mps']:.2f} v_max={rec['speed_max_mps']:.2f} "
              f"a_h_p95={rec['accel_h_p95_mps2']:.2f} tilt>=10deg={rec['frac_tilt_ge_10deg']:.2f}")
