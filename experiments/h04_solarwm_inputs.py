"""H04: build the SolarWM-5B camera-pose inputs for the continuous-attitude campaign.

Follows ``experiments/h02_solarwm_inputs.py`` exactly in format (image, absolute OpenCV c2w
[81,4,4], prompt, meta) and differs only in the population: 22 continuous-attitude paths from
``experiments/h04_paths.py``, three arms, three draws, six first frames, plus the two
convention probes and the exploratory scene swap.  See
``configs/h04_solarwm_continuous_prereg.yaml`` and its amendment 1.

    pixi run -e gpu python experiments/h04_solarwm_inputs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.generation.attitude import body_rotation
from src.perception.frames import camera_to_body_pitched

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/h04_solarwm_inputs"
SHARED = Path("/home/kevin/Documents/Projects/aerial-world-model-verifier")
PATHS_NPZ = ROOT / "results/h04/paths/paths.npz"
PAD_W, PAD_H = 864, 480
N = 81
ARMS = ("A_flatness", "B_level", "C_mirrored_roll")
N_GEN = 3
PROMPT = ("Aerial drone footage over a sunlit city with a river, tall buildings and a "
          "hazy horizon, photorealistic, camera moving over the city.")

SWAP_PATHS = ["arc_10", "s_10", "slalom_10", "fig8_10"]
SWAP_FRAMES = ["F4", "F5", "F6"]


def _first_frame_anwm(idx: int) -> Path:
    """The idx-th ANWM release trajectory's smallest recorded frame (H02's rule)."""
    dirs = sorted(p for p in (SHARED / "data/raw/anwm_frames").iterdir() if p.is_dir())
    jpgs = sorted(dirs[idx].glob("*.jpg"), key=lambda p: int(p.stem))
    return jpgs[0]


def _first_frame_track_i(clip: str) -> Path:
    src = Path("/tmp/h3data/i") / clip / f"{clip}.mp4"
    import cv2
    cap = cv2.VideoCapture(str(src))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read first frame of {src}")
    return fr


def prepare_image(img, dst: Path) -> None:
    """Scale to height 480 keeping the aspect ratio, then edge-pad to 864x480."""
    import cv2
    if not isinstance(img, np.ndarray):
        img = np.asarray(Image.open(img).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    nw = int(round(w * PAD_H / h))
    im = cv2.resize(img, (nw, PAD_H), interpolation=cv2.INTER_CUBIC)
    if nw > PAD_W:
        im = cv2.resize(img, (PAD_W, int(round(h * PAD_W / w))), interpolation=cv2.INTER_CUBIC)
        nw = PAD_W
        im = cv2.resize(im, (PAD_W, PAD_H), interpolation=cv2.INTER_CUBIC)
        nw = PAD_W
    left = (PAD_W - nw) // 2
    right = PAD_W - nw - left
    im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_REPLICATE)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im)


def poses_for(arm: str, yaw, roll, pitch) -> np.ndarray:
    R_bc = camera_to_body_pitched(0.0)
    phi = np.zeros_like(roll) if arm == "B_level" else (
        -roll if arm == "C_mirrored_roll" else roll)
    th = np.zeros_like(pitch) if arm == "B_level" else pitch
    ps = np.tile(np.eye(4), (N, 1, 1))
    ps[:, :3, :3] = body_rotation(yaw, th, phi) @ R_bc[None]
    return ps


def main() -> int:
    d = np.load(PATHS_NPZ)
    names = sorted({k.split("__")[0] for k in d.files})
    order = [n for n in names]
    summary = json.loads((PATHS_NPZ.parent / "paths_summary.json").read_text())
    order = [s["path"] for s in summary]
    frames = {
        "F1": ("anwm_city", _first_frame_anwm(0)),
        "F2": ("anwm_city", _first_frame_anwm(1)),
        "F3": ("anwm_city", _first_frame_anwm(2)),
        "F4": ("urban_checker", _first_frame_track_i("i02_dev_004")),
        "F5": ("industrial", _first_frame_track_i("i02_dev_000")),
        "F6": ("forest_ridge", _first_frame_track_i("i02_dev_002")),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"samples": [], "frames": {}, "n_out": N}
    for k, (scene, src) in frames.items():
        dst = OUT / "frames" / f"{k}.png"
        prepare_image(src, dst)
        manifest["frames"][k] = {"scene": scene, "path": str(dst.relative_to(ROOT))}

    def write(sid: str, frame_key: str, poses: np.ndarray, meta: dict) -> None:
        s = OUT / sid
        s.mkdir(parents=True, exist_ok=True)
        Image.open(OUT / "frames" / f"{frame_key}.png").save(s / "image.png")
        np.save(s / "poses.npy", poses.astype(np.float32))
        (s / "prompt.txt").write_text(PROMPT)
        meta = {**meta, "frame_key": frame_key,
                "scene": frames[frame_key][0]}
        (s / "meta.json").write_text(json.dumps(meta, indent=1))
        manifest["samples"].append(sid)

    # main grid: 22 paths x 3 arms x 3 draws
    for i, name in enumerate(order):
        fk = f"F{1 + (i % 6)}"
        yaw, roll, pitch = (d[f"{name}__yaw"], d[f"{name}__roll"], d[f"{name}__pitch"])
        pos = d[f"{name}__pos"]
        for arm in ARMS:
            ps = poses_for(arm, yaw, roll, pitch)
            ps[:, :3, 3] = pos
            for g in range(N_GEN):
                write(f"main_{name}_{arm}_g{g}", fk, ps,
                      {"kind": "main", "path": name, "arm": arm, "draw": g,
                       "path_index": i})
    # exploratory scene swap: 4 paths x 3 scenes x 3 arms x 1 draw
    for name in SWAP_PATHS:
        yaw, roll, pitch = (d[f"{name}__yaw"], d[f"{name}__roll"], d[f"{name}__pitch"])
        pos = d[f"{name}__pos"]
        for fk in SWAP_FRAMES:
            for arm in ARMS:
                ps = poses_for(arm, yaw, roll, pitch)
                ps[:, :3, 3] = pos
                write(f"swap_{name}_{fk}_{arm}_g0", fk, ps,
                      {"kind": "scene_swap", "path": name, "arm": arm, "draw": 0})
    # convention probes: pure roll ramps on a straight level path, frame F1
    base = poses_for("B_level", d["arc_10__yaw"] * 0.0, d["arc_10__roll"] * 0.0,
                     d["arc_10__pitch"] * 0.0)
    for tag, peak in (("p20", 20.0), ("m20", -20.0)):
        ps = base.copy()
        for k in range(N):
            ang = np.deg2rad(peak * k / (N - 1))
            Rz = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                           [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]])
            ps[k, :3, :3] = base[k, :3, :3] @ Rz
        write(f"probe_{tag}", "F1", ps,
              {"kind": "convention_probe", "peak_deg": peak})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    from collections import Counter
    kinds = Counter(s.split("_", 1)[0] for s in manifest["samples"])
    print(json.dumps({"samples": len(manifest["samples"]), "kinds": kinds,
                      "paths": len(order)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
