"""Preview and geometry check of the edit conditions on real TII frames.

Runs the same warp the GPU worker runs, on the locally available sampled frames, and
writes a contact sheet plus the valid-fraction statistics. This is a visual and
geometric check of the edit pipeline: the tilt edits must move the horizon/ceiling
band, the yaw control must keep the band's angle and lose the same borders, and the
identity must be pixel-exact.

Output: results/phase3a_verdict/<run_id>_edit_preview/{sheet_<flight>.png, valid_frac.json}
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.tii import load_tii
from src.datasets.tii_video import frame_table
from src.negatives import equidistant_f, remap_arrays, warp_with_remap
from src.negatives.edits import condition_maps
from src.perception.frames import camera_to_body_pitched

ROOT = Path(__file__).resolve().parents[1]
CONDS = [("identity", {}), ("tilt_k0", {"k": 0.0}), ("tilt_k05", {"k": 0.5}),
         ("tilt_k15", {"k": 1.5}), ("mirror_roll", {}), ("yaw_control", {})]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    import cv2
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default="data/raw/tii_frames")
    ap.add_argument("--specs", default="data/processed/p3a/p3a_v1")
    ap.add_argument("--out", default="results/phase3a_verdict")
    ap.add_argument("--flights", default="flight-01p-ellipse,flight-13p-trackRATM")
    args = ap.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / f"{run_id}_edit_preview"
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    for flight in [f for f in args.flights.split(",") if f]:
        spec_path = ROOT / args.specs / f"{flight}.npz"
        spec = np.load(spec_path, allow_pickle=False)
        names = [str(n) for n in spec["names"]]
        # the sampled development frames use the same "<index>_<timestamp>.jpg" names
        pool = {Path(p).name: p for f in glob.glob(str(ROOT / args.frames_root / flight / "*.jpg"))
                for p in [f]}
        take = [n for n in names if n in pool][:6]
        if not take:
            print(f"{flight}: no sampled frames match the spec names, skipped")
            continue
        R_bw = Rotation.from_quat(spec["quat"]).as_matrix()
        idx = [names.index(n) for n in take]
        R_cb = camera_to_body_pitched(np.deg2rad(float(spec["uptilt_deg"])))
        R_cw = np.einsum("nij,jk->nik", R_bw[idx], R_cb)
        img0 = cv2.imread(pool[take[0]])
        h, w = img0.shape[:2]
        fpx = equidistant_f(w, h, 175.0)

        fig, axes = plt.subplots(3, 6, figsize=(15, 8))
        for c, (cond, kw) in enumerate(CONDS):
            if cond.startswith("tilt_k"):
                M = condition_maps("tilt_k", R_bw[idx], R_cw, k=kw["k"])
            elif cond == "mirror_roll":
                M = condition_maps("mirror_roll", R_bw[idx], R_cw)
            elif cond == "yaw_control":
                from src.negatives.edits import rotation_angle_deg
                mag = np.minimum(rotation_angle_deg(condition_maps("tilt_k", R_bw[idx], R_cw, 0.0)), 60.0)
                M = condition_maps("yaw_control", R_bw[idx], R_cw, yaw_rot_rad=np.deg2rad(mag))
            else:
                M = condition_maps("identity", R_bw[idx], R_cw)
            vf, ang = [], []
            for r in range(3):
                img = cv2.imread(pool[take[r]])
                mx, my = remap_arrays(M[r], w, h, fpx)
                warped, valid = warp_with_remap(img, mx, my)
                vf.append(float(valid.mean()))
                ang.append(float(np.degrees(Rotation.from_matrix(M[r]).magnitude())))
                axes[r, c].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                axes[r, c].set_title(f"{cond}\n{ang[-1]:.0f} deg, {100*vf[-1]:.0f}% valid",
                                     fontsize=7)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
            stats.setdefault(flight, {})[cond] = {"valid_frac": vf, "angle_deg": ang}
        fig.suptitle(f"{flight}: edit conditions, same three frames", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"sheet_{flight}.png", dpi=120)
        plt.close(fig)
        ident = stats[flight]["identity"]
        assert all(v > 0.999 for v in ident["valid_frac"]), "identity must keep every pixel"
        assert all(a == 0.0 for a in ident["angle_deg"])
        print(f"{flight}: " + ", ".join(
            f"{c} {np.mean(v['valid_frac']):.2f}/{np.mean(v['angle_deg']):.0f}deg"
            for c, v in stats[flight].items()))

    (out / "valid_frac.json").write_text(json.dumps(
        {"git": git_hash(), "note": "identity must be pixel-exact (valid 1.0, angle 0); the "
                                    "yaw control loses the same border as the tilt edit by "
                                    "construction", "flights": stats}, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
