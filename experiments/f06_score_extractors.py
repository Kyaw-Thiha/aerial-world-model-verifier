"""Track F2, part 3: score the new feed-forward extractors against truth (CPU).

Writes ``results/f06_feedforward/summary.json``: the preflight record (licence, gating, code,
intrinsics input, run/not-run) and Track C's bake-off metrics for MapAnything and Pi3X on the
ten UZH sequences.

    python experiments/f06_score_extractors.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.pathrec.evaluate import score_clip

SEQS = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7",
        "indoor_forward_9", "indoor_forward_10", "outdoor_forward_1", "outdoor_forward_3",
        "outdoor_forward_5", "indoor_45_9")
PATHS = ("mapanything", "pi3x")
PREFLIGHT = {
    "mapanything": {
        "code": "https://github.com/facebookresearch/map-anything (Apache-2.0)",
        "weights": "facebook/map-anything-apache (Apache-2.0); facebook/map-anything "
                   "(CC BY-NC 4.0)",
        "gated": False,
        "intrinsics_input": "documented per-view key, but broken in the released code: "
                            "IndexError in _encode_and_fuse_ray_dirs when a 640x480 K is "
                            "combined with load_images' 392x518 resize",
        "run": "image-only, estimated intrinsics",
    },
    "pi3x": {
        "code": "https://github.com/yyfz/Pi3 (BSD-3 code, BSD-2 weights per C02)",
        "weights": "yyfz233/Pi3X",
        "gated": False,
        "intrinsics_input": "intrinsics conditioning accepted and used (Kalibr K)",
        "run": "yes, with the Kalibr intrinsics",
    },
    "vggt": {
        "code": "https://github.com/facebookresearch/vggt",
        "weights": "facebook/VGGT-1B (CC BY-NC 4.0)",
        "gated": False,
        "intrinsics_input": "none in the released API",
        "run": "not run with a calibration; C03's uncalibrated 2/9 quoted",
    },
    "ffvo": {
        "code": "none found (arXiv 2609.13733 has no code or project link)",
        "weights": "none",
        "gated": None,
        "intrinsics_input": "unknown",
        "run": "not runnable",
    },
}


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    import yaml
    crit = yaml.safe_load(Path("configs/pathrec_prereg.yaml").read_text())["gate1_pass_criteria"]
    uzh = Path("data/uzh")
    out = {"provenance": {"commit": _git_head(),
                          "prereg": "configs/f04_round2_prereg.yaml"},
           "preflight": PREFLIGHT, "paths": {}}
    for path_name in PATHS:
        root = Path("data/pathrec/poses") / path_name
        rows = {}
        for seq in SEQS:
            p = root / f"{seq}.npz"
            if not p.exists():
                continue
            poses = np.load(p)["c2w"]
            d = np.load(uzh / seq / "truth.npz")
            meta = json.loads((uzh / seq / "meta.json").read_text())
            q = np.asarray(d["quat"], dtype=np.float64)
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            n = min(len(poses), len(d["pos"]))
            qe = Rotation.from_matrix(poses[:n, :3, :3]).as_quat()
            sc = score_clip(poses[:n, :3, 3], d["pos"][:n],
                            dt=1.0 / float(meta.get("fps_measured", meta.get("fps"))),
                            est_quat=qe, true_quat=q[:n], window_s=0.5)
            ok = (sc["accel_dir_err_deg_median"] <= crit["acceleration_direction"]["threshold_deg"]
                  and crit["acceleration_magnitude"]["interval"][0] <= sc["accel_mag_ratio"]
                  <= crit["acceleration_magnitude"]["interval"][1]
                  and sc["coverage_fraction"] >= 0.2)
            rows[seq] = {**{k: sc.get(k) for k in (
                "accel_dir_err_deg_median", "accel_mag_ratio", "coverage_fraction",
                "ate_sim3_m", "scale_sim3", "step_dir_err_deg_median",
                "rel_rot_err_deg_median")}, "pass": bool(ok)}
        if rows:
            dirs = [r["accel_dir_err_deg_median"] for r in rows.values()]
            ratios = [r["accel_mag_ratio"] for r in rows.values()]
            scales = [r["scale_sim3"] for r in rows.values()]
            out["paths"][path_name] = {
                "n_sequences": len(rows), "n_pass": int(sum(r["pass"] for r in rows.values())),
                "median_dir_err_deg": float(np.median(dirs)),
                "median_mag_ratio": float(np.median(ratios)),
                "min_scale_sim3": float(np.min(scales)),
                "max_scale_sim3": float(np.max(scales)),
                "rows": rows}
    Path("results/f06_feedforward").mkdir(parents=True, exist_ok=True)
    Path("results/f06_feedforward/summary.json").write_text(json.dumps(out, indent=1))
    for pn, b in out["paths"].items():
        print(f"{pn:12s} {b['n_pass']}/{b['n_sequences']} pass, median dir "
              f"{b['median_dir_err_deg']:.1f} deg, median ratio {b['median_mag_ratio']:.2f}, "
              f"Sim(3) scale {b['min_scale_sim3']:.2f}-{b['max_scale_sim3']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
