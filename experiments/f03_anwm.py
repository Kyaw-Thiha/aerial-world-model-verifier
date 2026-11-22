"""Track F, stage 2: score the ANWM extraction against the commanded path.

The ANWM clips carry a commanded path, so this is a direct accuracy measurement, not a
verdict. Everything is read from ``data/pathrec/poses_anwm/vipe/*.npz`` (poses + commanded
positions written by ``remote/f_anwm.py``) and scored with Track C's ``score_clip`` at the
clips' own 1 s step (the window is 0.5 s, i.e. 5 samples at 1 Hz, the minimum the
differentiator accepts).

Crash / degenerate rules follow C09: no pose produced, non-finite poses, a magnitude ratio
above 10, or fewer than 12 usable poses counts as a failure; a usable clip is one that
survives all of them.

    python experiments/f03_anwm.py --poses data/pathrec/poses_anwm/vipe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.pathrec.evaluate import score_clip

MAG_RATIO_DEGENERATE = 10.0
MIN_POSES = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="data/pathrec/poses_anwm/vipe")
    ap.add_argument("--out", default="results/f03_stage2/anwm.json")
    ap.add_argument("--expected-level", type=int, default=82)
    ap.add_argument("--expected-injected", type=int, default=45)
    args = ap.parse_args()
    root = Path(args.poses)
    rows = []
    for p in sorted(root.glob("*.npz")):
        clip = p.stem.replace("__", "/")
        z = np.load(p)
        poses = np.asarray(z["c2w"], dtype=np.float64)
        rec = {"clip": clip, "n_poses": int(len(poses)),
               "finite": bool(np.isfinite(poses).all())}
        if "commanded" not in z.files:
            rec["status"] = "no_commanded_path"
            rows.append(rec)
            continue
        cmd = np.asarray(z["commanded"], dtype=np.float64)
        n = min(len(poses), len(cmd), 16)
        rec["n_used"] = int(n)
        if n < MIN_POSES or not rec["finite"]:
            rec["status"] = "failed"
            rec["why"] = "too few or non-finite poses"
            rows.append(rec)
            continue
        q = Rotation.from_matrix(poses[:n, :3, :3]).as_quat()
        try:
            sc = score_clip(poses[:n, :3, 3], cmd[:n], dt=1.0, est_quat=q,
                            true_quat=np.tile([0, 0, 0, 1.0], (n, 1)), window_s=0.5)
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "failed"
            rec["why"] = f"{type(exc).__name__}: {exc}"
            rows.append(rec)
            continue
        rec.update({k: sc.get(k) for k in ("accel_dir_err_deg_median", "accel_mag_ratio",
                                           "coverage_fraction", "ate_sim3_m",
                                           "step_dir_err_deg_median", "scale_sim3")})
        degenerate = (not np.isfinite(rec.get("accel_mag_ratio") or np.nan)
                      or rec["accel_mag_ratio"] > MAG_RATIO_DEGENERATE)
        rec["status"] = "degenerate" if degenerate else "usable"
        rows.append(rec)

    usable = [r for r in rows if r["status"] == "usable"]
    failed = [r for r in rows if r["status"] != "usable"]
    # the bake-off's per-clip rule, for comparability with E7 (coverage is not meaningful on
    # 16 samples, so only direction and ratio are applied)
    def bakeoff_pass(r):
        d, q = r.get("accel_dir_err_deg_median"), r.get("accel_mag_ratio")
        return bool(d is not None and q is not None and d <= 30 and 0.7 <= q <= 1.4)

    n_bakeoff_pass = int(sum(bakeoff_pass(r) for r in usable))
    present_level = sum(1 for r in rows if r["clip"].startswith("clips/"))
    present_inj = sum(1 for r in rows if r["clip"].startswith("clips_injected/"))
    out = {
        "n_clips": len(rows),
        "expected_level": args.expected_level,
        "expected_injected": args.expected_injected,
        "missing_level": max(0, args.expected_level - present_level),
        "missing_injected": max(0, args.expected_injected - present_inj),
        "n_usable": len(usable),
        "n_failed": len(failed),
        "n_bakeoff_pass_among_usable": n_bakeoff_pass,
        "failed_by_status": {s: sum(1 for r in failed if r["status"] == s)
                             for s in {r["status"] for r in failed}},
        "median_dir_err_deg_usable": float(np.median(
            [r["accel_dir_err_deg_median"] for r in usable])) if usable else None,
        "median_mag_ratio_usable": float(np.median(
            [r["accel_mag_ratio"] for r in usable])) if usable else None,
        "median_ate_sim3_m_usable": float(np.median(
            [r["ate_sim3_m"] for r in usable])) if usable else None,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
