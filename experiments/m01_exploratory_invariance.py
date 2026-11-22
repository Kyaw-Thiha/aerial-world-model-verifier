"""Track M, exploratory: two yaw-invariant replacements for the per-axis coherence tier.

Not pre-registered.  Written after M01 showed that the yaw control damages the per-axis
correlation systematically.  A tilt-preserving warp rotates the lean vector's azimuth, so a
statistic that is invariant to that rotation should not see it at all.  Two candidates:

* **complex coherence** ``|sum(l_dep conj(l_req))| / sqrt(sum|l_dep|^2 sum|l_req|^2)`` --
  exactly invariant to a global rotation of the depicted lean, so the yaw control cannot move
  it, and a mirrored roll (a conjugation, not a rotation) still can;
* **magnitude coherence** ``corr(|l_dep|, |l_req|)`` -- invariant to any azimuth change,
  including a mirror, which is why it cannot see a mirrored roll.

Both are scored the same way as the published tier: the maximum over 8-frame windows of
``1 - statistic``, leave-one-flight-out 95th-percentile thresholds on the gate-kept development
clips.  Reported as exploratory in every table.

    PYTHONPATH=. pixi run python experiments/m01_exploratory_invariance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.airframe.kinematics import TII_FLIGHTS, load_tii_kinematics
from src.airframe.schema_v3 import required_lean_at
from src.verdict.schema import load_clip

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data/processed/track_d/clips"
OUT = ROOT / "results/m01"
DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
WINDOW, HOP = 8, 4
CONDS = ("real", "tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5",
         "yaw_control", "null_edit")


def complex_coherence(d, q) -> float:
    z1 = d[:, 0] + 1j * d[:, 1]
    z2 = q[:, 0] + 1j * q[:, 1]
    ok = np.isfinite(z1) & np.isfinite(z2)
    z1, z2 = z1[ok], z2[ok]
    if len(z1) < 6:
        return float("nan")
    den = np.sqrt(np.sum(np.abs(z1) ** 2) * np.sum(np.abs(z2) ** 2))
    return float(np.abs(np.sum(z1 * np.conj(z2))) / den) if den > 1e-12 else float("nan")


def magnitude_coherence(d, q) -> float:
    a = np.linalg.norm(np.asarray(d, float), axis=1)
    b = np.linalg.norm(np.asarray(q, float), axis=1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 6 or np.std(a) < 1e-4 or np.std(b) < 1e-4:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def windowed(fn, dep, req, take_min: bool) -> float:
    n = len(dep)
    spans = [(0, n)] if n <= WINDOW else [(s, s + WINDOW)
                                          for s in range(0, n - WINDOW + 1, HOP)]
    vals = [fn(dep[lo:hi], req[lo:hi]) for lo, hi in spans]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return 1.0 - (min(vals) if take_min else max(vals))


def main() -> None:
    keep = pd.read_csv(OUT / "signal_by_clip.csv")[["clip_id", "kept_S3"]]
    keepmap = dict(zip(keep.clip_id, keep.kept_S3.astype(bool)))
    kin = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    rows = []
    for f in TII_FLIGHTS:
        cache = {}
        for d in sorted((CLIPS / f).iterdir()):
            cid, cond = d.name.split("__")
            if cond not in CONDS:
                continue
            c = load_clip(d / "verdict_clip.npz")
            if cid not in cache:
                cache[cid] = required_lean_at(kin[f], "M0", {}, np.asarray(c.t, float),
                                              np.asarray(c.yaw, float))["lean"]
            dep = np.asarray(c.depicted_lean, float)
            rows.append({"clip_id": cid, "flight": f, "cond": cond,
                         "kept": keepmap.get(cid, False),
                         "cx_wmax": windowed(complex_coherence, dep, cache[cid], True),
                         "mag_wmax": windowed(magnitude_coherence, dep, cache[cid], False)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exploratory_invariance.csv", index=False)
    dev = df[df.kept & df.flight.isin(DEV)]
    out = {}
    for stat in ("cx_wmax", "mag_wmax"):
        real = dev[dev.cond == "real"]
        ent = {}
        for cond in CONDS:
            hit = tot = 0
            for _, r in dev[dev.cond == cond].iterrows():
                fit = real[real.flight != r.flight][stat].values
                fit = fit[np.isfinite(fit)]
                if len(fit) < 5:
                    continue
                tot += 1
                hit += int(np.isfinite(r[stat]) and r[stat] > np.quantile(fit, 0.95))
            ent[cond] = f"{hit}/{tot}"
        out[stat] = ent
    out["median_by_cond"] = {stat: dev.groupby("cond")[stat].median().round(3).to_dict()
                             for stat in ("cx_wmax", "mag_wmax")}
    out["label"] = "exploratory, not pre-registered"
    (OUT / "exploratory_invariance.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
