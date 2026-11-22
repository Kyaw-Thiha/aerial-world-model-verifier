"""Track K diagnostics: why the airframe models fail (exploratory, after the pre-registered fits).

For the current model (M0, k_d = 0.295) this measures, per flight/sequence, the residual
level, its mean direction, its correlation with speed / acceleration / jerk, and how its
variance splits across frequency bands.  It is a diagnostic of the negative result, not a
new model and not a pre-registered test.

    PYTHONPATH=. python experiments/k02_diagnostics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from src.airframe.fit import fit_lag, residual, shift_kinematics
from src.airframe.kinematics import load_all
from src.airframe.models import axis_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/k02_diagnostics"


def band_shares(x: np.ndarray, fs: float) -> dict:
    """Variance share below 0.5, 0.5-2, 2-10 and above 10 Hz, per axis pooled."""
    bands = [(0.0, 0.5), (0.5, 2.0), (2.0, 10.0), (10.0, None)]
    tot = np.zeros(len(bands))
    n = 0
    for j in range(x.shape[1]):
        f, p = welch(x[:, j], fs=fs, nperseg=min(len(x), 1024))
        for i, (lo, hi) in enumerate(bands):
            m = (f >= lo) & ((f < hi) if hi is not None else np.ones_like(f, bool))
            tot[i] += float(np.sum(p[m]))
        n += 1
    s = tot.sum()
    return {f"{lo}-{hi if hi is not None else 'inf'}Hz": float(tot[i] / s if s else np.nan)
            for i, (lo, hi) in enumerate(bands)}


def run(units, model="M0", params=None):
    params = params or {}
    rows = []
    for k in units:
        lag, _ = fit_lag(k, model, params)
        ks = shift_kinematics(k, lag)
        ang, lean = residual(ks, model, params)
        m = np.isfinite(ang)
        r = lean[m]
        sp = np.linalg.norm(k.v[m], axis=1)
        ac = np.linalg.norm(k.a[m], axis=1)
        jk = np.linalg.norm(k.j[m], axis=1)
        mag = np.linalg.norm(r, axis=1)
        # remove the mean (a constant airframe offset is not a drag effect)
        cen = r - r.mean(axis=0)
        rows.append({
            "name": k.name, "dataset": k.dataset, "n": int(m.sum()),
            "rms_deg": float(np.sqrt(np.mean(np.degrees(mag) ** 2))),
            "median_deg": float(np.median(np.degrees(mag))),
            "mean_lean_deg": np.degrees(r.mean(axis=0)).tolist(),
            "frac_gt10": float(np.mean(np.degrees(mag) > 10)),
            "frac_gt20": float(np.mean(np.degrees(mag) > 20)),
            "corr_speed": float(np.corrcoef(mag, sp)[0, 1]),
            "corr_accel": float(np.corrcoef(mag, ac)[0, 1]),
            "corr_jerk": float(np.corrcoef(mag, jk)[0, 1]),
            "bands_centred": band_shares(cen, 1.0 / k.dt),
            "lag_s": float(lag),
        })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tii = load_all("tii")
    uzh = load_all("uzh", rig="forward")
    doc = {"model": "M0", "k_d": 0.295,
           "note": "exploratory diagnostic; residual is the M0 angle with the per-unit lag",
           "tii": run(tii), "uzh_forward": run(uzh)}
    (OUT / "diagnostics.json").write_text(json.dumps(doc, indent=1))
    print(f"{'unit':28s} {'rms':>6s} {'med':>6s} {'>10%':>6s} {'>20%':>6s} "
          f"{'spd':>6s} {'acc':>6s} {'jerk':>6s}  bands(<0.5/0.5-2/2-10/>10)")
    for r in doc["tii"] + doc["uzh_forward"]:
        b = r["bands_centred"]
        print(f"{r['name']:28s} {r['rms_deg']:6.2f} {r['median_deg']:6.2f} "
              f"{100*r['frac_gt10']:6.1f} {100*r['frac_gt20']:6.1f} "
              f"{r['corr_speed']:6.2f} {r['corr_accel']:6.2f} {r['corr_jerk']:6.2f}  "
              f"{b['0.0-0.5Hz']:.2f}/{b['0.5-2.0Hz']:.2f}/{b['2.0-10.0Hz']:.2f}/"
              f"{b['10.0-infHz']:.2f}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
