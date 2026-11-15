"""UZH official-v2 analysis from Track E's per-clip rows (read-only).

Track E's `metrics_w370ms.json` carries per-clip gx/gy, coherence flags and testability for
the official schema-v2 clips.  The dense `lean_req` itself is not on R2, so this script uses
their rows for (i) the TII-band result, (ii) a UZH leave-one-sequence-out R2 band, and
(iii) the coherence tier on the same population.

    PYTHONPATH=. pixi run python experiments/d_uzh_v2_analysis.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDITS = ["tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control"]


def fit_band(rows):
    gx = np.array([r["gx"] for r in rows], float)
    gy = np.array([r["gy"] for r in rows], float)
    center = {"gx": float(np.median(gx)), "gy": float(np.median(gy))}
    scale = {}
    for k, v in (("gx", gx), ("gy", gy)):
        q = np.quantile(v, [0.025, 0.975])
        scale[k] = float(max((q[1] - q[0]) / 3.92, 1e-6))
    stats = [max(abs(r["gx"] - center["gx"]) / scale["gx"],
                 abs(r["gy"] - center["gy"]) / scale["gy"]) for r in rows]
    return center, scale, float(np.quantile(stats, 0.95)), len(rows)


def flag(row, band):
    center, scale, tau, _ = band
    t = max(abs(row["gx"] - center["gx"]) / scale["gx"],
            abs(row["gy"] - center["gy"]) / scale["gy"])
    return bool(t > tau)


def rate(rows, key, seqs, n_boot=5000, seed=20261114):
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(seqs), size=len(seqs), replace=True)
        vals.append(float(pd.concat([df[df.seq == seqs[i]] for i in pick])[key].mean()))
    return {"rate": float(df[key].mean()) if len(df) else float("nan"),
            "lo": float(np.percentile(vals, 2.5)) if vals else float("nan"),
            "hi": float(np.percentile(vals, 97.5)) if vals else float("nan"),
            "n": int(len(df))}


def main() -> int:
    d = json.loads((ROOT / "results/d_uzh_v2/track_e_metrics_w370ms.json").read_text())
    rows = [r for r in d["rows"] if r["testable"]]
    real = [r for r in rows if r["cond"] == "real"]
    seqs = sorted({r["seq"] for r in real})
    out = {"n_real": len(real), "sequences": seqs,
           "tii_band": {}, "uzh_loso": {}, "coherence": {}}
    # TII band from Track E's flags
    out["tii_band"]["real"] = rate([{"seq": r["seq"], "v": bool(r["flagged_r2"])}
                                    for r in real], "v", seqs)
    for cond in EDITS:
        sub = [r for r in rows if r["cond"] == cond]
        if sub:
            out["tii_band"][cond] = rate([{"seq": r["seq"], "v": bool(r["flagged_r2"])}
                                          for r in sub], "v", seqs)
    # UZH leave-one-sequence-out band
    for hold in seqs:
        fit = [r for r in real if r["seq"] != hold]
        if len(fit) < 8:
            continue
        band = fit_band(fit)
        for r in [x for x in rows if x["seq"] == hold]:
            r[f"loso_{hold}"] = flag(r, band)
    # aggregate LOSO flags
    for cond in ["real"] + EDITS:
        sub = [r for r in rows if r["cond"] == cond and any(
            k.startswith("loso_") for k in r)]
        if sub:
            # one LOSO flag per row: the flag from its held-out sequence column
            flags = []
            for r in sub:
                keys = [k for k in r if k.startswith("loso_")]
                flags.append(bool(r[keys[0]]))
            out["uzh_loso"][cond] = rate(
                [{"seq": r["seq"], "v": f} for r, f in zip(sub, flags)], "v", seqs)
    # coherence (TII tau, Track E flags)
    for cond in ["real"] + EDITS:
        sub = [r for r in rows if r["cond"] == cond]
        if sub:
            out["coherence"][cond] = rate(
                [{"seq": r["seq"], "v": bool(r["flagged_coh"])} for r in sub],
                "v", seqs)
    out["band"] = {"tii": {"tau": 2.1922, "center": {"gx": 0.8137, "gy": 0.7181},
                           "scale": {"gx": 0.2654, "gy": 0.0735}}}
    out["loso_bands"] = {
        hold: {"center": c, "scale": s, "tau": t, "n_fit": n}
        for hold in seqs
        for c, s, t, n in [fit_band([r for r in real if r["seq"] != hold])]
        if len([r for r in real if r["seq"] != hold]) >= 8}
    (ROOT / "results/d_uzh_v2/analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
