"""UZH forward-clip diagnostic: reproduce Track E's coherence transfer and the 0.10 s gain.

This is a diagnostic, not the pre-registered label-free window rule.  It applies:
* the coherence tier at a 0.10 s required lean, frozen tau 0.6197 / 0.2921;
* the gain tier at 0.10 s and 0.37 s with the TII R2 band unchanged.

The forward subset is the 9 sequences whose name contains ``forward`` (27 real clips),
which is the set behind Track E's 0.111 / 1.0 / 1.0.

    PYTHONPATH=. pixi run python experiments/d_uzh_diagnostic.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.verdict import FitOptions, coherence, fit_gain
from src.verdict.decide import JointBand, flag
from src.verdict.lean import required_lean
from src.verdict.schema import Clip

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["real", "tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll",
              "yaw_control", "null_edit"]
EDITS = ["tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control"]


def load_uzh(path: Path) -> Clip:
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        return Clip(t=np.asarray(d["t"]), pos=np.asarray(d["pos"]),
                    yaw=np.asarray(d["yaw"]),
                    depicted_lean=np.asarray(d["depicted_lean"]),
                    depicted_conf=np.asarray(d["depicted_conf"]), meta=meta)


def frozen_band() -> JointBand:
    cfg = yaml.safe_load((ROOT / "configs/gain_frozen.yaml").read_text())
    a = cfg["bands"]["axis"]
    return JointBand(tuple(a["keys"]), {k: float(v) for k, v in a["center"].items()},
                     {k: float(v) for k, v in a["scale"].items()},
                     float(a["threshold"]), 0.05, int(a["n"]))


def cluster_rate(rows: list[dict], column: str, sequences: list[str],
                 n_boot: int = 5000, seed: int = 20261114) -> dict:
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(sequences), size=len(sequences), replace=True)
        sub = pd.concat([df[df.sequence == sequences[i]] for i in pick])
        boots.append(float(sub[column].mean()))
    return {"rate": float(df[column].mean()),
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "n": int(len(df))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/d_uzh_diagnostic")
    args = ap.parse_args()
    out = ROOT / args.out / "d01"
    out.mkdir(parents=True, exist_ok=True)
    clips = {}
    for path in sorted((ROOT / "data/processed/track_e_uzh/clips").rglob("*.npz")):
        c = load_uzh(path)
        if "forward" not in str(c.meta.get("sequence")):
            continue
        seq = str(c.meta["sequence"])
        cond = str(c.meta["condition"])
        if cond not in CONDITIONS:
            continue
        clips[(seq, c.meta["clip_id"], cond)] = c
    bases = {}
    for (seq, cid, cond), c in clips.items():
        base = cid[: -len("_" + cond)]
        bases.setdefault((seq, base), {})[cond] = c
    seqs = sorted({s for s, _ in bases})
    band = frozen_band()
    coh_cfg = yaml.safe_load((ROOT / "configs/coherence_frozen.yaml").read_text())
    tau_c = float(coh_cfg["thresholds"]["tau_correlation"])
    tau_s = float(coh_cfg["thresholds"]["tau_sign"])

    rows_coh, rows_gain = [], {0.10: [], 0.37: []}
    for (seq, base), conds in sorted(bases.items()):
        if "real" not in conds:
            continue
        for window in (0.10, 0.37):
            real = conds["real"]
            req = required_lean(real.pos, real.yaw, real.t, window_s=window)
            cw = replace(real, lean_req=req.l_req0, vel_req=req.velocity_reg,
                         tilt_req_deg=req.tilt_deg)
            r = fit_gain(cw, FitOptions(bootstrap=False, sg_window_s=window))
            rows_gain[window].append({"sequence": seq, "clip_id": base,
                                      "condition": "real", "gain": r.gain,
                                      "flagged": bool(flag(r, "R2_axis_gains", BandsProxy(band)))})
            for cond in EDITS:
                if cond not in conds:
                    continue
                c = conds[cond]
                req = required_lean(c.pos, c.yaw, c.t, window_s=window)
                cw = replace(c, lean_req=req.l_req0, vel_req=req.velocity_reg,
                             tilt_req_deg=req.tilt_deg)
                rr = fit_gain(cw, FitOptions(bootstrap=False, sg_window_s=window))
                rows_gain[window].append({"sequence": seq, "clip_id": base,
                                          "condition": cond, "gain": rr.gain,
                                          "flagged": bool(flag(rr, "R2_axis_gains",
                                                               BandsProxy(band)))})
            # coherence at 0.10 s (the diagnostic window)
            req = required_lean(real.pos, real.yaw, real.t, window_s=0.10)
            cw = replace(real, lean_req=req.l_req0, vel_req=req.velocity_reg,
                         tilt_req_deg=req.tilt_deg)
            r = fit_gain(cw, FitOptions(bootstrap=False, sg_window_s=0.10))
            coh = coherence(cw, lag_ms=r.lag_ms)
            rows_coh.append({"sequence": seq, "clip_id": base, "condition": "real",
                             "corr": bool(coh.correlation_score > tau_c),
                             "sign": bool(coh.sign_score > tau_s)})
            for cond in EDITS:
                if cond not in conds:
                    continue
                c = conds[cond]
                req = required_lean(c.pos, c.yaw, c.t, window_s=0.10)
                cw = replace(c, lean_req=req.l_req0, vel_req=req.velocity_reg,
                             tilt_req_deg=req.tilt_deg)
                rr = fit_gain(cw, FitOptions(bootstrap=False, sg_window_s=0.10))
                coh = coherence(cw, lag_ms=rr.lag_ms)
                rows_coh.append({"sequence": seq, "clip_id": base, "condition": cond,
                                 "corr": bool(coh.correlation_score > tau_c),
                                 "sign": bool(coh.sign_score > tau_s)})

    metrics = {"n_sequences": len(seqs), "n_real": sum(1 for r in rows_coh
                                                       if r["condition"] == "real")}
    # coherence rates
    real_coh = [r for r in rows_coh if r["condition"] == "real"]
    metrics["coherence"] = {
        "FRR_corr": cluster_rate(real_coh, "corr", seqs),
        "FRR_sign": cluster_rate(real_coh, "sign", seqs),
    }
    for cond in EDITS:
        sub = [r for r in rows_coh if r["condition"] == cond]
        metrics["coherence"][f"det_corr_{cond}"] = cluster_rate(sub, "corr", seqs)
        metrics["coherence"][f"det_sign_{cond}"] = cluster_rate(sub, "sign", seqs)
    # gain rates
    for window, rows in rows_gain.items():
        real = [r for r in rows if r["condition"] == "real"]
        metrics[f"gain_{window}"] = {"FRR": cluster_rate(real, "flagged", seqs)}
        for cond in EDITS:
            sub = [r for r in rows if r["condition"] == cond]
            metrics[f"gain_{window}"][f"det_{cond}"] = cluster_rate(sub, "flagged", seqs)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    lines = ["# UZH forward diagnostic (not the label-free rule)", "",
             f"{metrics['n_sequences']} sequences, {metrics['n_real']} real clips", "",
             "| statistic | FRR | removal | mirror | x0.5 | x1.5 | yaw |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    c = metrics["coherence"]
    lines.append(f"| coherence corr | {c['FRR_corr']['rate']:.3f} | "
                 f"{c['det_corr_tilt_removal']['rate']:.3f} | "
                 f"{c['det_corr_mirror_roll']['rate']:.3f} | "
                 f"{c['det_corr_scale_0.5']['rate']:.3f} | "
                 f"{c['det_corr_scale_1.5']['rate']:.3f} | "
                 f"{c['det_corr_yaw_control']['rate']:.3f} |")
    for window in (0.10, 0.37):
        g = metrics[f"gain_{window}"]
        lines.append(f"| gain {window:.2f}s, TII band | {g['FRR']['rate']:.3f} | "
                     f"{g['det_tilt_removal']['rate']:.3f} | "
                     f"{g['det_mirror_roll']['rate']:.3f} | "
                     f"{g['det_scale_0.5']['rate']:.3f} | "
                     f"{g['det_scale_1.5']['rate']:.3f} | "
                     f"{g['det_yaw_control']['rate']:.3f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


class BandsProxy:
    """Minimal object with the scalar/axis attributes flag() expects."""

    def __init__(self, axis):
        self.axis = axis
        self.scalar = None
        self.joint = None


if __name__ == "__main__":
    raise SystemExit(main())
