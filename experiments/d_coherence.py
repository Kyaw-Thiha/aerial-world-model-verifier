"""Task 3: the scale-free coherence tier on TII, leave-one-flight-out.

Reads the schema-v2 clips and the D02 gain fits (for the lag and the pair-level testable
set), computes per-axis correlation and sign agreement between depicted and required lean,
and reports threshold-free AUROC and the frozen-threshold operating point.

Pre-registration: ``configs/coherence_prereg.yaml`` (committed before this script was run
on any edited clip).  The frozen thresholds go to ``configs/coherence_frozen.yaml`` for
Track E to apply unchanged to UZH-FPV.

    PYTHONPATH=. pixi run python experiments/d_coherence.py --run results/d_eval/d02
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.score.verdict import auroc_detect, wilson_interval
from src.verdict import coherence, load_clip
from src.verdict.coherence import coherence as coherence_fn

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
TRACK = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}
CONDITIONS = ["real", "null_edit", "tilt_removal", "scale_0.5", "scale_1.5",
              "mirror_roll", "yaw_control"]
EDITS = ["tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control"]


def flight_type(flight: str) -> str:
    if flight in TRACK:
        return "track"
    return "piloted" if "p" in flight.split("-")[1] else "autonomous"


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def rate(values) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def wilson(values) -> list[float]:
    if not len(values):
        return [float("nan"), float("nan")]
    return list(wilson_interval(int(np.sum(values)), len(values)))


def load_clips() -> dict:
    out = {}
    for path in sorted((ROOT / "data/processed/track_d/clips").glob("*/*/verdict_clip.npz")):
        clip = load_clip(path)
        out[(clip.flight, clip.clip_id, clip.condition)] = clip
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/d_eval/d02")
    ap.add_argument("--out", default="results/d_coherence")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--frozen", default="configs/coherence_frozen.yaml")
    args = ap.parse_args()
    run = args.run
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    from experiments.d_eval import load_results, real_testable
    gain_prereg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    gate = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    results = load_results(ROOT / run / "primary_results.json")
    clips = load_clips()
    testable = real_testable(clips, results, gain_prereg["abstention"], gate)

    rows = []
    for (flight, cid, cond), clip in clips.items():
        if cond not in CONDITIONS or not testable.get((flight, cid), {}).get("testable"):
            continue
        res = results.get((flight, cid, cond))
        if res is None:
            continue
        coh = coherence_fn(clip, lag_ms=res.lag_ms)
        rows.append({"flight": flight, "type": flight_type(flight), "clip_id": cid,
                     "condition": cond, "lag_ms": res.lag_ms,
                     "r_x": coh.r_x, "r_y": coh.r_y, "r_min": coh.r_min,
                     "s_x": coh.s_x, "s_y": coh.s_y, "s_min": coh.s_min,
                     "correlation_score": coh.correlation_score,
                     "sign_score": coh.sign_score, "n_used": coh.n_used,
                     "track": flight in TRACK})
    df = pd.DataFrame(rows)
    df.to_csv(out / "coherence_table.csv", index=False)

    real = df[df.condition == "real"]
    pooled = {
        "tau_correlation": float(np.quantile(real.correlation_score, 0.95)),
        "tau_sign": float(np.quantile(real.sign_score, 0.95)),
        "n_real": int(len(real)),
    }
    per_fold = {}
    for flight in FLIGHTS:
        fit = real[real.flight != flight]
        per_fold[flight] = {
            "tau_correlation": float(np.quantile(fit.correlation_score, 0.95))
            if len(fit) >= 8 else float("nan"),
            "tau_sign": float(np.quantile(fit.sign_score, 0.95))
            if len(fit) >= 8 else float("nan"),
            "n_fit": int(len(fit)),
        }

    # AUROC is threshold-free; the operating point uses the fold's fit threshold.
    aurocs, operating = {}, {}
    for family in EDITS:
        neg = df[df.condition == family]
        pairs = neg.merge(real[["flight", "clip_id", "correlation_score", "sign_score"]],
                          on=["flight", "clip_id"], suffixes=("", "_real"))
        aurocs[family] = {
            "n": int(len(pairs)),
            "auroc_correlation": float(auroc_detect(
                pairs.correlation_score_real.to_numpy(),
                pairs.correlation_score.to_numpy())),
            "auroc_sign": float(auroc_detect(
                pairs.sign_score_real.to_numpy(), pairs.sign_score.to_numpy())),
        }
        frr_c, det_c, frr_s, det_s = [], [], [], []
        for flight in FLIGHTS:
            thr = per_fold[flight]
            if not np.isfinite(thr["tau_correlation"]):
                continue
            sub_real = real[(real.flight == flight)]
            sub_neg = df[(df.flight == flight) & (df.condition == family)]
            frr_c.extend((sub_real.correlation_score > thr["tau_correlation"]).tolist())
            det_c.extend((sub_neg.correlation_score > thr["tau_correlation"]).tolist())
            frr_s.extend((sub_real.sign_score > thr["tau_sign"]).tolist())
            det_s.extend((sub_neg.sign_score > thr["tau_sign"]).tolist())
        operating[family] = {
            "frr_correlation": rate(frr_c), "frr_correlation_wilson": wilson(frr_c),
            "det_correlation": rate(det_c), "det_correlation_wilson": wilson(det_c),
            "frr_sign": rate(frr_s), "frr_sign_wilson": wilson(frr_s),
            "det_sign": rate(det_s), "det_sign_wilson": wilson(det_s),
            "n_det": len(det_c),
        }

    # yaw control beside every detection at the same fold thresholds
    yaw = df[df.condition == "yaw_control"]
    distributions = {}
    for cond in ("real", "null_edit", "yaw_control"):
        sub = df[df.condition == cond]
        distributions[cond] = {}
        for typ in ("piloted", "autonomous", "track"):
            s = sub[sub.type == typ]
            distributions[cond][typ] = {
                "n": int(len(s)),
                "r_min_median": float(s.r_min.median()) if len(s) else float("nan"),
                "r_min_iqr": [float(s.r_min.quantile(0.25)),
                              float(s.r_min.quantile(0.75))] if len(s) else [float("nan")] * 2,
                "s_min_median": float(s.s_min.median()) if len(s) else float("nan"),
                "s_min_iqr": [float(s.s_min.quantile(0.25)),
                              float(s.s_min.quantile(0.75))] if len(s) else [float("nan")] * 2,
            }

    metrics = {
        "run": tag, "git": git_hash(), "source_run": run,
        "prereg": "configs/coherence_prereg.yaml",
        "n_real_testable": int(len(real)),
        "pooled_thresholds": pooled, "per_fold_thresholds": per_fold,
        "auroc": aurocs, "operating_point": operating,
        "distributions": distributions,
        "per_flight_real": {
            f: {"n": int(len(real[real.flight == f])),
                "r_min_median": float(real[real.flight == f].r_min.median())
                if len(real[real.flight == f]) else float("nan"),
                "s_min_median": float(real[real.flight == f].s_min.median())
                if len(real[real.flight == f]) else float("nan")}
            for f in FLIGHTS},
        "note": ("x0.5/x1.5 are expected to sit at chance: a positive scale leaves the "
                 "correlation unchanged. Track flights are not validated (4 testable clips)."),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))

    # frozen thresholds for Track E
    frozen = {
        "version": 1, "date": str(date.today()), "status": "frozen",
        "source_run": str(Path(run) / "coherence"), "commit": git_hash(),
        "rule": {
            "correlation": "flag if 1 - min(r_x, r_y) > tau_correlation",
            "sign": "flag if 1 - min(s_x, s_y) > tau_sign",
        },
        "thresholds": pooled, "per_fold": per_fold,
        "inputs": {"schema": "verdict_clip v2", "lag": "from the gain fit"},
        "no_recalibration": True,
        "note": metrics["note"],
    }
    (ROOT / args.frozen).write_text(yaml.safe_dump(frozen, sort_keys=False))

    # report table
    lines = ["# Track D coherence tier", "",
             f"run `{tag}` · n real {len(real)} · x0.5/x1.5 expected at chance", "",
             "| family | n | AUROC corr | corr det | AUROC sign | sign det | yaw det (corr) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    yaw_det = operating["yaw_control"]["det_correlation"]
    for family in EDITS:
        a, o = aurocs[family], operating[family]
        lines.append(f"| {family} | {a['n']} | {a['auroc_correlation']:.3f} | "
                     f"{o['det_correlation']:.3f} | {a['auroc_sign']:.3f} | "
                     f"{o['det_sign']:.3f} | {yaw_det:.3f} |")
    lines += ["", "Track flights: 4 testable clips; not validated.", ""]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print((out / "table.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
