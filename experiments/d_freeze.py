"""Freeze the selected gain statistic into ``configs/gain_frozen.yaml``.

Run only after Gate D1.  The file is what Track E reads unchanged; it records the
statistic, its nuisance settings, the abstention rules, the confidence gate, the pooled
conformal bands and the per-fold bands, plus the commit that produced them.

    PYTHONPATH=. pixi run python experiments/d_freeze.py --run results/d_eval/d01
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import yaml

from src.verdict import Bands, fit_bands, fit_gain, gate_kept_from_conf, load_clip
from src.verdict.abstain import assess

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _bands_dict(bands: Bands) -> dict:
    out = {}
    if bands.scalar:
        out["scalar"] = bands.scalar.as_dict()
    if bands.axis:
        out["axis"] = bands.axis.as_dict()
    if bands.joint:
        out["joint"] = bands.joint.as_dict()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/d_eval/d01")
    ap.add_argument("--out", default="configs/gain_frozen.yaml")
    args = ap.parse_args()
    run = ROOT / args.run
    metrics = json.loads((run / "metrics.json").read_text())
    rule = metrics["selected_pooled"]
    prereg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    gate_cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())

    clips = {}
    for path in sorted((ROOT / "data/processed/track_d/clips").glob("*/*/verdict_clip.npz")):
        clip = load_clip(path)
        clips[(clip.flight, clip.clip_id, clip.condition)] = clip

    from experiments.d_eval import FitOptions, load_results, real_testable
    opts = FitOptions()
    results = load_results(run / "primary_results.json")
    testable = real_testable(clips, results, prereg["abstention"], gate_cfg)
    real = [results[(f, cid, "real")] for (f, cid), v in testable.items()
            if v["testable"]]
    pooled = fit_bands(real, rule, 0.05)
    per_fold = {}
    for flight in FLIGHTS:
        fit = [results[(f, cid, "real")] for (f, cid), v in testable.items()
               if v["testable"] and f != flight]
        per_fold[flight] = _bands_dict(fit_bands(fit, rule, 0.05))

    doc = {
        "version": 1,
        "date": str(date.today()),
        "status": "frozen",
        "source_run": str(run.relative_to(ROOT)),
        "commit": git_hash(),
        "rule": rule,
        "statistic": ("per-clip regression of depicted lean on required lean; "
                      "l_dep = G1*l_req0 + G2*(R_yaw v/g) + b"),
        "options": {k: v for k, v in FitOptions().__dict__.items()
                    if k not in ("bootstrap",)},
        "abstention": prereg["abstention"],
        "gate": {"config": "configs/gate_frozen.yaml", "signal": gate_cfg["signal"],
                 "tau": gate_cfg["tau"], "direction": gate_cfg["direction"],
                 "application": "pair_level in evaluation; per clip when scoring one clip"},
        "bands": _bands_dict(pooled),
        "bands_per_fold": per_fold,
        "n_real_testable": len(real),
        "note": ("A pass means no violation of the tested size on a clip with enough "
                 "excitation. It is not proof that the flight is physically consistent."),
    }
    (ROOT / args.out).write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"wrote {args.out}: rule {rule}, bands {_bands_dict(pooled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
