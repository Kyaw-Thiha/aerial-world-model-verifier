"""H04 amendment 3: score the 14B pilot (20 arm-A continuous + 15 held-bank clips).

Reuses the 5B campaign's scoring machinery unchanged (``experiments/h04_score.py``): the same
reader rows, the same frozen tiers and abstention rules, the same required-lean construction.
Only the populations differ:

* ``a_<path>``  — arm A, draw 0, of the 20 bank-profile paths; required side from
  ``results/h04/paths/paths.npz`` at the ``0.5/f95`` window;
* ``b_<h02>``   — the 15 held-bank flatness clips from H02's grid; required side from
  ``results/h04/heldbank_required_w0.37.npz`` (the frozen 0.37 s window, because the rule
  window is 2-8 s on a 16 s designed path and over-smooths it).

    PYTHONPATH=. pixi run -e gpu python experiments/h04_14b_score.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import experiments.h04_score as S
from src.perception.frames import camera_to_body_pitched
from src.verdict import Clip, assess, coherence, depicted_lean_from_gravity, fit_gain, \
    gate_kept_from_conf

ROOT = Path(__file__).resolve().parents[1]
TAG = "pilot14b"
YAW_TAG = "pilot14b_yaw+20"
WINDOW = "rule"


def main() -> int:
    gain_cfg = yaml.safe_load((ROOT / "configs/gain_frozen.yaml").read_text())
    coh_cfg = yaml.safe_load((ROOT / "configs/coherence_frozen.yaml").read_text())
    gate_cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    d = np.load(ROOT / "results/h04/paths/paths.npz")
    held = np.load(ROOT / "results/h04/heldbank_required_w0.37.npz")
    inputs = json.loads((ROOT / "data/h04_14b_inputs/manifest.json").read_text())
    rows = S.load_rows(ROOT / f"results/h04/depicted_{TAG}/depicted.jsonl")
    yaw_rows = S.load_rows(ROOT / f"results/h04/depicted_{YAW_TAG}/depicted.jsonl")

    def build(sid: str, source_rows: dict, draw: int = 0) -> Clip | None:
        frames = source_rows.get(sid)
        if not frames:
            return None
        idx = sorted(frames)
        meta_path = ROOT / "data/h04_14b_inputs" / sid / "meta.json"
        meta = json.loads(meta_path.read_text())
        if sid.startswith("a_"):
            name = sid[2:]
            t = d[f"{name}__t"][:len(idx)]
            lean_req = d[f"{name}__lean_req_{WINDOW}"][:len(idx)]
            vel_req = d[f"{name}__vel_req_{WINDOW}"][:len(idx)]
            tilt_req = d[f"{name}__tilt_req_{WINDOW}"][:len(idx)]
            pos = d[f"{name}__pos"][:len(idx)]
            yaw = d[f"{name}__yaw"][:len(idx)]
            path = name
        elif sid.startswith("b_"):
            name = meta["path"]
            t = held[f"{name}__t"][:len(idx)]
            lean_req = held[f"{name}__lean_req"][:len(idx)]
            vel_req = held[f"{name}__vel_req"][:len(idx)]
            tilt_req = held[f"{name}__tilt_req"][:len(idx)]
            pos = np.zeros((len(idx), 3))
            yaw = np.zeros(len(idx))
            path = f"h02_{name}"
        else:
            return None
        up = np.array([frames[i]["gravity_up_cam"] for i in idx], float)
        conf = np.array([frames[i].get("latitude_confidence_mean", np.nan) for i in idx])
        lean = depicted_lean_from_gravity(up, camera_to_body_pitched(0.0))
        return Clip(t=np.asarray(t, float), pos=pos, yaw=yaw, depicted_lean=lean,
                    depicted_conf=conf,
                    meta={"clip_id": sid, "condition": "flatness", "path": path},
                    lean_req=lean_req, vel_req=vel_req, tilt_req_deg=tilt_req)

    scored = []
    for sid in sorted(inputs["samples"]):
        clip = build(sid, rows)
        if clip is None:
            continue
        s = S.score_clip(clip, gain_cfg, coh_cfg, gate_cfg)
        s["population"] = "armA_continuous" if sid.startswith("a_") else "held_bank"
        s["path"] = clip.meta["path"]
        scored.append(s)

    yaw_scored = []
    for sid in sorted(inputs["samples"]):
        if not sid.startswith("a_"):
            continue
        clip = build(sid, yaw_rows)
        if clip is None:
            continue
        s = S.score_clip(clip, gain_cfg, coh_cfg, gate_cfg)
        s["population"] = "yaw_control"
        yaw_scored.append(s)

    out = {"tag": TAG, "n_rows": sum(len(v) for v in rows.values()),
           "n_clips": len(scored), "per_population": {},
           "yaw_control": S.rate_block(yaw_scored) if yaw_scored else {"n": 0},
           "rows": scored, "yaw_rows": yaw_scored}
    for pop in ("armA_continuous", "held_bank"):
        out["per_population"][pop] = S.rate_block(
            [s for s in scored if s["population"] == pop])
    (ROOT / "results/h04/scored_pilot14b.json").write_text(
        json.dumps(out, indent=1, default=float))
    for pop, b in out["per_population"].items():
        print(f"{pop:18s} n={b['n']} testable {b['testable_k']} accepted {b['accepted_k']} "
              f"coh {b['flag_coh_k']}/{b['testable_k']} r2 {b['flag_r2_k']}/{b['testable_k']} "
              f"resid {b['median_residual_deg']:.2f} ratio "
              f"{b['depicted_over_required_median']:.2f} reasons {b['abstention_reasons']}")
    print("yaw", {k: out["yaw_control"].get(k) for k in
                  ("n", "testable_k", "flag_coh_k", "accepted_k")})
    # H02 comparison for the held-bank clips
    for s in scored:
        if s["population"] != "held_bank":
            continue
        print(f"  {s['clip_id'][:46]:46s} testable {s['testable']} accepted {s['accepted']} "
              f"coh {s['flagged_coh']} resid {s['median_residual_deg']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
