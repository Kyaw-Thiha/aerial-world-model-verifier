"""Track M, Stage 3 (generators): the gates applied to the published H04 and H01 verdicts.

One-shot test, no tuning.  The verdicts themselves are read from the tracks that produced them:

* H04 SolarWM-5B, 234 clips, ``results/h04/scored_main_rows.json`` (Track H3, copied
  byte-identically; sha256 recorded below);
* H01 ANWM, 80 designed paths x 5 seeds, ``results/h01_tables/t1_seed_variance.csv`` and
  ``results/h01_seed_specs/specs.json`` (Track H, read from this branch).

What this track adds is the *abstention*: Gate A's path-side signal (from the published
required lean of each H04 path, and from the designed path for H01) and Gate B's disagreement
(from ``track-m``'s own reader pass).  For every gate configuration we report how many clips
each arm keeps, how many of the kept clips the published verdict accepted, and the reason codes.

    PYTHONPATH=. pixi run python experiments/m03_generators.py --tag v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "results/m03/inputs"
OUT = ROOT / "results/m03"
WINDOW = 8
ARMS = ("A_flatness", "B_level", "C_mirrored_roll")


def frozen() -> dict:
    import yaml
    a = yaml.safe_load((ROOT / "configs/m01_gate_a_frozen.yaml").read_text())
    b = yaml.safe_load((ROOT / "configs/m02_gate_b_frozen.yaml").read_text())
    return {"lam_a": float(a["gate_a"]["lam_deg"]),
            "near_vertical": float(a["gate_a"]["near_vertical_deg"]),
            "tau_b": float(b["gate_b"]["tau_deg"])}


def sigma_min_of(lean: np.ndarray, tilt: np.ndarray | None = None) -> tuple[float, float]:
    l = np.asarray(lean, float)
    if tilt is None:
        tilt = np.degrees(np.arcsin(np.clip(np.linalg.norm(l, axis=1), 0, 1)))
    ok = np.isfinite(l).all(1)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    sd = l[ok].std(0)
    return float(np.degrees(sd.min())), float(np.max(tilt[ok]))


def h04() -> pd.DataFrame:
    rows = json.loads((INP / "h04_scored_main_rows.json").read_text())
    df = pd.DataFrame(rows)
    paths = np.load(INP / "h04_paths.npz")
    sig = {}
    for key in paths.files:
        if key.endswith("__lean_req_rule"):
            p = key.split("__")[0]
            lean = np.asarray(paths[key], float)
            tilt = np.asarray(paths.get(f"{p}__tilt_req_rule", np.full(len(lean), np.nan)),
                              float)
            sig[p] = sigma_min_of(lean, tilt)
    df["sigma_min"] = df.path.map(lambda p: sig.get(p, (np.nan, np.nan))[0])
    df["tilt_max"] = df.path.map(lambda p: sig.get(p, (np.nan, np.nan))[1])
    return df


def h01() -> pd.DataFrame:
    table = pd.read_csv(INP / "h01_t1_seed_variance.csv")
    specs = json.loads((INP / "h01_specs.json").read_text())
    spec_by_id = {s["id"]: s for s in specs}
    rows = []
    for _, r in table.iterrows():
        sid = r["path"]
        s = spec_by_id.get(sid)
        if s is None:
            continue
        pos = np.asarray(s["positions"], float)
        # the ANWM designed path is one waypoint per commanded step; the required lean of a
        # step is the flatness lean of the velocity change across it
        v = np.diff(pos, axis=0) / 1.0                      # 1 s per action (H01's reading)
        a = np.diff(v, axis=0)
        ax = a + np.array([0.0, 0.0, 9.81]) + 0.295 * v[1:]
        ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
        yaw = np.asarray(s["yaw"], float)[1:-1] if len(s["yaw"]) >= len(ax) else np.zeros(len(ax))
        c, sn = np.cos(yaw[:len(ax)]), np.sin(yaw[:len(ax)])
        lean = np.column_stack([c * ax[:, 0] + sn * ax[:, 1], -sn * ax[:, 0] + c * ax[:, 1]])
        tilt = np.degrees(np.arccos(np.clip(ax[:, 2], -1, 1)))
        sig, tmax = sigma_min_of(lean, tilt)
        rows.append({"clip_id": sid, "designed_deg": float(r["designed_deg"]),
                     "residual_median_deg": float(r["residual_median_deg"]),
                     "feasible_seeds": int(r["feasible_seeds"]), "seeds": int(r["seeds"]),
                     "sigma_min": sig, "tilt_max": tmax})
    return pd.DataFrame(rows)


def counts(df: pd.DataFrame, fr: dict, label_col: str, dis_col: str | None) -> dict:
    out = {}
    for cfg in ("G0_S3_only", "GA", "GB", "GAB"):
        keep = pd.Series(True, index=df.index)
        reasons = pd.Series("", index=df.index)
        if cfg in ("GA", "GAB"):
            weak = (df.sigma_min >= fr["lam_a"]) & (df.tilt_max <= fr["near_vertical"])
            keep &= weak.fillna(False)
            reasons = reasons.mask(~weak.fillna(False), "weak_signal_or_near_vertical")
        if cfg in ("GB", "GAB") and dis_col is not None:
            agree = df[dis_col] <= fr["tau_b"]
            keep &= agree.fillna(False)
            reasons = reasons.mask(~agree.fillna(False), "readers_disagree")
        sub = df[keep]
        ent = {"kept": int(len(sub)), "total": int(len(df)),
               "kept_fraction": len(sub) / max(len(df), 1)}
        for arm, g in sub.groupby(label_col):
            ent[str(arm)] = int(len(g))
        out[cfg] = ent
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    fr = frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    dis = {}
    p = ROOT / "data/m02_scores" / args.tag / "h04_shard0.jsonl"
    paths = sorted((ROOT / "data/m02_scores" / args.tag).glob("h04_shard*.jsonl"))
    if paths:
        rows = []
        for f in paths:
            rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        d = pd.DataFrame(rows)
        from src.verdict.gate_b import disagreement, gravity_gc, gravity_pf, lean_difference_deg, lean_from_gravity
        for cid, g in d.groupby("clip_id"):
            g = g.sort_values("frame_index")
            R_cb = np.eye(3)
            l_pf = lean_from_gravity(np.vstack(g.pf_gravity.values), R_cb)
            l_gc = lean_from_gravity(np.vstack(g.gc_gravity.values), R_cb)
            dis[cid] = disagreement(lean_difference_deg(l_pf, l_gc)).median

    h4 = h04()
    h4["dis_med"] = h4.clip_id.map(dis)
    h4.to_csv(OUT / "h04_clips.csv", index=False)
    res = {"frozen": fr,
           "h04": {"by_arm_all": h4.groupby("arm").size().to_dict(),
                   "accepted_by_arm_all": h4.groupby("arm").accepted.sum().to_dict(),
                   "configs": counts(h4, fr, "arm", "dis_med")}}

    h1 = h01()
    h1["dis_med"] = h1.clip_id.map(dis)
    h1.to_csv(OUT / "h01_clips.csv", index=False)
    thr = 8.87
    h1["infeasible"] = h1.residual_median_deg > thr
    res["h01"] = {"n": int(len(h1)), "infeasible_all": int(h1.infeasible.sum()),
                  "configs": counts(h1, fr, "infeasible", "dis_med")}
    (OUT / "generators.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
