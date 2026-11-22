"""H04: score the SolarWM continuous-attitude campaign with the frozen verifier.

The required side is the *commanded* path (the generator's best case), differentiated at
1 kHz with the label-free window rule (0.5 / f95) as primary and the frozen 0.37 s as
secondary.  The depicted side is the frozen reader.  Tiers: coherence (primary), gain R2
band, median residual.  Abstention is the frozen rule set.

Also computed, as pre-registered:
* the yaw control (arm A, draw 0, +20 deg world-yaw warp), scored with the same statistic;
* the same-arm null (one draw's depicted lean in place of the required side);
* per-path sample spread and the oracle best-of-3.

    PYTHONPATH=. pixi run -e gpu python experiments/h04_score.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.perception.frames import camera_to_body_pitched
from src.verdict import (Clip, FitOptions, assess, coherence, depicted_lean_from_gravity,
                         fit_gain, gate_kept_from_conf)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A_flatness", "B_level", "C_mirrored_roll")


def load_rows(path: Path) -> dict:
    out = defaultdict(dict)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["clip_id"]][int(Path(r["frame"]).stem)] = r
    return out


def gain_opts(cfg: dict) -> FitOptions:
    o = cfg["options"]
    return FitOptions(drag=o["drag"], fixed_drag=o["fixed_drag"], intercept=o["intercept"],
                      lag_search=o["lag_search"],
                      lag_bounds_ms=tuple(o["lag_bounds_ms"]),
                      lag_step_ms=o["lag_step_ms"], estimator=o["estimator"],
                      smoothing_s=o["smoothing_s"],
                      confidence_weights=o["confidence_weights"],
                      sg_window_s=o["sg_window_s"], sg_polyorder=o["sg_polyorder"],
                      bootstrap=True, n_boot=o["n_boot"], seed=o["seed"],
                      min_frames=o["min_frames"])


def make_clip(sid: str, name: str, arm: str, draw: int, frame_key: str, d, rows,
              window: str, gain_cfg: dict) -> Clip:
    frames = rows[sid]
    idx = sorted(frames)
    t = d[f"{name}__t"]
    pos = d[f"{name}__pos"]
    up = np.array([frames[i]["gravity_up_cam"] for i in idx], dtype=float)
    conf = np.array([frames[i].get("latitude_confidence_mean", np.nan) for i in idx])
    R_cb = camera_to_body_pitched(0.0)
    lean = depicted_lean_from_gravity(up, R_cb)
    # the model's frames start at 0 and are 81 long; keep t aligned to the actual rows
    t = t[idx] if len(idx) == len(t) else t[:len(idx)]
    meta = {"dataset": "solarwm", "clip_id": sid, "condition": arm, "flight": name,
            "path": name, "arm": arm, "draw": draw, "frame_key": frame_key,
            "window": window}
    return Clip(t=np.asarray(t, dtype=float), pos=pos[:len(idx)], yaw=d[f"{name}__yaw"][:len(idx)],
                depicted_lean=lean, depicted_conf=conf, meta=meta,
                lean_req=d[f"{name}__lean_req_{window}"][:len(idx)],
                vel_req=d[f"{name}__vel_req_{window}"][:len(idx)],
                tilt_req_deg=d[f"{name}__tilt_req_{window}"][:len(idx)])


def score_clip(clip: Clip, gain_cfg: dict, coh_cfg: dict, gate_cfg: dict) -> dict:
    res = fit_gain(clip, gain_opts(gain_cfg))
    gate_kept = gate_kept_from_conf(clip.depicted_conf, gate_cfg)
    ab = assess(res, gain_cfg.get("abstention"), gate_kept=gate_kept)
    coh = coherence(clip, lag_ms=res.lag_ms)
    band = gain_cfg["bands"]["axis"]
    zx = abs((res.axis_gains[0] - band["center"]["gx"]) / band["scale"]["gx"])
    zy = abs((res.axis_gains[1] - band["center"]["gy"]) / band["scale"]["gy"])
    flagged_r2 = bool(max(zx, zy) > band["threshold"])
    ct, st = coh_cfg["thresholds"]["tau_correlation"], coh_cfg["thresholds"]["tau_sign"]
    flagged_c = bool(coh.correlation_score > ct)
    flagged_s = bool(coh.sign_score > st)
    tilt_dep = np.degrees(np.arcsin(np.clip(np.linalg.norm(clip.depicted_lean, axis=1), 0, 1)))
    req = clip.tilt_req_deg
    ok = np.isfinite(tilt_dep) & np.isfinite(req)
    resid = float(np.median(np.abs(tilt_dep[ok] - req[ok]))) if ok.sum() else float("nan")
    return {"clip_id": clip.clip_id, "path": clip.meta["path"], "arm": clip.meta["arm"],
            "draw": clip.meta["draw"], "frame_key": clip.meta["frame_key"],
            "testable": ab.testable, "reasons": ab.reasons, "gate_kept": gate_kept,
            "gain": res.gain, "gx": res.axis_gains[0], "gy": res.axis_gains[1],
            "drag": res.drag, "lag_ms": res.lag_ms, "r2": res.r2,
            "n_used": res.n_frames, "excitation_deg": res.excitation_deg,
            "r_x": coh.r_x, "r_y": coh.r_y, "s_x": coh.s_x, "s_y": coh.s_y,
            "corr_score": coh.correlation_score, "sign_score": coh.sign_score,
            "flagged_coh": flagged_c or flagged_s, "flagged_coh_corr": flagged_c,
            "flagged_coh_sign": flagged_s, "flagged_r2": flagged_r2,
            "median_residual_deg": resid,
            "depicted_tilt_median_deg": float(np.nanmedian(tilt_dep)),
            "required_tilt_median_deg": float(np.nanmedian(req)),
            "required_tilt_peak_deg": float(np.nanmax(req)),
            "accepted": bool(ab.testable and not (flagged_c or flagged_s))}


def parse_sid(sid: str):
    """``main_<path>_<arm>_g<k>`` or ``swap_<path>_<F#>_<arm>_g<k>`` -> parts."""
    for arm in ARMS:
        marker = f"_{arm}_g"
        if marker not in sid:
            continue
        head, tail = sid.split(marker)
        try:
            draw = int(tail)
        except ValueError:
            return None
        kind = ""
        for pre in ("main_", "swap_"):
            if head.startswith(pre):
                head, kind = head[len(pre):], pre.strip("_")
        frame_key = ""
        if kind == "swap":
            head, frame_key = head.rsplit("_", 1)
        return head, arm, draw, frame_key
    return None


def rate_block(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    test = [i for i in items if i["testable"]]
    acc = [i for i in items if i["accepted"]]
    reasons = Counter(r.split(" ")[0] for i in items for r in i["reasons"])
    return {"n": n, "testable_k": len(test), "testable_n": n,
            "accepted_k": len(acc), "accepted_n": n,
            "flag_coh_k": sum(i["flagged_coh"] for i in test),
            "flag_r2_k": sum(i["flagged_r2"] for i in test),
            "median_residual_deg": (float(np.median([i["median_residual_deg"] for i in test]))
                                    if test else float("nan")),
            "median_gx": (float(np.median([i["gx"] for i in test])) if test else float("nan")),
            "median_gy": (float(np.median([i["gy"] for i in test])) if test else float("nan")),
            "abstention_reasons": dict(reasons)}


def bootstrap_median(vals, n=5000, seed=1):
    v = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(v) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    med = np.median(v[rng.integers(0, len(v), size=(n, len(v)))], axis=1)
    return [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def null_scores(d: dict, rows: dict, names: list[str], coh_cfg: dict) -> list[dict]:
    """Same-arm null: draw i's depicted lean against draw j's depicted lean (lag 0)."""
    out = []
    pairs = [(0, 1), (1, 2), (0, 2)]
    for name in names:
        sids = [s for s in rows if s == f"main_{name}_A_flatness_g0" or
                s.startswith(f"main_{name}_A_flatness_g")]
        sids = sorted(sids)
        for i, j in pairs:
            a, b = f"main_{name}_A_flatness_g{i}", f"main_{name}_A_flatness_g{j}"
            if a not in rows or b not in rows:
                continue
            cl = make_clip(a, name, "A_flatness", i, "", d, rows, "rule", {})
            other = make_clip(b, name, "A_flatness", j, "", d, rows, "rule", {})
            cl2 = Clip(t=cl.t, pos=cl.pos, yaw=cl.yaw, depicted_lean=cl.depicted_lean,
                       depicted_conf=cl.depicted_conf, meta=cl.meta,
                       lean_req=other.depicted_lean, vel_req=np.zeros((len(cl.t), 2)))
            coh = coherence(cl2, lag_ms=0.0)
            ct, st = coh_cfg["thresholds"]["tau_correlation"], coh_cfg["thresholds"]["tau_sign"]
            out.append({"path": name, "pair": f"{i}-{j}",
                        "corr_score": coh.correlation_score,
                        "sign_score": coh.sign_score,
                        "flagged": bool(coh.correlation_score > ct or coh.sign_score > st)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--yaw-tag", default="main_yaw+20")
    ap.add_argument("--window", default="rule")
    args = ap.parse_args()
    gain_cfg = yaml.safe_load((ROOT / "configs/gain_frozen.yaml").read_text())
    coh_cfg = yaml.safe_load((ROOT / "configs/coherence_frozen.yaml").read_text())
    gate_cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    d = np.load(ROOT / "results/h04/paths/paths.npz")
    summ = {s["path"]: s for s in json.loads(
        (ROOT / "results/h04/paths/paths_summary.json").read_text())}
    rows = load_rows(ROOT / f"results/h04/depicted_{args.tag}/depicted.jsonl")
    yaw_rows = load_rows(ROOT / f"results/h04/depicted_{args.yaw_tag}/depicted.jsonl")
    names = [s["path"] for s in summ.values()]

    scored = []
    for sid in sorted(rows):
        parsed = parse_sid(sid)
        if parsed is None:
            continue
        name, arm, draw_i, frame_key = parsed
        clip = make_clip(sid, name, arm, draw_i, "", d, rows, args.window, gain_cfg)
        s = score_clip(clip, gain_cfg, coh_cfg, gate_cfg)
        s["scene"] = summ[name].get("first_frame") if name in summ else ""
        s["family"] = summ[name]["family"] if name in summ else ""
        s["requested_peak"] = summ[name]["requested_peak_bank_deg"] if name in summ else np.nan
        scored.append(s)

    out = {"tag": args.tag, "window": args.window, "n_clips": len(scored), "per_arm": {},
           "per_arm_family": {}, "per_scene": {}}
    for arm in ARMS:
        out["per_arm"][arm] = rate_block([s for s in scored if s["arm"] == arm])
    for arm in ARMS:
        for fam in sorted({s["family"] for s in scored}):
            blk = [s for s in scored if s["arm"] == arm and s["family"] == fam]
            if blk:
                out["per_arm_family"][f"{arm}/{fam}"] = rate_block(blk)
    for fk in sorted({s["scene"] for s in scored}):
        out["per_scene"][fk or "?"] = rate_block([s for s in scored if s["scene"] == fk])
    # sample spread and best-of-3, arm A
    per_path = defaultdict(list)
    for s in scored:
        if s["arm"] == "A_flatness" and s["path"] in summ:
            per_path[s["path"]].append(s)
    spread, r1, r3, flips = [], [], [], 0
    for name, items in per_path.items():
        res = [i["median_residual_deg"] for i in items if np.isfinite(i["median_residual_deg"])]
        if len(res) >= 2:
            spread.append({"path": name, "sd": float(np.std(res)),
                           "range": float(np.max(res) - np.min(res)),
                           "residuals": res})
        if len(res) >= 3:
            r1.append(res[0])
            r3.append(min(res))
            cls = [i["accepted"] for i in items]
            if (not cls[0]) and any(cls[1:]):
                flips += 1
    out["sample_spread"] = {
        "n_paths": len(spread),
        "median_sd_deg": float(np.median([s["sd"] for s in spread])) if spread else float("nan"),
        "median_range_deg": float(np.median([s["range"] for s in spread])) if spread else
        float("nan"),
        "per_path": spread}
    out["best_of_3"] = {
        "n_paths": len(r1),
        "median_r1": float(np.median(r1)) if r1 else float("nan"),
        "median_min_r3": float(np.median(r3)) if r3 else float("nan"),
        "median_reduction_deg": float(np.median(np.array(r1) - np.array(r3))) if r1 else float("nan"),
        "reduction_ci": bootstrap_median(np.array(r1) - np.array(r3)) if r1 else [],
        "paths_flipping_to_accept": int(flips)}
    # null and yaw control
    out["null_same_arm"] = null_scores(d, rows, names, coh_cfg)
    if yaw_rows:
        yaw_scored = []
        for sid in sorted(yaw_rows):
            parsed = parse_sid(sid)
            if parsed is None or parsed[1] != "A_flatness" or parsed[2] != 0:
                continue
            name = parsed[0]
            clip = make_clip(sid, name, "A_flatness", 0, "", d, yaw_rows, args.window,
                             gain_cfg)
            yaw_scored.append(score_clip(clip, gain_cfg, coh_cfg, gate_cfg))
        out["yaw_control"] = rate_block(yaw_scored)
        out["yaw_control_pairs"] = [
            {"path": y["path"], "yaw_corr": y["corr_score"], "yaw_flagged": y["flagged_coh"],
             "main_corr": next((s["corr_score"] for s in scored
                                if s["path"] == y["path"] and s["arm"] == "A_flatness"
                                and s["draw"] == 0), None),
             "main_flagged": next((s["flagged_coh"] for s in scored
                                   if s["path"] == y["path"] and s["arm"] == "A_flatness"
                                   and s["draw"] == 0), None)} for y in yaw_scored]
    (ROOT / "results/h04").mkdir(parents=True, exist_ok=True)
    (ROOT / f"results/h04/scored_{args.tag}.json").write_text(json.dumps(out, indent=1,
                                                                        default=float))
    (ROOT / f"results/h04/scored_{args.tag}_rows.json").write_text(
        json.dumps(scored, indent=1, default=float))
    for arm in ARMS:
        b = out["per_arm"][arm]
        print(f"{arm:16s} n={b['n']} testable {b['testable_k']}/{b['testable_n']} "
              f"accepted {b['accepted_k']}/{b['accepted_n']} "
              f"coh {b['flag_coh_k']}/{b['testable_k']} r2 {b['flag_r2_k']}/{b['testable_k']} "
              f"gy {b['median_gy']:.2f}")
    print(json.dumps(out["sample_spread"] | {"per_path": None}, indent=1))
    print(json.dumps(out["best_of_3"], indent=1))
    if "yaw_control" in out:
        print("yaw", json.dumps({k: out["yaw_control"][k] for k in
                                 ("n", "testable_k", "flag_coh_k", "accepted_k")}))
    if out["null_same_arm"]:
        print("null flagged", sum(n["flagged"] for n in out["null_same_arm"]),
              "/", len(out["null_same_arm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
