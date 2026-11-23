"""Track L, L02 downstream: coherence-tier detection with each reader on Track A's edits.

Consumes ``results/l02/downstream/<model>__<flight>.json`` (written by
``remote/l_edit_reader.py``) and the frozen clip store. The clip's path side (``lean_req``,
``vel_req``, ``yaw``, ``t``) and its frozen ``depicted_conf`` come from
``data/processed/track_d/clips/.../verdict_clip.npz``; only the depicted lean is replaced by
each reader's estimate on the same frames.

Threshold: leave-one-flight-out 95th percentile of the coherence score on the other flights'
real, gate-kept clips (the frozen rule), beside the frozen absolute 1.75. Intervals:
flight-clustered bootstrap, 5,000 replicates, seed 1; 8 clusters, so descriptive.

    PYTHONPATH=. python experiments/l02_downstream.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.verdict.abstain import gate_kept_from_conf
from src.verdict.coherence import coherence
from src.verdict.fit import FitOptions, fit_gain
from src.verdict.schema import load_clip
from src.verdict.segments import frame_residual_deg

CLIPS = Path("data/processed/track_d/clips")
IN = Path("results/l02/downstream")
OUT = Path("results/l02")
GATE = __import__("yaml").safe_load(Path("configs/gate_frozen.yaml").read_text())
FLIGHTS = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
           "flight-07a-lemniscate", "flight-07p-lemniscate", "flight-13a-trackRATM",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
CONDS = ("real", "null_edit", "tilt_k0", "mirror_roll", "yaw_control")
TAU_ABS = 1.75


def _clip_rows(flight: str, model: str) -> dict[str, list[dict]]:
    p = IN / f"{model}__{flight}.json"
    if not p.exists():
        return {}
    rec = json.loads(p.read_text())
    return {c["clip_id"]: c for c in rec["clips"]}


def _rebuild(store_clip, rows: list[dict | None], idx):
    lean = np.full_like(np.asarray(store_clip.depicted_lean, float), np.nan)
    by_frame = {r["frame"]: r for r in rows if r is not None} if rows else {}
    for k, gi in enumerate(idx):
        r = by_frame.get(gi)
        if r is not None:
            lean[k] = r["lean"]
    return lean


def _stats(store_clip, lean, lag_ms):
    ft = type(store_clip)(t=store_clip.t, pos=store_clip.pos, yaw=store_clip.yaw,
                          depicted_lean=lean, depicted_conf=store_clip.depicted_conf,
                          meta=store_clip.meta, lean_req=store_clip.lean_req,
                          vel_req=store_clip.vel_req,
                          tilt_req_deg=store_clip.tilt_req_deg)
    coh = coherence(ft, lag_ms=lag_ms)
    resid = frame_residual_deg(lean, store_clip.lean_req, store_clip.yaw)
    return {"coherence": float(coh.correlation_score), "r_x": float(coh.r_x),
            "r_y": float(coh.r_y), "median_residual": float(np.nanmedian(resid)),
            "n_used": int(coh.n_used)}


def _bootstrap(rows, key, stat, n_boot=5000, seed=1):
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["flight"], []).append(r)
    flights = [f for f in by if by[f]]
    if not flights:
        return None
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(flights, size=len(flights), replace=True)
        sub = [r for f in pick for r in by[f]]
        vals.append(stat(sub))
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if not vals.size:
        return None
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def main() -> int:
    models = sorted({p.name.split("__")[0] for p in IN.glob("*__*.json")})
    if not models:
        raise SystemExit("no downstream files: pull them from awmv-l-data /l02/downstream")
    out = {"models": {}, "tau_abs": TAU_ABS}
    clips_json = json.loads((Path("data/processed/p3a/p3a_v1") / "clips.json").read_text())["clips"]
    by_flight_clips = {}
    for c in clips_json:
        by_flight_clips.setdefault(c["flight"], []).append(c)

    for model in models:
        rows_out = []
        for flight in FLIGHTS:
            data = _clip_rows(flight, model)
            for c in by_flight_clips.get(flight, []):
                cid = c["clip_id"]
                if cid not in data:
                    continue
                d = CLIPS / flight / f"{cid}__real"
                if not d.exists():
                    continue
                store = load_clip(d)
                idx = list(range(int(c["index"][0]), int(c["index"][1])))
                rec = data[cid]
                # amendment B: the new readers' real condition is the identity-warped frame,
                # so real and edits share resampling; the raw frame is a sensitivity.
                real_key = "null_edit" if model != "frozen" else "real"
                real_rows = rec["real"] if real_key == "real" else rec["conditions"][real_key]
                real_lean = _rebuild(store, real_rows, idx)
                raw_lean = _rebuild(store, rec["real"], idx)
                # lag from the real condition, used for every condition (frozen convention)
                ft = type(store)(t=store.t, pos=store.pos, yaw=store.yaw,
                                 depicted_lean=real_lean, depicted_conf=store.depicted_conf,
                                 meta=store.meta, lean_req=store.lean_req,
                                 vel_req=store.vel_req, tilt_req_deg=store.tilt_req_deg)
                lag = fit_gain(ft, FitOptions(bootstrap=False)).lag_ms
                entry = {"flight": flight, "clip_id": cid,
                         "gate_kept": bool(gate_kept_from_conf(store.depicted_conf, GATE)),
                         "lag_ms": float(lag), "real_key": real_key, "conditions": {}}
                entry["conditions"]["real"] = _stats(store, raw_lean, lag)
                entry["conditions"]["real_null"] = _stats(store, real_lean, lag)
                for cond in CONDS:
                    if cond == "real":
                        continue
                    lean = _rebuild(store, rec["conditions"][cond], idx)
                    entry["conditions"][cond] = _stats(store, lean, lag)
                rows_out.append(entry)
        # LOFO threshold on the real, gate-kept coherence score
        kept = [r for r in rows_out if r["gate_kept"]]
        per_flight_tau = {}
        for r in kept:
            others = [o["conditions"]["real_null"]["coherence"] for o in kept
                      if o["flight"] != r["flight"]
                      and np.isfinite(o["conditions"]["real_null"]["coherence"])]
            per_flight_tau[r["flight"]] = float(np.quantile(others, 0.95)) if len(others) >= 8 else float("nan")
        for r in rows_out:
            r["tau_lofo"] = per_flight_tau.get(r["flight"], float("nan"))
        # rates
        def rate(sub, cond, tau_key, tier="coherence"):
            sub = [r for r in sub if r["gate_kept"]]
            if not sub:
                return None
            hit = 0
            for r in sub:
                t = TAU_ABS if tau_key == "tau_abs" else r[tau_key]
                v = r["conditions"][cond][tier]
                if np.isfinite(t) and np.isfinite(v) and v > t:
                    hit += 1
            return {"k": hit, "n": len(sub), "rate": hit / len(sub)}
        res = {"per_clip": rows_out, "rates": {}}
        for cond in ("real_null",) + tuple(c for c in CONDS if c != "real") + ("real",):
            res["rates"][cond] = {"lofo": rate(rows_out, cond, "tau_lofo"),
                                  "abs_1.75": rate(rows_out, cond, "tau_abs")}
        sens = [r for r in rows_out if r["flight"] not in ("flight-13p-trackRATM",
                                                            "flight-16p-trackRATM")]
        res["rates_excluding_racing"] = {
            cond: {"lofo": rate(sens, cond, "tau_lofo"),
                   "abs_1.75": rate(sens, cond, "tau_abs")}
            for cond in ("real_null", "null_edit", "tilt_k0", "mirror_roll", "yaw_control")}
        # cluster bootstrap on the LOFO real false-reject and removal detection
        res["intervals"] = {
            "real_false_reject": _bootstrap(
                [r for r in rows_out if r["gate_kept"]], "coherence",
                lambda sub: np.mean([r["conditions"]["real_null"]["coherence"] > r["tau_lofo"]
                                     for r in sub if np.isfinite(r["tau_lofo"])])),
            "removal_detection": _bootstrap(
                [r for r in rows_out if r["gate_kept"]], "coherence",
                lambda sub: np.mean([r["conditions"]["tilt_k0"]["coherence"] > r["tau_lofo"]
                                     for r in sub if np.isfinite(r["tau_lofo"])])),
            "mirror_detection": _bootstrap(
                [r for r in rows_out if r["gate_kept"]], "coherence",
                lambda sub: np.mean([r["conditions"]["mirror_roll"]["coherence"] > r["tau_lofo"]
                                     for r in sub if np.isfinite(r["tau_lofo"])])),
        }
        out["models"][model] = res

    (OUT / "downstream.json").write_text(json.dumps(out, indent=1, default=float))
    for m, res in out["models"].items():
        print(f"== {m}")
        for k, v in res["rates"].items():
            print(f"  {k:12s} lofo {v['lofo']} abs {v['abs_1.75']}")
        print("  intervals", res["intervals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
