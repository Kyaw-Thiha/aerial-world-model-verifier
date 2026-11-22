"""Track K, Part 2 downstream: schema v3, thresholds, and edit detection.

Reads the LOFO model fits from ``results/k02_fit/<run>/fit.json`` (the pre-registered
selection rule picks the model), builds the model-required lean for every TII clip
without using that clip's own flight, and reports:

* J03's F0 threshold on the gate-kept real clips;
* J02's mocap-attitude anchor with its schema semantics corrected;
* detection of the edits at the matched false-reject, for the median and coherence
  tiers, with the yaw control and null edit beside every rate.

CPU only. Writes ``results/k02_downstream/<run>/``.

    PYTHONPATH=. python experiments/k02_downstream.py --fit results/k02_fit/<run>/fit.json
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.spatial.transform import Rotation

from src.airframe.kinematics import TII_FLIGHTS, load_tii_kinematics
from src.airframe.models import axis_for, unit
from src.airframe.schema_v3 import write_v3_clip
from src.depicted.clips import up_body_from_lean
from src.depicted.stat_a2 import depicted_axis_from_up_body
from src.physics.flatness import G0, angle_between_deg
from src.verdict.lean import world_to_yaw_frame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/k02_downstream"
DEV = {"flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"}
HELDOUT = {"flight-13p-trackRATM", "flight-07a-lemniscate"}
K0 = 0.295
WIN, HOP = 8, 4
FAMILIES = ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5")
CONTROLS = ("yaw_control", "null_edit")
COVERAGES = (("partial_25", 0.25), ("partial_50", 0.50), ("full", 1.0))
SIGMA_DEG, BIAS_DEG, TAU_S, FAST_SHARE, ANCHOR_DRAWS = 4.8, 2.0, 0.5, 0.5, 120


# --------------------------------------------------------------------------- inputs
def gate_keep() -> dict[str, bool]:
    raw = (ROOT / "configs/gate_frozen.yaml").read_text()
    g = json.loads("\n".join(l for l in raw.splitlines() if not l.strip().startswith("#")))
    rows = list(csv.DictReader(open(ROOT / "results/a2_gate/20261109T201449Z_298ff20/"
                                    "clip_table.csv")))
    return {r["clip_id"]: float(r["identity::S3_latitude_confidence"]) <= float(g["tau"])
            for r in rows}


def load_clips() -> dict[tuple[str, str], dict]:
    out = {}
    for f in TII_FLIGHTS:
        spec = np.load(ROOT / f"data/processed/p3a/p3a_v1/{f}.npz")
        R = Rotation.from_quat(spec["quat"]).as_matrix()
        z_all = R[:, :, 2]
        for d in sorted((ROOT / f"data/processed/track_d/clips/{f}").iterdir()):
            if not d.is_dir():
                continue
            z = np.load(d / "verdict_clip.npz")
            cond = d.name.split("__")[-1]
            i = np.clip(np.searchsorted(spec["t"], z["t"]), 0, len(spec["t"]) - 1)
            yaw = np.asarray(z["yaw"], float)
            lean_act = world_to_yaw_frame(z_all[i][:, :2], yaw)
            out[(d.name.split("__")[0], cond)] = {
                "flight": f, "clip_id": d.name.split("__")[0], "cond": cond,
                "t": np.asarray(z["t"], float), "yaw": yaw,
                "dep": np.asarray(z["depicted_lean"], float),
                "lean_req_v2": np.asarray(z["lean_req"], float),
                "vel_req_v2": np.asarray(z["vel_req"], float),
                "lean_act": lean_act,
            }
    return out


def sample_axis(k, axis: np.ndarray, t_frame: np.ndarray,
                yaw_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a native axis at frame times, then express it in the frame's yaw."""
    t = k.t
    tt = np.clip(np.asarray(t_frame, float), t[0], t[-1])
    good = (k.valid & np.isfinite(axis).all(axis=1)
            & np.isfinite(k.yaw))
    out = np.column_stack([np.interp(tt, t, axis[:, j]) for j in range(3)])
    idx = np.clip(np.searchsorted(t, tt), 1, len(t) - 1)
    bad = ~good[idx - 1] | ~good[idx]
    lo, hi = np.argmax(good), len(good) - 1 - np.argmax(good[::-1])
    bad |= (tt < t[lo]) | (tt > t[hi])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.where(np.isfinite(n) & (n > 1e-9), n, np.nan)
    out[bad] = np.nan
    lean = world_to_yaw_frame(out[:, :2], yaw_frame)
    lean[bad] = np.nan
    tilt = np.degrees(np.arccos(np.clip(out[:, 2], -1.0, 1.0)))
    tilt[bad] = np.nan
    return lean, tilt


def exact_v2_lean(k, t_frame: np.ndarray, yaw_frame: np.ndarray,
                  k_d: float = K0) -> np.ndarray:
    u = unit(k.a + np.array([0.0, 0.0, G0]) + k_d * k.v)
    return sample_axis(k, u, t_frame, yaw_frame)[0]


def v3_lean(k, model: str, params: dict, t_frame: np.ndarray,
            yaw_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return sample_axis(k, axis_for(model, params, k), t_frame, yaw_frame)


def load_params(p: dict) -> dict:
    out = dict(p)
    for key in ("W", "feat_mean", "feat_sd"):
        if key in out:
            out[key] = np.asarray(out[key], float)
    return out


# --------------------------------------------------------------------------- statistics
def frame_residual_deg(dep: np.ndarray, req: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    ax_d = depicted_axis_from_up_body(up_body_from_lean(dep), yaw)
    ax_r = depicted_axis_from_up_body(up_body_from_lean(req), yaw)
    return angle_between_deg(ax_d, ax_r)


def _windows(n: int) -> list[tuple[int, int]]:
    if n <= WIN:
        return [(0, n)]
    return [(s, s + WIN) for s in range(0, n - WIN + 1, HOP)]


def tier_values(b: dict, dep: np.ndarray, req: np.ndarray) -> tuple[float, float]:
    resid = frame_residual_deg(dep, req, b["yaw"])
    ok = np.isfinite(resid)
    med = float(np.nanmedian(resid[ok])) if ok.any() else np.nan
    vals = []
    for lo, hi in _windows(len(resid)):
        r = resid[lo:hi]
        if np.isfinite(r).sum() >= 6:
            vals.append(float(np.nanmedian(r)))
    return med, (float(max(vals)) if vals else np.nan)


def coherence_stat(b: dict, dep: np.ndarray, req: np.ndarray) -> float:
    vals = []
    for lo, hi in _windows(len(dep)):
        d, q = dep[lo:hi], req[lo:hi]
        ok = np.isfinite(d).all(1) & np.isfinite(q).all(1)
        d, q = d[ok], q[ok]
        if len(d) < 6:
            continue
        rs = []
        for j in range(2):
            if np.std(d[:, j]) < 0.01 or np.std(q[:, j]) < 0.01:
                rs = []
                break
            rs.append(float(np.corrcoef(d[:, j], q[:, j])[0, 1]))
        if rs:
            vals.append(1.0 - min(rs))
    return float(max(vals)) if vals else np.nan


def compose_partial(real: np.ndarray, edit: np.ndarray, frac: float) -> np.ndarray:
    n = len(real)
    m = max(1, int(round(frac * n)))
    a = (n - m) // 2
    out = real.copy()
    out[a:a + m] = edit[a:a + m]
    return out


# --------------------------------------------------------------------------- anchor
def dct_split(x: np.ndarray, fs: float, cutoff: float = 0.5):
    n = len(x)
    coef = dct(x - x.mean(), type=2, norm="ortho")
    k = np.arange(n) * fs / (2.0 * n)
    slow = idct(np.where(k < cutoff, coef, 0.0), type=2, norm="ortho")
    return slow, x - x.mean() - slow


def make_error(n: int, dt: float, sigma: float, tau_s: float, fast_share: float,
               rng: np.random.Generator) -> np.ndarray:
    """J02's error series without its bias; the caller adds a per-flight bias vector."""
    rho = float(np.exp(-dt / max(tau_s, 1e-6)))
    white = rng.standard_normal((n, 2))
    ar = np.zeros_like(white)
    ar[0] = white[0]
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + np.sqrt(1.0 - rho ** 2) * white[i]
    slow = np.zeros_like(ar)
    fast = np.zeros_like(ar)
    for j in range(2):
        s, f = dct_split(ar[:, j], 1.0 / dt)
        slow[:, j], fast[:, j] = s, f

    def scale(a, target):
        sd = float(np.std(a))
        return a * (np.sqrt(target) / sd) if sd > 1e-9 else a

    slow = scale(slow, (1.0 - fast_share) * sigma ** 2)
    fast = scale(fast, fast_share * sigma ** 2)
    return np.deg2rad(slow + fast)


def anchor(real: list[dict], req_key: str) -> dict:
    """J02 arm (c): mocap attitude lean + synthetic error, required = req_key."""
    rng = np.random.default_rng(1)
    fl = np.array([b["flight"] for b in real])
    hs, ts = [], []
    for _ in range(ANCHOR_DRAWS):
        # one bias vector per flight per draw
        bias = {}
        for f in sorted(set(fl)):
            ang = rng.uniform(0, 2 * np.pi)
            bias[f] = BIAS_DEG * np.array([np.cos(ang), np.sin(ang)])
        vals = []
        for b in real:
            dt = float(np.median(np.diff(b["t"])))
            err = make_error(len(b["t"]), dt, SIGMA_DEG, TAU_S, FAST_SHARE, rng)
            dep = b["lean_act"] + err + np.deg2rad(bias[b["flight"]])[None, :]
            vals.append(float(np.nanmedian(frame_residual_deg(dep, b[req_key], b["yaw"]))))
        vals = np.asarray(vals, float)
        hs.append(float(np.nanmedian(vals)))
        thr = [float(np.quantile(vals[fl != fl[i]], 0.95))
               for i in range(len(vals)) if np.isfinite(vals[i])]
        ts.append(float(np.median(thr)))
    return {"h0_median_deg": float(np.median(hs)),
            "threshold_lofo_median_deg": float(np.median(ts)),
            "h0_p10_p90_deg": [float(np.percentile(hs, 10)), float(np.percentile(hs, 90))],
            "threshold_p10_p90_deg": [float(np.percentile(ts, 10)),
                                      float(np.percentile(ts, 90))],
            "draws": ANCHOR_DRAWS}


# --------------------------------------------------------------------------- rates
def lofo_threshold_for(values: np.ndarray, flights: np.ndarray, i: int) -> float:
    fit = values[flights != flights[i]]
    fit = fit[np.isfinite(fit)]
    return float(np.quantile(fit, 0.95)) if len(fit) >= 5 else float("nan")


def tier_rows(rows: list[dict], tier: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use = [r for r in rows if r["flight"] in DEV]
    real = [r for r in use if r["cond"] == "real"]
    fl = np.array([r["flight"] for r in real])
    vals = np.array([r[tier] for r in real], float)
    thr = np.array([lofo_threshold_for(vals, fl, i) for i in range(len(vals))])
    return vals, thr, fl


def rates(rows: list[dict], tier: str) -> dict:
    vals, thr, fl = tier_rows(rows, tier)
    out = {"false_reject": f"{int(np.sum(vals > thr))}/{len(vals)}"}
    for fam in FAMILIES + CONTROLS:
        for cov, _ in COVERAGES:
            sub = [r for r in rows if r["flight"] in DEV and r["cond"] == fam
                   and r["coverage"] == cov]
            hits = tot = 0
            for r in sub:
                m = fl == r["flight"]
                if not m.any():
                    continue
                hits += float(r[tier] > thr[np.flatnonzero(m)[0]])
                tot += 1
            out[f"{fam}/{cov}"] = f"{int(hits)}/{tot}"
    return out


def matched_rates(rows: list[dict], tier: str, budget: int) -> dict:
    real = np.array([r[tier] for r in rows if r["flight"] in DEV and r["cond"] == "real"],
                    float)
    real = real[np.isfinite(real)]
    if budget <= 0:
        thr = float(real.max() + 1e-9)
    else:
        thr = float(np.sort(real)[-budget])
        thr = float(np.nextafter(thr, -np.inf))
    out = {"false_reject": f"{int(np.sum(real > thr))}/{len(real)}", "threshold": thr}
    for fam in FAMILIES + CONTROLS:
        for cov, _ in COVERAGES:
            sub = np.array([r[tier] for r in rows if r["flight"] in DEV
                            and r["cond"] == fam and r["coverage"] == cov], float)
            sub = sub[np.isfinite(sub)]
            out[f"{fam}/{cov}"] = f"{int(np.sum(sub > thr))}/{len(sub)}"
    return out


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=None)
    args = ap.parse_args()
    fit_path = Path(args.fit) if args.fit else sorted(OUT.parent.glob("k02_fit/*/fit.json"))[-1]
    fit = json.loads(Path(fit_path).read_text())
    model = fit["selection"]["selected"]
    print("selected model", model, fit["selection"], flush=True)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)

    keep = gate_keep()
    data = load_clips()
    real_ids = sorted({cid for (cid, cond) in data if cond == "real"})
    kd = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    lofo_params = {u["name"]: load_params(u["params"])
                   for u in fit["lofo"][model]["units"]}

    for (cid, cond), b in data.items():
        k = kd[b["flight"]]
        b["req_v2"] = exact_v2_lean(k, b["t"], b["yaw"])
        b["req_v3"] = v3_lean(k, model, lofo_params[b["flight"]], b["t"], b["yaw"])[0]
        b["req_lin"] = b["lean_req_v2"] + K0 * b["vel_req_v2"]

    # ---- schema v3 delivery files (full-fit parameters; new tree, v2 untouched)
    full = load_params(fit["full_fit"][model]["params"])
    v3_root = ROOT / "data/processed/track_k/clips_v3"
    manifest = ["flight,clip_id,condition,sha256,model"]
    for (cid, cond), b in sorted(data.items()):
        req = v3_lean(kd[b["flight"]], model, full, b["t"], b["yaw"])
        src = ROOT / f"data/processed/track_d/clips/{b['flight']}/{cid}__{cond}"
        dst = v3_root / b["flight"] / f"{cid}__{cond}"
        _, h = write_v3_clip(src, dst, req[0], req[1], model, full,
                             "all-8-flights (delivery copy)",
                             f"schema v3: {model} required lean at frame times, fitted on "
                             f"all eight TII flights (delivery copy; downstream uses LOFO)")
        manifest.append(f"{b['flight']},{cid},{cond},{h},{model}")
    (out / "schema_v3_manifest.csv").write_text("\n".join(manifest) + "\n")

    # ---- per-condition rows for the three required sides and both tiers
    rows = {"v2_exact": [], "v2_linear": [], "v3": []}
    for key, req_key in (("v2_exact", "req_v2"), ("v2_linear", "req_lin"), ("v3", "req_v3")):
        for (cid, cond), b in sorted(data.items()):
            if cond.startswith(("tilt_removal", "mirror_roll", "scale_")) or \
                    cond in CONTROLS:
                for cov, frac in COVERAGES:
                    real = data[(cid, "real")]
                    dep = (real["dep"] if frac >= 1.0
                           else compose_partial(real["dep"], b["dep"], frac))
                    med, win = tier_values(b, dep, b[req_key])
                    rows[key].append({"flight": b["flight"], "clip_id": cid, "cond": cond,
                                      "coverage": cov, "median": med, "window": win,
                                      "coh": coherence_stat(b, dep, b[req_key])})
            else:
                med, win = tier_values(b, b["dep"], b[req_key])
                rows[key].append({"flight": b["flight"], "clip_id": cid, "cond": cond,
                                  "coverage": "full", "median": med, "window": win,
                                  "coh": coherence_stat(b, b["dep"], b[req_key])})

    # ---- F0 thresholds
    f0 = {}
    for key in rows:
        rr = [r for r in rows[key] if r["cond"] == "real" and keep.get(r["clip_id"], False)
              and r["flight"] in DEV]
        fl = np.array([r["flight"] for r in rr])
        vals = np.array([r["median"] for r in rr], float)
        thr = np.array([lofo_threshold_for(vals, fl, i) for i in range(len(vals))])
        f0[key] = {"n": len(rr), "median_residual": float(np.nanmedian(vals)),
                   "threshold_lofo_median": float(np.nanmedian(thr))}

    # ---- anchor
    anchor_out = {}
    for key, req_key in (("v2_exact", "req_v2"), ("v3", "req_v3")):
        rr = []
        for cid in real_ids:
            b = data[(cid, "real")]
            if b["flight"] in DEV:
                rr.append(dict(b, **{req_key: b[req_key]}))
        anchor_out[key] = anchor(rr, req_key)

    # ---- detection, natural and matched
    det = {}
    for key in rows:
        det[key] = {}
        for tier, name in (("median", "median"), ("coh", "coherence")):
            nat = rates(rows[key], tier)
            det[key][name] = nat
            det[key][name]["matched"] = matched_rates(
                rows[key], tier, int(nat["false_reject"].split("/")[0]))

    doc = {"run": run, "fit_file": str(fit_path), "model": model,
           "selection": fit["selection"],
           "lofo_rms": {m: fit["lofo"][m]["pooled_rms_frame_weighted"] for m in fit["lofo"]},
           "f0": f0, "anchor": anchor_out, "detection": det,
           "population": {"gate_kept_dev_real": sum(
               1 for r in rows["v3"] if r["cond"] == "real"
               and keep.get(r["clip_id"], False) and r["flight"] in DEV),
               "dev_flights": sorted(DEV), "heldout_unscored": sorted(HELDOUT)}}
    (out / "downstream.json").write_text(json.dumps(_jsonable(doc), indent=1))
    print("F0", json.dumps(f0, indent=1))
    print("ANCHOR", json.dumps(anchor_out, indent=1))
    for key, d in det.items():
        print(key, "median FR", d["median"]["false_reject"], "coh FR",
              d["coherence"]["false_reject"])
        for fam in FAMILIES + CONTROLS:
            print("   ", fam, "matched med", d["median"]["matched"].get(fam + "/full"),
                  "coh", d["coherence"]["matched"].get(fam + "/full"))
    print("wrote", out)
    return 0


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    raise SystemExit(main())
