"""Track J, Stage 1a: per-clip calibration on the gate-kept population and its absorption.

Pre-registered in ``configs/j03_perclip_prereg.yaml``, written before any fit or threshold.
CPU only, stored clips, zero spend.

Nuisances allowed (pre-registered): a free 2-vector offset per clip, and k_d on a fixed grid in
[0, 0.40]. A gain is NEVER fitted, in any arm, and nothing is fitted inside a window.

Usage
-----
PYTHONPATH=. python experiments/j03_per_clip_calibration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.verdict.schema import load_clip
from src.verdict.segments import frame_residual_deg

TII = [
    "flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
    "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM",
]
DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse", "flight-07p-lemniscate",
       "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELDOUT = ["flight-13p-trackRATM", "flight-07a-lemniscate"]
CLIPS = Path("data/processed/track_d/clips")
OUT = Path("results/j03_perclip")
K0 = 0.295
K_GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.295, 0.35, 0.40])
HUBER_DEG = 3.0
WIN, HOP = 8, 4
DELTAS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0])
FAMILIES = ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5")
CONTROLS = ("yaw_control", "null_edit")
COVERAGES = (("partial_25", 0.25), ("partial_50", 0.50), ("full", 1.0))


# --------------------------------------------------------------------------- inputs
def gate_keep() -> dict[str, bool]:
    import csv

    raw = Path("configs/gate_frozen.yaml").read_text()
    g = json.loads("\n".join(l for l in raw.splitlines() if not l.strip().startswith("#")))
    tau = float(g["tau"])
    rows = list(csv.DictReader(open("results/a2_gate/20261109T201449Z_298ff20/clip_table.csv")))
    return {r["clip_id"]: float(r["identity::S3_latitude_confidence"]) <= tau for r in rows}


def load_clips() -> dict[tuple[str, str], dict]:
    """(clip_id, condition) -> arrays."""
    out = {}
    for f in TII:
        for d in sorted((CLIPS / f).iterdir()):
            cond = d.name.split("__")[-1]
            clip_id = d.name.split("__")[0]
            try:
                c = load_clip(d / "verdict_clip.npz")
            except Exception:
                continue
            if c.lean_req is None:
                continue
            out[(clip_id, cond)] = {
                "flight": f, "clip_id": clip_id, "cond": cond,
                "dep": np.asarray(c.depicted_lean, float),
                "req": np.asarray(c.lean_req, float),
                "vel": np.asarray(c.vel_req, float),
                "yaw": np.asarray(c.yaw, float),
                "n": len(c.t),
            }
    return out


def lean_at(b: dict, k_d: float) -> np.ndarray:
    return b["req"] + (k_d - K0) * b["vel"]


# --------------------------------------------------------------------------- fit
def huber_offset(dep: np.ndarray, req: np.ndarray, delta: float, iters: int = 6) -> np.ndarray:
    """Robust 2-vector offset: sums of a Huber loss on the residual direction."""
    r = dep - req
    ok = np.isfinite(r).all(1)
    if ok.sum() < 4:
        return np.zeros(2)
    r = r[ok]
    o = np.median(r, axis=0)
    for _ in range(iters):
        e = np.linalg.norm(r - o, axis=1)
        w = np.where(e <= delta, 1.0, delta / np.maximum(e, 1e-12))
        o = np.sum(w[:, None] * r, axis=0) / np.sum(w)
    return o


def huber_loss(dep: np.ndarray, req: np.ndarray, delta: float) -> float:
    o = huber_offset(dep, req, delta)
    e = np.linalg.norm(dep - req - o, axis=1)
    e = e[np.isfinite(e)]
    d = np.deg2rad(delta)
    h = np.where(e <= d, 0.5 * e ** 2, d * (e - 0.5 * d))
    return float(np.mean(h)) if len(h) else np.inf


def fit_clip(b: dict, k_d: float | None, fit_on: dict | None = None) -> tuple[float, np.ndarray]:
    """(k_d used, offset). ``fit_on`` lets the offset be fitted on another condition."""
    src = b if fit_on is None else fit_on
    delta = np.sin(np.deg2rad(HUBER_DEG))
    if k_d is not None:
        req = lean_at(src, k_d)
        return k_d, huber_offset(src["dep"], req, delta)
    best = None
    for k in K_GRID:
        loss = huber_loss(src["dep"], lean_at(src, k), delta)
        if best is None or loss < best[0]:
            best = (loss, k)
    k = best[1]
    return k, huber_offset(src["dep"], lean_at(src, k), delta)


# --------------------------------------------------------------------------- statistics
def windows(n: int) -> list[tuple[int, int]]:
    if n <= WIN:
        return [(0, n)]
    return [(s, s + WIN) for s in range(0, n - WIN + 1, HOP)]


def tier_values(b: dict, dep: np.ndarray, k_d: float, o: np.ndarray) -> tuple[float, float]:
    """(clip median residual, window-max residual) after removing the fitted offset."""
    resid = frame_residual_deg(dep, lean_at(b, k_d) + o, b["yaw"])
    ok = np.isfinite(resid)
    med = float(np.nanmedian(resid[ok])) if ok.any() else float("nan")
    vals = []
    for lo, hi in windows(len(resid)):
        r = resid[lo:hi]
        if np.isfinite(r).sum() >= 6:
            vals.append(float(np.nanmedian(r)))
    return med, (float(max(vals)) if vals else float("nan"))


def coherence_stat(b: dict, dep: np.ndarray) -> float:
    """Max over windows of 1 - min(r_x, r_y); invariant to an offset and to a gain."""
    req = b["req"]
    vals = []
    for lo, hi in windows(len(dep)):
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
    return float(max(vals)) if vals else float("nan")


def compose_partial(real: np.ndarray, edit: np.ndarray, frac: float) -> np.ndarray:
    n = len(real)
    m = max(1, int(round(frac * n)))
    a = (n - m) // 2
    out = real.copy()
    out[a:a + m] = edit[a:a + m]
    return out


def inject_step(b: dict, delta_deg: float, frac: float) -> np.ndarray:
    """Subtract ``delta`` degrees of tilt along the REQUIRED direction from the DEPICTED lean.

    This keeps the reader's error in the injected region (only the tilt is removed), which is
    what makes the injected interval a violation rather than a cleaner patch. It is the same
    construction as J02: ``dep + (apply_violation(req) - req)``.
    """
    req = b["req"]
    mag = np.linalg.norm(req, axis=-1, keepdims=True)
    lhat = np.divide(req, mag, out=np.zeros_like(req), where=mag > 1e-9)
    target = b["dep"] - np.sin(np.deg2rad(delta_deg)) * lhat
    m = np.linalg.norm(target, axis=-1, keepdims=True)
    target = np.where(m > 0.995, target * (0.995 / np.maximum(m, 1e-12)), target)
    return compose_partial(b["dep"], target, frac)


# --------------------------------------------------------------------------- evaluation
def lofo_tau(rows: list[dict], key: str) -> float:
    fit = np.array([r[key] for r in rows if r["flight"] not in HELDOUT])
    fit = fit[np.isfinite(fit)]
    return float(np.quantile(fit, 0.95)) if len(fit) >= 5 else float("nan")


def detect(rows: list[dict], key: str, arm: str) -> tuple[int, int]:
    """k/n flagged, thresholds leave-one-flight-out on the real gate-kept dev clips."""
    hits = tot = 0
    for r in rows:
        if r["arm"] != arm or r["flight"] in HELDOUT:
            continue
        if r["cond"] == "real":
            continue
        fit = np.array([x[key] for x in rows
                        if x["arm"] == arm and x["cond"] == "real"
                        and x["flight"] != r["flight"] and x["flight"] not in HELDOUT])
        fit = fit[np.isfinite(fit)]
        if len(fit) < 5:
            continue
        hits += float(r[key] > float(np.quantile(fit, 0.95)))
        tot += 1
    return int(hits), int(tot)


def false_reject(rows: list[dict], key: str, arm: str) -> tuple[int, int]:
    hits = tot = 0
    for r in rows:
        if r["arm"] != arm or r["flight"] in HELDOUT or r["cond"] != "real":
            continue
        fit = np.array([x[key] for x in rows
                        if x["arm"] == arm and x["cond"] == "real"
                        and x["flight"] != r["flight"] and x["flight"] not in HELDOUT])
        fit = fit[np.isfinite(fit)]
        if len(fit) < 5:
            continue
        hits += float(r[key] > float(np.quantile(fit, 0.95)))
        tot += 1
    return int(hits), int(tot)


def mdv_of(curve: dict, thresh: float = 0.5) -> float:
    ks = sorted(curve)
    for i, k in enumerate(ks):
        if curve[k] >= thresh:
            if i == 0:
                return float(k)
            a, b = ks[i - 1], k
            ra, rb = curve[a], curve[b]
            return float(k if rb <= ra else a + (thresh - ra) * (b - a) / (rb - ra))
    return float("nan")


# --------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keep = gate_keep()
    data = load_clips()
    real = {cid: b for (cid, cond), b in data.items() if cond == "real"}

    rows: list[dict] = []
    # ---------------- arm statistics on the real and edited clips
    for (cid, cond), b in sorted(data.items()):
        if not keep.get(cid, False):
            continue
        for arm in ("F0", "C1", "C2"):
            if arm == "F0":
                k, o = K0, np.zeros(2)
            elif arm == "C1":
                k, o = K0, huber_offset(b["dep"], lean_at(b, K0), np.sin(np.deg2rad(HUBER_DEG)))
            else:
                k, o = fit_clip(b, None)
            med, wmax = tier_values(b, b["dep"], k, o)
            rows.append({"arm": arm, "clip_id": cid, "flight": b["flight"], "cond": cond,
                         "coverage": "full", "k_d": float(k),
                         "clip": med, "window": wmax})
        coh = coherence_stat(b, b["dep"])
        rows.append({"arm": "R", "clip_id": cid, "flight": b["flight"], "cond": cond,
                     "coverage": "full", "k_d": None, "clip": np.nan, "window": coh})

    # ---------------- partial edits: compose on the depicted side, refit per arm
    for (cid, cond), b in sorted(data.items()):
        if not keep.get(cid, False) or cond not in FAMILIES + CONTROLS:
            continue
        base = real[cid]
        for cov_name, frac in (("partial_25", 0.25), ("partial_50", 0.50)):
            dep = compose_partial(base["dep"], b["dep"], frac)
            for arm in ("F0", "C1", "C2"):
                if arm == "F0":
                    k, o = K0, np.zeros(2)
                elif arm == "C1":
                    k, o = K0, huber_offset(dep, lean_at(base, K0), np.sin(np.deg2rad(HUBER_DEG)))
                else:
                    k, o = fit_clip(base, None, fit_on={**base, "dep": dep})
                med, wmax = tier_values(base, dep, k, o)
                rows.append({"arm": arm, "clip_id": cid, "flight": base["flight"], "cond": cond,
                             "coverage": cov_name, "k_d": float(k),
                             "clip": med, "window": wmax})
            rows.append({"arm": "R", "clip_id": cid, "flight": base["flight"], "cond": cond,
                         "coverage": cov_name, "k_d": None, "clip": np.nan,
                         "window": coherence_stat(base, dep)})

    # ---------------- tables
    table = {}
    for arm in ("F0", "C1", "C2", "R"):
        for tier in ("clip", "window"):
            if arm == "R" and tier == "clip":
                continue
            taus = []
            for r in rows:
                if (r["arm"] != arm or r["flight"] in HELDOUT or r["cond"] != "real"):
                    continue
                fit = np.array([x[tier] for x in rows
                                if x["arm"] == arm and x["cond"] == "real"
                                and x["flight"] != r["flight"] and x["flight"] not in HELDOUT])
                fit = fit[np.isfinite(fit)]
                if len(fit) >= 5:
                    taus.append(float(np.quantile(fit, 0.95)))
            ent = {"false_reject": "%d/%d" % false_reject(rows, tier, arm),
                   "tau_lofo_median_deg": float(np.median(taus)) if taus else float("nan")}
            for fam in FAMILIES + CONTROLS:
                for cov_name, frac in COVERAGES:
                    sub = [r for r in rows if r["arm"] == arm and r["cond"] == fam
                           and r["coverage"] == cov_name and r["flight"] not in HELDOUT]
                    hits = tot = 0
                    for r in sub:
                        fit = np.array([x[tier] for x in rows
                                        if x["arm"] == arm and x["cond"] == "real"
                                        and x["flight"] != r["flight"]
                                        and x["flight"] not in HELDOUT])
                        fit = fit[np.isfinite(fit)]
                        if len(fit) < 5:
                            continue
                        hits += float(r[tier] > float(np.quantile(fit, 0.95)))
                        tot += 1
                    ent[f"{fam}/{cov_name}"] = f"{int(hits)}/{int(tot)}"
            table[f"{arm}/{tier}"] = ent

    absorp = {}
    for tier in ("clip", "window"):
        a = {}
        for fam in FAMILIES:
            for cov_name, _ in COVERAGES:
                def num(arm):
                    s = table[f"{arm}/{tier}"][f"{fam}/{cov_name}"]
                    k, n = s.split("/")
                    return int(k), int(n)
                f0, c2 = num("F0"), num("C2")
                a[f"{fam}/{cov_name}"] = {
                    "F0": f0, "C2": c2,
                    "absorption_hits": f0[0] - c2[0],
                    "absorption_pp": (100.0 * f0[0] / f0[1] - 100.0 * c2[0] / c2[1])
                    if f0[1] and c2[1] else float("nan"),
                }
        absorp[tier] = a

    # ---------------- MDV sweep on injected local steps
    mdv_rows = []
    for cid, b in sorted(real.items()):
        if not keep.get(cid, False):
            continue
        for frac, cov_name in ((0.25, "partial_25"), (0.50, "partial_50")):
            for delta in DELTAS:
                dep = inject_step(b, delta, frac)
                out = {}
                for arm in ("F0", "C1", "C2"):
                    if arm == "F0":
                        k, o = K0, np.zeros(2)
                    elif arm == "C1":
                        k, o = K0, huber_offset(dep, lean_at(b, K0), np.sin(np.deg2rad(HUBER_DEG)))
                    else:
                        k, o = fit_clip(b, None, fit_on={**b, "dep": dep})
                    out[f"{arm}_clip"], out[f"{arm}_window"] = tier_values(b, dep, k, o)
                out["R_window"] = coherence_stat(b, dep)
                mdv_rows.append({"clip_id": cid, "flight": b["flight"], "coverage": cov_name,
                                 "delta": float(delta), "arm": "all", **out})
    keys = ("F0_clip", "C1_clip", "C2_clip", "F0_window", "C1_window", "C2_window", "R_window")
    curves = {}
    for key in keys:
        entry = {}
        for cov in ("partial_25", "partial_50", "both"):
            curve = {}
            for delta in DELTAS:
                hits = tot = 0
                for r in mdv_rows:
                    if r["flight"] in HELDOUT or r["delta"] != delta:
                        continue
                    if cov != "both" and r["coverage"] != cov:
                        continue
                    fit = np.array([x[key] for x in mdv_rows
                                    if x["delta"] == 0.0 and x["coverage"] == r["coverage"]
                                    and x["flight"] != r["flight"]
                                    and x["flight"] not in HELDOUT])
                    fit = fit[np.isfinite(fit)]
                    if len(fit) < 5:
                        continue
                    hits += float(r[key] > float(np.quantile(fit, 0.95)))
                    tot += 1
                curve[float(delta)] = round(hits / tot, 4) if tot else float("nan")
            entry[cov] = {"detection": curve, "mdv_deg": mdv_of(curve)}
        curves[key] = entry

    # ---------------- fit diagnostics
    kds = [r["k_d"] for r in rows if r["arm"] == "C2" and r["cond"] == "real"]
    diag = {"k_d_hist_real_C2": {str(k): int(np.sum(np.isclose(kds, k))) for k in K_GRID},
            "k_d_median_real_C2": float(np.median(kds)) if kds else float("nan"),
            "huber_delta_deg": HUBER_DEG,
            "k_d_grid": K_GRID.tolist(), "k0": K0}

    (OUT / "table.json").write_text(json.dumps(table, indent=1))
    (OUT / "absorption.json").write_text(json.dumps(absorp, indent=1))
    (OUT / "mdv.json").write_text(json.dumps(curves, indent=1))
    (OUT / "diagnostics.json").write_text(json.dumps(diag, indent=1))
    (OUT / "rows.csv").write_text(
        "arm,clip_id,flight,cond,coverage,k_d,clip,window\n" +
        "\n".join(f"{r['arm']},{r['clip_id']},{r['flight']},{r['cond']},{r['coverage']},"
                  f"{'' if r['k_d'] is None else r['k_d']},{r['clip']:.4f},{r['window']:.4f}"
                  for r in rows) + "\n")
    (OUT / "mdv_rows.csv").write_text(
        "clip_id,flight,coverage,delta," + ",".join(
            k for k in ("F0_clip", "C1_clip", "C2_clip", "F0_window", "C1_window", "C2_window",
                        "R_window")) + "\n" +
        "\n".join(",".join([r["clip_id"], r["flight"], r["coverage"], str(r["delta"])] +
                          [f"{r[k]:.4f}" for k in ("F0_clip", "C1_clip", "C2_clip", "F0_window",
                                                   "C1_window", "C2_window", "R_window")])
                  for r in mdv_rows) + "\n")

    print(json.dumps(table, indent=1))
    print(json.dumps(absorp, indent=1))
    print(json.dumps({k: {c: v[c]["mdv_deg"] for c in v} for k, v in curves.items()},
                     indent=1))
    print(json.dumps(diag, indent=1))


if __name__ == "__main__":
    main()
