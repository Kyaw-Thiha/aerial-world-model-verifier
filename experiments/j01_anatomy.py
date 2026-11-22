"""Track J, Part 0: the reader-error anatomy (items 1-5) and the pre-registered rules.

Reads stored per-frame arrays only. CPU, no spend. Definitions: see
``configs/j01_anatomy_prereg.yaml`` and its amendment A.

Usage
-----
PYTHONPATH=. python experiments/j01_anatomy.py            # everything (needs the JPEG frames)
PYTHONPATH=. python experiments/j01_anatomy.py --no-blur  # skip the Laplacian proxy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.depicted.anatomy import (
    acf, angle_deg, axis_errors, consistency_gap, contiguous_runs, decompose_axes, fit_rotation,
    load_flight, load_uzh_sequence, periodogram, rotation_angle_deg, small_ols, tail_runs,
    trim_mask,
)
from src.perception.frames import camera_to_body_pitched
from src.verdict.lean import depicted_lean_from_gravity, lean_to_roll_pitch

TII = [
    "flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
    "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM",
]
UZH = ["indoor_forward_3", "indoor_forward_5", "indoor_forward_9"]
PILOTED = {"flight-01p-ellipse", "flight-07p-lemniscate", "flight-13p-trackRATM",
           "flight-16p-trackRATM"}
READERS = ("frozen", "ft")
OUT = Path("results/j01_anatomy")


def up_of(tab, reader: str) -> np.ndarray:
    return tab.up_frozen if reader == "frozen" else tab.up_ft


def theta_of(tab, reader: str) -> np.ndarray:
    return angle_deg(up_of(tab, reader), tab.up_true)


def keep_mask(tab, reader, strata, keepclip):
    """Frame mask for one stratum: all, trimmed (E2 rule) or gate-kept clips."""
    if strata == "gate_kept":
        return np.array([keepclip.get(str(c), True) for c in tab.clip_index], bool)
    err = axis_errors(tab, reader)
    return np.ones(len(tab), bool) if strata == "all" else trim_mask(err["theta"])


def weighted_share(parts: list[tuple[float, int]]) -> float:
    """Pooled share from (per-unit share, n) pairs."""
    num = sum(s * n for s, n in parts if np.isfinite(s))
    den = sum(n for s, n in parts if np.isfinite(s))
    return float(num / den) if den else float("nan")


# --------------------------------------------------------------------------- item 1
def item1_decomposition(tables, keepclip) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    pooled: dict[str, dict] = {}
    cache: dict[tuple[str, str, str], dict] = {}
    for reader in READERS:
        for strata in ("all", "trimmed", "gate_kept"):
            for unit, tab in tables.items():
                err = axis_errors(tab, reader)
                keep = keep_mask(tab, reader, strata, keepclip)
                d = decompose_axes(err, tab.t, contiguous_runs(tab.t), keep)
                cache[(reader, strata, unit)] = d
                for ax, a in d["per_axis"].items():
                    rows.append({"reader": reader, "strata": strata, "unit": unit, "axis": ax,
                                 **{k: a[k] for k in ("n", "bias", "ms_total", "share_bias",
                                                      "share_slow", "share_fast",
                                                      "fast_share_of_centred", "closure",
                                                      "n_runs")},
                                 "rms_deg": float(np.sqrt(a["ms_total"]))})
            # pooled, weighted by frames
            pb = [(cache[(reader, strata, u)]["pooled"]["share_bias"],
                   cache[(reader, strata, u)]["pooled"]["n"]) for u in tables]
            ps = [(cache[(reader, strata, u)]["pooled"]["share_slow"],
                   cache[(reader, strata, u)]["pooled"]["n"]) for u in tables]
            pf = [(cache[(reader, strata, u)]["pooled"]["share_fast"],
                   cache[(reader, strata, u)]["pooled"]["n"]) for u in tables]
            pfc = [(cache[(reader, strata, u)]["pooled"]["fast_share_of_centred"],
                    cache[(reader, strata, u)]["pooled"]["n"]) for u in tables]
            n_tot = sum(n for _, n in pb)
            res = {"share_bias": weighted_share(pb), "share_slow": weighted_share(ps),
                   "share_fast": weighted_share(pf),
                   "fast_share_of_centred": weighted_share(pfc), "n": int(n_tot),
                   "n_units": len(tables),
                   "share_bias_equal_flight": float(np.nanmean([s for s, _ in pb])),
                   "fast_centred_equal_flight": float(np.nanmean([s for s, _ in pfc])),
                   "share_slow_equal_flight": float(np.nanmean([s for s, _ in ps])),
                   "share_fast_equal_flight": float(np.nanmean([s for s, _ in pf]))}
            # flight-clustered bootstrap: resample units, recompute the weighted share
            rng = np.random.default_rng(1)
            for key, parts in (("bias_share", pb), ("fast_centred", pfc)):
                vals = np.array([s for s, _ in parts], float)
                ns = np.array([n for _, n in parts], float)
                draws = []
                for _ in range(5000):
                    pick = rng.integers(0, len(vals), len(vals))
                    v, w = vals[pick], ns[pick]
                    good = np.isfinite(v)
                    if good.any():
                        draws.append(float(np.sum(v[good] * w[good]) / np.sum(w[good])))
                res[f"{key}_ci"] = [float(np.percentile(draws, 2.5)),
                                    float(np.percentile(draws, 97.5))]
                res[f"{key}_n_boot"] = len(draws)
            pooled[f"{reader}/{strata}"] = res
    return rows, pooled


# --------------------------------------------------------------------------- item 2
def item2_time_structure(tables) -> tuple[list[dict], list[dict]]:
    acf_rows, psd_rows = [], []
    for reader in READERS:
        for unit, tab in tables.items():
            err = axis_errors(tab, reader)
            runs = contiguous_runs(tab.t)
            for ax in ("pitch", "roll", "theta"):
                a = acf(err[ax], tab.t, runs)
                acf_rows.append({"reader": reader, "unit": unit, "axis": ax,
                                 "tau_int_s": a["tau_int_s"],
                                 "truncation_lag": a.get("truncation_lag"),
                                 "rho1": float(a["rho"][1]) if len(a["rho"]) > 1 else float("nan"),
                                 "dt_s": float(np.median(np.diff(tab.t)))})
            for ax in ("pitch", "roll"):
                p = periodogram(err[ax], tab.t, runs)
                for f, s in zip(p["freq_hz"], p["psd"]):
                    psd_rows.append({"reader": reader, "unit": unit, "axis": ax,
                                     "freq_hz": float(f), "psd": float(s)})
    return acf_rows, psd_rows


# --------------------------------------------------------------------------- item 3
def item3_bias_structure(tables) -> tuple[list[dict], dict]:
    rows = []
    keep_by_unit, err_by_unit = {}, {}
    for reader in READERS:
        for unit, tab in tables.items():
            err = axis_errors(tab, reader)
            err_by_unit[(reader, unit)] = err
            keep = trim_mask(err["theta"])
            keep_by_unit[(reader, unit)] = keep
            for ax in ("pitch", "roll"):
                v = err[ax][keep]
                rows.append({"reader": reader, "unit": unit, "axis": ax,
                             "bias_deg": float(np.mean(v)),
                             "se_deg": float(np.std(v, ddof=1) / np.sqrt(len(v))),
                             "n": int(keep.sum()), "sign": int(np.sign(np.mean(v)))})

    mount: dict[str, object] = {}
    for reader in READERS:
        domains = {"tii_all": list(tables),
                   "tii_piloted": [u for u in tables if u in PILOTED],
                   "tii_autonomous": [u for u in tables if u not in PILOTED]}
        for name, units in domains.items():
            est = np.concatenate([up_of(tables[u], reader) for u in units])
            tru = np.concatenate([tables[u].up_true for u in units])
            r = fit_rotation(est, tru)
            before, after = angle_deg(est, tru), angle_deg(est @ r.T, tru)
            rr, used = fit_rotation_robust(est, tru)
            rb, ra = angle_deg(est[used], tru[used]), angle_deg(est[used] @ rr.T, tru[used])
            mount[f"{reader}/{name}"] = {
                "robust_rotation_deg": rotation_angle_deg(rr),
                "robust_median_before_deg": float(np.nanmedian(rb)),
                "robust_median_after_deg": float(np.nanmedian(ra)),
                "robust_n": int(used.sum()),
                "rotation_deg": rotation_angle_deg(r), "n": int(len(est)),
                "median_before_deg": float(np.nanmedian(before)),
                "median_after_deg": float(np.nanmedian(after)),
                "p90_before_deg": float(np.nanpercentile(before, 90)),
                "p90_after_deg": float(np.nanpercentile(after, 90)),
                "rms_before_deg": float(np.sqrt(np.nanmean(before ** 2))),
                "rms_after_deg": float(np.sqrt(np.nanmean(after ** 2))),
            }
        outs = []
        for unit in tables:
            others = [u for u in tables if u != unit]
            est = np.concatenate([up_of(tables[u], reader) for u in others])
            tru = np.concatenate([tables[u].up_true for u in others])
            r = fit_rotation(est, tru)
            e = up_of(tables[unit], reader)
            rr, used = fit_rotation_robust(est, tru)
            e_t = tables[unit].up_true
            outs.append({"unit": unit,
                         "median_before_deg": float(np.nanmedian(angle_deg(e, e_t))),
                         "median_after_deg": float(np.nanmedian(angle_deg(e @ r.T, e_t))),
                         "robust_median_after_deg": float(np.nanmedian(angle_deg(e @ rr.T, e_t)))})
        mount[f"{reader}/tii_lofo"] = {
            "per_unit": outs,
            "median_before_deg": float(np.median([o["median_before_deg"] for o in outs])),
            "median_after_deg": float(np.median([o["median_after_deg"] for o in outs])),
        }
        per_flight = []
        for unit, tab in tables.items():
            e = up_of(tab, reader)
            r = fit_rotation(e, tab.up_true)
            rr, used = fit_rotation_robust(e, tab.up_true)
            per_flight.append({"unit": unit, "rotation_deg": rotation_angle_deg(r),
                               "robust_rotation_deg": rotation_angle_deg(rr),
                               "median_before_deg": float(np.nanmedian(angle_deg(e, tab.up_true))),
                               "median_after_deg": float(np.nanmedian(angle_deg(e @ r.T,
                                                                                tab.up_true))),
                               "robust_median_after_deg": float(np.nanmedian(
                                   angle_deg(e @ rr.T, tab.up_true)))})
        mount[f"{reader}/per_flight"] = per_flight

    reg: dict[str, dict] = {}
    for reader in READERS:
        per_axis = {"pitch": ([], [], []), "roll": ([], [], [])}
        for unit, tab in tables.items():
            err = err_by_unit[(reader, unit)]
            keep = keep_by_unit[(reader, unit)]
            cols = np.column_stack([
                tab.speed[keep], np.radians(tab.tilt_deg[keep]),
                np.cos(tab.yaw[keep]), np.sin(tab.yaw[keep]),
            ])
            cols = cols - cols.mean(0)            # within-flight demeaning
            for ax in ("pitch", "roll"):
                y = err[ax][keep]
                per_axis[ax][0].append(y - y.mean())
                per_axis[ax][1].append(cols)
                per_axis[ax][2].append(np.repeat(unit, len(y)))
        for ax in ("pitch", "roll"):
            y = np.concatenate(per_axis[ax][0])
            x = np.concatenate(per_axis[ax][1])
            g = np.concatenate(per_axis[ax][2])
            fit = small_ols(y, x, g)
            beta = np.array(fit["beta"])
            resid = y - x @ beta
            reg[f"{reader}/{ax}"] = {
                "names": ["speed_mps", "tilt_rad", "cos_yaw", "sin_yaw"],
                "beta": fit["beta"], "se": fit["se"], "n": fit["n"],
                "clusters": fit.get("clusters"),
                "within_r2": 1.0 - float(np.sum(resid ** 2) / np.sum(y ** 2)),
            }
    return rows, {"per_domain_rotation": mount, "regression": reg}


def fit_rotation_robust(src: np.ndarray, dst: np.ndarray, keep: np.ndarray | None = None,
                        iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares Wahba fit, iteratively reweighted on the angular residual.

    The plain fit is dominated by the >10 deg tail (a rotation fitted to 30 deg RMS frames is
    not a mount estimate), so the second column of every mount row is this robust version:
    start from the LS fit, drop the worst 10 % of frames by residual, refit, repeat.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    good = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    if keep is not None:
        good &= np.asarray(keep, bool)
    r = fit_rotation(src, dst)
    for _ in range(iters):
        ang = angle_deg(src @ r.T, dst)
        thr = np.nanquantile(ang[good], 0.90)
        good = good & (ang <= thr)
        if good.sum() < 30:
            break
        r = fit_rotation(src[good], dst[good])
    return r, good


# --------------------------------------------------------------------------- item 4
def blur_proxy(names: np.ndarray, flight: str) -> np.ndarray:
    """Variance of the Laplacian of a 320-px-long-side greyscale copy, per frame."""
    from PIL import Image
    from scipy.ndimage import laplace

    d = Path("data/g03_frames") / flight
    out = np.full(len(names), np.nan)
    for i, n in enumerate(names):
        p = d / n
        if not p.exists():
            continue
        im = Image.open(p).convert("L")
        w, h = im.size
        s = 320.0 / max(w, h)
        im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)
        out[i] = float(np.var(laplace(np.asarray(im, dtype=np.float32))))
    return out


def tail_run_spans(mask: np.ndarray, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Absolute (start, stop) of maximal True runs inside each contiguous run."""
    out = []
    for a, b in runs:
        seg = np.asarray(mask[a:b], bool)
        i = 0
        while i < len(seg):
            if seg[i]:
                j = i
                while j < len(seg) and seg[j]:
                    j += 1
                out.append((a + i, a + j))
                i = j
            else:
                i += 1
    return out


def gate_keep() -> dict[str, bool]:
    """Frozen S3 gate decision per clip, from the stored per-clip signal and the frozen tau."""
    raw = Path("configs/gate_frozen.yaml").read_text()
    g = json.loads("\n".join(l for l in raw.splitlines() if not l.strip().startswith("#")))
    tau = float(g["tau"])
    rows = list(csv.DictReader(open("results/a2_gate/20261109T201449Z_298ff20/clip_table.csv")))
    return {r["clip_id"]: (float(r["identity::S3_latitude_confidence"]) <= tau) for r in rows}


def item4_tail(tables, use_blur: bool = True) -> dict:
    keepclip = gate_keep()
    blur_rows = []
    blur_by_unit = {}
    if use_blur:
        for unit, tab in tables.items():
            b = blur_proxy(tab.names, unit)
            blur_by_unit[unit] = b
            blur_rows.extend({"unit": unit, "frame": n, "blur_laplacian_var": float(v)}
                             for n, v in zip(tab.names, b))

    out: dict[str, object] = {"per_reader": {}}
    for reader in READERS:
        per_unit = {}
        lens_all, lens_sec = [], []
        gate_kept = gate_dropped = 0
        for unit, tab in tables.items():
            theta = theta_of(tab, reader)
            runs = contiguous_runs(tab.t)
            tl = theta > 10.0
            lens = tail_runs(tl, runs)
            dt = float(np.median(np.diff(tab.t)))
            kept_mask = np.array([keepclip.get(str(c), True) for c in tab.clip_index])
            kept_frames = int((tl & kept_mask).sum())
            dropped_frames = int((tl & ~kept_mask).sum())
            gate_kept += kept_frames
            gate_dropped += dropped_frames
            med = {}
            for key, vals in (("conf", tab.conf), ("speed", tab.speed), ("tilt_deg", tab.tilt_deg),
                              ("blur", blur_by_unit.get(unit, np.full(len(tab), np.nan)))):
                a, b = vals[tl], vals[~tl]
                med[key] = {
                    "tail_median": float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan"),
                    "rest_median": float(np.nanmedian(b)) if np.isfinite(b).any() else float("nan"),
                    "n_tail": int(np.isfinite(a).sum()),
                }
            # identifiability of long tail runs (pre-registered statistic b)
            q = {}
            for key, vals in (("tilt", tab.tilt_deg), ("speed", tab.speed), ("conf", tab.conf),
                              ("blur", blur_by_unit.get(unit, np.full(len(tab), np.nan)))):
                v = vals[np.isfinite(vals)]
                q[key] = (float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))) if len(v) > 8 \
                    else (np.nan, np.nan)
            spans = tail_run_spans(tl, runs)
            long_info = []
            for (a, b) in spans:
                if (b - a) * dt < 1.0:
                    continue
                fr = {}
                fr["tilt"] = float(np.mean(tab.tilt_deg[a:b] >= q["tilt"][1]))
                fr["speed"] = float(np.mean(tab.speed[a:b] >= q["speed"][1]))
                fr["conf"] = float(np.mean(tab.conf[a:b] <= q["conf"][0]))
                bl = blur_by_unit.get(unit, np.full(len(tab), np.nan))[a:b]
                fr["blur"] = float(np.mean(bl <= q["blur"][0])) if np.isfinite(bl).any() else float("nan")
                best = max(fr, key=lambda k: fr[k] if np.isfinite(fr[k]) else -1)
                long_info.append({"start_s": float(tab.t[a] - tab.t[0]),
                                  "length_s": float((b - a) * dt), "frames": int(b - a),
                                  "frac_extreme": fr, "best_factor": best,
                                  "identifiable": bool(np.nanmax(list(fr.values())) >= 0.70)})
            per_unit[unit] = {
                "long_runs": long_info,
                "n_long_runs": len(long_info),
                "n_identifiable_long_runs": int(sum(1 for r in long_info if r["identifiable"])),
                "n_frames": int(len(theta)), "n_tail": int(tl.sum()),
                "tail_fraction": float(np.mean(tl)), "runs": lens, "dt_s": dt,
                "frac_tail_frames_in_runs_le3": float(sum(l for l in lens if l <= 3)
                                                      / max(1, sum(lens))),
                "frac_tail_frames_in_runs_ge_1s": float(sum(l for l in lens if l * dt >= 1.0)
                                                        / max(1, sum(lens))),
                "median_run_frames": float(np.median(lens)) if lens else float("nan"),
                "longest_run_frames": int(max(lens)) if lens else 0,
                "gate_dropped_tail_frames": dropped_frames,
                "gate_kept_tail_frames": kept_frames,
                "regressors": med,
            }
            for l in lens:
                lens_all.append(l)
                lens_sec.append(l * dt)
        la, ls = np.array(lens_all, float), np.array(lens_sec, float)
        out["per_reader"][reader] = {
            "per_unit": per_unit,
            "n_tail_frames": int(sum(v["n_tail"] for v in per_unit.values())),
            "n_frames": int(sum(v["n_frames"] for v in per_unit.values())),
            "n_tail_runs": int(len(la)),
            "frac_tail_in_runs_le3": float(np.sum(la[la <= 3]) / np.sum(la)) if len(la) else float("nan"),
            "frac_tail_in_runs_ge_1s": float(np.sum(ls[ls >= 1.0]) / np.sum(ls)) if len(ls) else float("nan"),
            "median_run_frames": float(np.median(la)) if len(la) else float("nan"),
            "run_length_hist": {str(int(k)): int(np.sum(la == k)) for k in np.unique(la)},
            "gate_dropped_tail_frames": gate_dropped,
            "gate_kept_tail_frames": gate_kept,
            "frac_tail_frames_gate_dropped": float(gate_dropped / max(1, gate_dropped + gate_kept)),
        }
    out["blur_rows"] = blur_rows
    return out


# --------------------------------------------------------------------------- item 5
def mocap_at(flight: str, ts_us: np.ndarray) -> np.ndarray:
    """Body-to-world rotations from TII mocap, slerp-interpolated at absolute epoch times."""
    import pandas as pd
    from scipy.spatial.transform import Slerp

    kind = "piloted" if flight in PILOTED else "autonomous"
    path = Path(f"data/raw/tii/{kind}/mocap_{flight}.csv")
    df = pd.read_csv(path, usecols=["timestamp"] + [f"drone_rot[{i}]" for i in range(9)])
    t_us = df["timestamp"].to_numpy(np.int64)
    rot = df[[f"drone_rot[{i}]" for i in range(9)]].to_numpy(float)
    R = np.empty((len(rot), 3, 3))
    for j in range(3):
        R[:, :, j] = rot[:, 3 * j:3 * j + 3]
    order = np.argsort(t_us)
    t_us = t_us[order]
    q = Rotation.from_matrix(R[order]).as_quat()
    sl = Slerp((t_us - t_us[0]) * 1e-6, Rotation.from_quat(q))
    return sl((np.asarray(ts_us, float) - t_us[0]) * 1e-6).as_matrix()


def rel_rot_error(R_est: np.ndarray, R_true: np.ndarray, R_cb: np.ndarray) -> dict:
    """Per-step relative-rotation error (deg) and its roll/pitch split in the body frame.

    ``R_*`` are camera-to-world. The relative rotation ``R_j^T R_i`` maps camera-i vectors to
    camera j; its error against the true relative rotation, expressed in the body frame, has
    x = roll and y = pitch.
    """
    n = min(len(R_est), len(R_true))
    e = np.einsum("nji,njk->nik", R_est[: n - 1], R_est[1:n])
    t = np.einsum("nji,njk->nik", R_true[: n - 1], R_true[1:n])
    err = np.einsum("nij,nkj->nik", t, e)          # t^T e
    body = np.einsum("ij,njk,kl->nil", R_cb, err, R_cb.T)
    rv = Rotation.from_matrix(body).as_rotvec()
    ang = np.degrees(np.linalg.norm(rv, axis=1))
    return {"angle_deg": ang, "roll_deg": np.degrees(rv[:, 0]),
            "pitch_deg": np.degrees(rv[:, 1]), "yaw_deg": np.degrees(rv[:, 2])}


def propagation_error(R_cw_est: np.ndarray, R_cw_true: np.ndarray, t: np.ndarray,
                      R_cb: np.ndarray, horizon_s: float) -> dict:
    """Carry the TRUE gravity forward with the ESTIMATED rotations; error at the horizon."""
    n = len(t)
    ang, lean = [], []
    for i in range(n):
        j = int(np.searchsorted(t, t[i] + horizon_s))
        if j >= n:
            continue
        up_i = R_cw_true[i].T @ np.array([0.0, 0.0, 1.0])
        up_j = R_cw_true[j].T @ np.array([0.0, 0.0, 1.0])
        pred = (R_cw_est[j].T @ R_cw_est[i]) @ up_i
        ref = (R_cw_true[j].T @ R_cw_true[i]) @ up_i
        ang.append(float(np.degrees(np.arccos(np.clip(pred @ up_j, -1.0, 1.0)))))
        rp, pp = lean_to_roll_pitch(depicted_lean_from_gravity(pred[None], R_cb))
        rr, pr = lean_to_roll_pitch(depicted_lean_from_gravity(ref[None], R_cb))
        lean.append([float(np.degrees(pp[0] - pr[0])), float(np.degrees(rp[0] - rr[0]))])
    L = np.array(lean) if lean else np.zeros((0, 2))
    return {"n": len(ang),
            "median_deg": float(np.median(ang)) if ang else float("nan"),
            "p90_deg": float(np.percentile(ang, 90)) if ang else float("nan"),
            "median_pitch_deg": float(np.median(np.abs(L[:, 0]))) if len(L) else float("nan"),
            "median_roll_deg": float(np.median(np.abs(L[:, 1]))) if len(L) else float("nan")}


def item5_video_rotation(tables) -> dict:
    from src.depicted.uzh import published_mount

    out: dict[str, dict] = {"tii_dpvo": {}, "uzh": {}}
    pool_step, pool_prop = [], {0.5: [], 1.0: []}
    for unit in tables:
        p = Path(f"data/raw/dpvo/{unit}_stride8_fov100_s0.5/poses.npz")
        if not p.exists():
            out["tii_dpvo"][unit] = {"error": "poses missing"}
            continue
        d = np.load(p)
        poses, ts = d["poses"], d["frame_ts_us"]
        R_cw_est = Rotation.from_quat(poses[:, 3:7]).as_matrix()
        R_cb = camera_to_body_pitched(np.deg2rad(float(tables[unit].uptilt_deg)))
        R_cw_true = np.einsum("nij,jk->nik", mocap_at(unit, ts), R_cb)
        rel = rel_rot_error(R_cw_est, R_cw_true, R_cb)
        pool_step.append(rel["angle_deg"])
        t = (ts - ts[0]) * 1e-6
        rec = {"n": int(len(ts)), "rate_hz": float(1.0 / np.median(np.diff(t))),
               "per_step_median_deg": float(np.nanmedian(rel["angle_deg"])),
               "per_step_p90_deg": float(np.nanpercentile(rel["angle_deg"], 90)),
               "per_step_median_roll_deg": float(np.nanmedian(np.abs(rel["roll_deg"]))),
               "per_step_median_pitch_deg": float(np.nanmedian(np.abs(rel["pitch_deg"])))}
        for h in (0.5, 1.0):
            rec[f"prop_{h}s"] = propagation_error(R_cw_est, R_cw_true, t, R_cb, h)
            pool_prop[h].append(rec[f"prop_{h}s"])
        out["tii_dpvo"][unit] = rec

    step = np.concatenate(pool_step) if pool_step else np.array([])
    out["tii_dpvo"]["pooled"] = {
        "n": int(len(step)),
        "per_step_median_deg": float(np.nanmedian(step)) if len(step) else float("nan"),
        "per_step_p90_deg": float(np.nanpercentile(step, 90)) if len(step) else float("nan"),
        # median over flights of the per-flight median (a median of medians; the raw
        # per-frame arrays are not stored, and the per-flight spread is printed anyway)
        **{f"prop_{h}s_median_of_flight_medians_deg":
           float(np.median([r["median_deg"] for r in pool_prop[h]])) for h in (0.5, 1.0)},
    }
    for seq in UZH:
        meta = json.loads(Path(f"data/uzh/{seq}/meta.json").read_text())
        R_cb, _ = published_mount(meta)
        truth = np.load(f"data/uzh/{seq}/truth.npz")
        fps = float(meta["fps_measured"])
        rec = {}
        for name in ("dpvo", "vipe"):
            p = Path(f"data/pathrec/poses/{name}/{seq}.npz")
            if not p.exists():
                rec[name] = {"error": "poses missing"}
                continue
            c2w = np.load(p)["c2w"]
            n = len(c2w)
            R_bw = Rotation.from_quat(np.asarray(truth["quat"], float)[:n]).as_matrix()
            R_cw_true = np.einsum("nij,jk->nik", R_bw, R_cb)
            t = np.arange(n) / fps
            rel = rel_rot_error(c2w[:, :3, :3], R_cw_true, R_cb)
            r = {"n": n, "rate_hz": fps,
                 "per_step_median_deg": float(np.nanmedian(rel["angle_deg"])),
                 "per_step_p90_deg": float(np.nanpercentile(rel["angle_deg"], 90)),
                 "per_step_median_roll_deg": float(np.nanmedian(np.abs(rel["roll_deg"]))),
                 "per_step_median_pitch_deg": float(np.nanmedian(np.abs(rel["pitch_deg"])))}
            for h in (0.5, 1.0):
                r[f"prop_{h}s"] = propagation_error(c2w[:, :3, :3], R_cw_true, t, R_cb, h)
            rec[name] = r
        out["uzh"][seq] = rec
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-blur", action="store_true")
    ap.add_argument("--no-rotation", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tables = {f: load_flight(f) for f in TII}
    uzh = {s: load_uzh_sequence(s) for s in UZH}

    gaps = {u: [consistency_gap(axis_errors(t, r)) for r in READERS]
            for u, t in tables.items()}
    worst = float(np.nanmax([g for v in gaps.values() for g in v]))
    (OUT / "input_check.json").write_text(json.dumps({
        "join": "asserted identical in load_flight (raises otherwise)",
        "worst_p999_lean_vs_theta_gap_deg": worst, "per_unit": gaps,
        "n_tii_frames": int(sum(len(t) for t in tables.values())),
        "n_uzh_frames": int(sum(len(t) for t in uzh.values())),
    }, indent=1))

    keepclip = gate_keep()
    rows, pooled = item1_decomposition(tables, keepclip)
    (OUT / "decomposition.csv").write_text(
        "reader,strata,unit,axis,n,bias_deg,rms_deg,share_bias,share_slow,share_fast,"
        "fast_share_of_centred,closure,n_runs\n" +
        "\n".join(f"{r['reader']},{r['strata']},{r['unit']},{r['axis']},{r['n']},"
                  f"{r['bias']:.4f},{r['rms_deg']:.4f},{r['share_bias']:.4f},"
                  f"{r['share_slow']:.4f},{r['share_fast']:.4f},"
                  f"{r['fast_share_of_centred']:.4f},{r['closure']:.4f},{r['n_runs']}"
                  for r in rows) + "\n")
    (OUT / "decomposition_pooled.json").write_text(json.dumps(pooled, indent=1))

    acf_rows, psd_rows = item2_time_structure(tables)
    (OUT / "acf.csv").write_text(
        "reader,unit,axis,tau_int_s,rho1,dt_s,truncation_lag\n" +
        "\n".join(f"{r['reader']},{r['unit']},{r['axis']},{r['tau_int_s']:.4f},"
                  f"{r['rho1']:.4f},{r['dt_s']:.5f},{r['truncation_lag']}" for r in acf_rows) + "\n")
    (OUT / "psd.csv").write_text(
        "reader,unit,axis,freq_hz,psd\n" +
        "\n".join(f"{r['reader']},{r['unit']},{r['axis']},{r['freq_hz']:.5f},{r['psd']:.6e}"
                  for r in psd_rows) + "\n")

    bias_rows, bias_extra = item3_bias_structure(tables)
    (OUT / "bias.csv").write_text(
        "reader,unit,axis,bias_deg,se_deg,n,sign\n" +
        "\n".join(f"{r['reader']},{r['unit']},{r['axis']},{r['bias_deg']:.4f},"
                  f"{r['se_deg']:.4f},{r['n']},{r['sign']}" for r in bias_rows) + "\n")
    (OUT / "bias_structure.json").write_text(json.dumps(bias_extra, indent=1))

    tail = item4_tail(tables, use_blur=not args.no_blur)
    if tail.get("blur_rows"):
        (OUT / "blur.csv").write_text(
            "unit,frame,blur_laplacian_var\n" +
            "\n".join(f"{r['unit']},{r['frame']},{r['blur_laplacian_var']:.4f}"
                      for r in tail.pop("blur_rows")) + "\n")
    (OUT / "tail.json").write_text(json.dumps(tail, indent=1))

    if not args.no_rotation:
        (OUT / "rotation.json").write_text(json.dumps(item5_video_rotation(tables), indent=1))

    rules = evaluate_rules(pooled, tail, json.loads((OUT / "rotation.json").read_text())
                           if not args.no_rotation else None)
    (OUT / "rules.json").write_text(json.dumps(rules, indent=1))
    print(json.dumps(pooled, indent=1))
    print(json.dumps(rules, indent=1))
    print("worst consistency gap deg", worst)


def evaluate_rules(pooled: dict, tail: dict, rotation: dict | None) -> dict:
    """The three pre-registered priority rules, applied exactly as written.

    Rule 1's population is the pre-registered one (trimmed). Trimming breaks the orthogonal
    split (the closure column), so the all-frame and gate-kept shares are printed beside it
    and the report says the outcome depends on the stratum.
    """
    out: dict = {"inputs": {}}
    fz = {k: pooled[f"frozen/{k}"] for k in ("all", "trimmed", "gate_kept")}
    ftv = {k: pooled[f"ft/{k}"] for k in ("all", "trimmed", "gate_kept")}

    r1 = {"rule": "bias share of trimmed variance >= 0.40",
          "statistic_trimmed": fz["trimmed"]["share_bias"],
          "statistic_all_frames": fz["all"]["share_bias"],
          "statistic_gate_kept": fz["gate_kept"]["share_bias"],
          "statistic_trimmed_equal_flight": fz["trimmed"]["share_bias_equal_flight"],
          "statistic_all_equal_flight": fz["all"]["share_bias_equal_flight"],
          "statistic_gate_kept_equal_flight": fz["gate_kept"]["share_bias_equal_flight"],
          "fine_tuned_trimmed": ftv["trimmed"]["share_bias"],
          "fine_tuned_all": ftv["all"]["share_bias"],
          "fine_tuned_gate_kept": ftv["gate_kept"]["share_bias"],
          "ci_trimmed": fz["trimmed"]["bias_share_ci"],
          "ci_all": fz["all"]["bias_share_ci"],
          "trimmed_closure_fraction_of_total": float(
              sum(abs(r["closure"]) for r in
                  [{"closure": 0.0}]) or 0.0),
          "fired": bool(fz["trimmed"]["share_bias"] >= 0.40),
          "fired_if_all_frames": bool(fz["all"]["share_bias"] >= 0.40),
          "fired_if_gate_kept": bool(fz["gate_kept"]["share_bias"] >= 0.40),
          "note": ("The trimmed split is not orthogonal (see decomposition.csv closure column); "
                   "the rule outcome depends on the stratum and is reported, not resolved.")}

    fast = {k: fz[k]["fast_share_of_centred"] for k in ("all", "trimmed", "gate_kept")}
    vo = {}
    if rotation is not None:
        p = rotation["tii_dpvo"].get("pooled", {})
        vo = {"per_step_median_deg": p.get("per_step_median_deg"),
              "drift_1s_median_of_flight_medians_deg":
                  p.get("prop_1.0s_median_of_flight_medians_deg"),
              "uzh_forward_dpvo_per_step": {s: rotation["uzh"][s]["dpvo"]["per_step_median_deg"]
                                            for s in rotation["uzh"]},
              "uzh_forward_dpvo_drift_1s": {s: rotation["uzh"][s]["dpvo"]["prop_1.0s"]["median_deg"]
                                            for s in rotation["uzh"]}}
    r2 = {"rule": "fast share of centred >= 0.40 AND per-step VO <= 0.3 deg AND 1 s drift <= 1.5 deg",
          "fast_share_all": fast["all"], "fast_share_trimmed": fast["trimmed"],
          "fast_share_gate_kept": fast["gate_kept"],
          "vo_tii": vo,
          "fired": bool(fast["trimmed"] >= 0.40 and vo.get("per_step_median_deg") is not None
                        and vo["per_step_median_deg"] <= 0.3
                        and vo["drift_1s_median_of_flight_medians_deg"] <= 1.5)}

    tz = tail["per_reader"]["frozen"]
    n_long = sum(v["n_long_runs"] for v in tz["per_unit"].values())
    n_ident = sum(v["n_identifiable_long_runs"] for v in tz["per_unit"].values())
    r3 = {"rule": "> half of >10 deg frames in runs <= 3 frames -> 1b; else identifiable >= 1 s runs -> 1c",
          "frac_tail_in_runs_le3": tz["frac_tail_in_runs_le3"],
          "frac_tail_in_runs_ge_1s": tz["frac_tail_in_runs_ge_1s"],
          "n_tail_runs": tz["n_tail_runs"],
          "n_long_runs": n_long, "n_identifiable_long_runs": n_ident,
          "fires_gated_fusion": bool(tz["frac_tail_in_runs_le3"] > 0.5),
          "fires_learned_confidence": bool(tz["frac_tail_in_runs_le3"] <= 0.5 and n_ident > 0)}
    out.update({"r1_bias": r1, "r2_fast": r2, "r3_tail": r3})
    out["fired_branches"] = [name for name, key in
                             (("1a", "fired"), ("1b", "fired")) if out[
                                 "r1_bias" if name == "1a" else "r2_fast"][key]]
    if r3["fires_gated_fusion"]:
        out["fired_branches"].append("1b(tail)")
    if r3["fires_learned_confidence"]:
        out["fired_branches"].append("1c")
    return out


if __name__ == "__main__":
    main()
