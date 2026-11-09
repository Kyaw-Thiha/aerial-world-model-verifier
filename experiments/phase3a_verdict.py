"""Phase 3a verdict test: real clips vs counterfactual negatives on TII.

Reads the frame/clip/edit specs (``experiments/phase3a_make_specs.py``) and the GeoCalib
outputs for each condition (``remote/a_geocalib_p3.py``), scores every clip under the
pre-registered rule, and reports Gate 3: held-out acceptance, rejection per negative
family, AUROC, the minimum-detectable-violation curve, monotonicity and the controls.

Nothing here may change a threshold or a rule: they are read from
``configs/verdict_preregistration.yaml`` and the run records its hash.

    pixi run -e gpu python experiments/phase3a_verdict.py --tag p3a_v1 \\
        --geocalib results/phase3a_verdict/geocalib_p3a_v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from src.score import (auroc, auroc_detect, conformal_quantile, score_clip, spearman)
from src.score.verdict import bootstrap_paired, wilson_interval
from src.score.prereg import verify_prereg
from src.score.verdict import clip_index, speed_bin

ROOT = Path(__file__).resolve().parents[1]
NEGATIVES = ["tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll", "yaw_control"]
FAMILY = {"tilt_k0": "tilt_removal", "tilt_k05": "tilt_scaling",
          "tilt_k15": "tilt_scaling", "mirror_roll": "mirrored_roll",
          "yaw_control": "control"}
TRACK_A_FLIGHTS = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}


def is_trackratm(r: dict) -> bool:
    """trackRATM by the flight's track label (the id set is a fallback for old records)."""
    return r.get("track") == "trackRATM" or r.get("flight") in TRACK_A_FLIGHTS


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["frame"]] = row
    return out


def align(spec: np.lib.npyio.NpzFile, rows: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """Up-in-camera and per-frame validity, in spec order (missing frames are NaN)."""
    names = [str(n) for n in spec["names"]]
    up = np.full((len(names), 3), np.nan)
    valid = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        r = rows.get(n)
        if r is not None:
            up[i] = r["gravity_up_cam"]
            valid[i] = float(r.get("valid_frac", 1.0)) >= 0.5
    return up, valid


def flight_arrays(spec: np.lib.npyio.NpzFile) -> dict:
    R_bw = Rotation.from_quat(spec["quat"]).as_matrix()
    yaw = Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0]
    return {"t": spec["t"], "implied": spec["implied_axis"], "yaw": yaw,
            "tilt_implied": spec["tilt_implied_deg"], "speed": spec["speed"],
            "uptilt": float(spec["uptilt_deg"]), "names": [str(n) for n in spec["names"]]}


def score_flight(spec_path: Path, cond_rows: dict[str, dict[str, dict]], clips: list[dict],
                 prereg: dict, ablate: str = "both") -> list[dict]:
    """Score every clip of one flight under every available condition."""
    spec = np.load(spec_path, allow_pickle=False)
    fa = flight_arrays(spec)
    safe = spec_path.name.replace(".npz", "")
    out = []
    for clip in clips:
        if clip["flight"] != safe:
            continue
        idx = clip_index(clip)
        sub = {k: (v[idx] if k not in ("uptilt", "names") else v) for k, v in fa.items()}
        for cond, rows in cond_rows.items():
            if rows is None:
                continue
            up_all, valid_all = align(spec, rows)
            up, valid = up_all[idx], valid_all[idx]
            nu = None
            if ablate != "both":
                from src.score.verdict import fit_nuisances
                m = np.isfinite(up).all(axis=1) & (sub["tilt_implied"] >= 10.0)
                uov = 0.0 if ablate in ("none", "no_uptilt") else None
                lov = 0.0 if ablate in ("none", "no_lag") else None
                nu = fit_nuisances(up, sub["implied"], sub["yaw"], sub["t"], fa["uptilt"],
                                   m, prereg, lag_override=lov,
                                   uptilt_offset_override=uov)
            r = score_clip(up, sub["implied"], sub["yaw"], sub["t"], fa["uptilt"],
                           sub["tilt_implied"], valid, prereg, nuisances=nu)
            r.update({"clip_id": clip["clip_id"], "flight": clip["flight"],
                      "track": clip["track"], "control": clip["control"],
                      "condition": cond, "speed": clip["speed_median"],
                      "bin": speed_bin(clip["speed_median"], prereg["thresholds"]["speed_bins_mps"]),
                      "tilt_implied_median": float(np.nanmedian(sub["tilt_implied"])),
                      "valid_frac_mean": float(np.mean(valid)),
                      "edit_angle_deg": float(np.median(spec[f"angle_{cond}"][idx]))
                      if f"angle_{cond}" in spec.files else 0.0})
            out.append(r)
    return out


def _bin_groups(fit: list[dict], bins: list[list[float]], min_n: int,
                alpha: float = 0.05) -> list[dict]:
    """Merge bins by the pre-registered rule: a bin under ``min_n`` joins the one below.

    Bins are ordered low speed first; the lowest bin has no neighbour below, so it joins
    the one above. The merge chain is returned so the report can show which bins the
    data actually supported.
    """
    keys = [f"{lo:g}-{hi:g}" for lo, hi in bins]
    counts = {k: sum(1 for r in fit if r["bin"] == k) for k in keys}
    groups = [{"label": k, "bins": [k], "n": counts[k]} for k in keys]
    i = 0
    while i < len(groups):
        if groups[i]["n"] >= min_n or len(groups) == 1:
            i += 1
            continue
        if i == 0:                      # lowest bin: merge upward
            groups[0] = {"label": f"{groups[0]['label']}+{groups[1]['label']}",
                         "bins": groups[0]["bins"] + groups[1]["bins"],
                         "n": groups[0]["n"] + groups[1]["n"]}
            del groups[1]
        else:                           # merge into the neighbour below
            groups[i - 1] = {"label": groups[i - 1]["label"],
                             "bins": groups[i - 1]["bins"] + groups[i]["bins"],
                             "n": groups[i - 1]["n"] + groups[i]["n"]}
            del groups[i]
        i = max(0, i - 1)
    for g in groups:
        vals = [r["score"] for r in fit if r["bin"] in g["bins"]]
        g["threshold"] = conformal_quantile(vals, alpha) if vals else float("nan")
    return groups


def thresholds_for(records: list[dict], prereg: dict, holdout: str | None,
                   policy: str = "non_trackratm") -> dict:
    """Conformal thresholds from the fit flights (all but ``holdout``), per speed bin.

    ``policy`` is the pre-registered ``threshold_fit_flights``: ``non_trackratm`` keeps
    the three estimator-failure flights out of the fit (amendment 1) while still scoring
    them; ``all_flights`` is reported as the alternative.
    """
    th = prereg["thresholds"]
    bins = th["speed_bins_mps"]
    alpha = th["alpha"]
    min_n = 25
    fit = [r for r in records if r["condition"] == "identity" and r["testable"]
           and (holdout is None or r["flight"] != holdout)
           and (policy == "all_flights"
                or (policy == "all_but_13a" and r["flight"] != "flight-13a-trackRATM")
                or (policy not in ("all_flights", "all_but_13a") and not is_trackratm(r)))]
    groups = _bin_groups(fit, bins, min_n, alpha) if fit else []
    per_bin = {g["label"]: {"n": g["n"], "bins": g["bins"], "threshold": g["threshold"]}
               for g in groups}
    pooled = {"n": len(fit),
              "threshold": conformal_quantile([r["score"] for r in fit], alpha)
              if fit else float("nan")}
    return {"per_bin": per_bin, "pooled": pooled,
            "merge_rule": f"bin under {min_n} clips merges into the neighbour below",
            "raw_counts": {f"{lo:g}-{hi:g}": sum(1 for r in fit if r["bin"] == f"{lo:g}-{hi:g}")
                           for lo, hi in bins}}


def threshold_lookup(th: dict, bin_key: str) -> float:
    """Threshold for a clip's speed bin, via the merged groups."""
    for info in th["per_bin"].values():
        if bin_key in info["bins"]:
            t = info["threshold"]
            if t is not None and np.isfinite(t):
                return t
            break
    return th["pooled"]["threshold"]


def evaluation(records: list[dict], prereg: dict,
               policy: str | None = None) -> dict:
    """Held-out acceptance, rejection, AUROC, MDV, Spearman, controls."""
    policy = policy or prereg["thresholds"].get("threshold_fit_flights", "all_flights")
    flights = sorted({r["flight"] for r in records})
    # keyed by clip, not by flight: keying by (flight, condition) keeps only the last
    # clip of each flight and makes every positive the same number
    real = {(r["flight"], r["clip_id"]): r for r in records if r["condition"] == "identity"}
    per_clip_threshold = {}
    heldout_accept = []
    folded = []
    uncertified = 0
    for f in flights:
        th = thresholds_for(records, prereg, holdout=f, policy=policy)
        for r in records:
            if r["flight"] == f and r["condition"] == "identity" and r["testable"]:
                t = threshold_lookup(th, r["bin"])
                if not np.isfinite(t):
                    # conformal cannot certify a 95th percentile from too few fit clips
                    uncertified += 1
                    continue
                heldout_accept.append(r["score"] <= t)
                per_clip_threshold[(f, r["clip_id"])] = t
        folded.append({"flight": f, "thresholds": th})
    pooled_th = thresholds_for(records, prereg, holdout=None, policy=policy)
    pooled_accept = [r["score"] <= threshold_lookup(pooled_th, r["bin"])
                     for r in records if r["condition"] == "identity" and r["testable"]
                     and np.isfinite(threshold_lookup(pooled_th, r["bin"]))]

    acc_ci = wilson_interval(int(np.sum(heldout_accept)), len(heldout_accept)) \
        if heldout_accept else (float("nan"), float("nan"))
    res = {"heldout_acceptance": float(np.mean(heldout_accept)) if heldout_accept else None,
           "heldout_acceptance_wilson95": acc_ci,
           "heldout_uncertified": uncertified,
           "heldout_n": len(heldout_accept),
           "pooled_acceptance": float(np.mean(pooled_accept)) if pooled_accept else None,
           "pooled_thresholds": pooled_th, "fold_thresholds": folded,
           "families": {}, "mdv": {}, "ablations": {}}

    # per-family rejection and AUROC against the paired real clip
    for cond in NEGATIVES:
        neg = [r for r in records if r["condition"] == cond and r["testable"]]
        pairs = [(real[(r["flight"], r["clip_id"])], r) for r in neg
                 if (r["flight"], r["clip_id"]) in real
                 and real[(r["flight"], r["clip_id"])]["testable"]]
        if not pairs:
            res["families"][cond] = {"n": 0}
            continue
        pos = np.array([p[0]["score"] for p in pairs])
        sco = np.array([p[1]["score"] for p in pairs])
        thr = np.array([per_clip_threshold.get((p[1]["flight"], p[1]["clip_id"]), np.nan)
                        for p in pairs])
        ok = np.isfinite(thr)
        res["families"][cond] = {
            "family": FAMILY[cond], "n": int(ok.sum()),
            "rejection_rate": float(np.mean(sco[ok] > thr[ok])) if ok.any() else None,
            "auroc": auroc_detect(pos[ok], sco[ok]) if ok.any() else None,
            "score_median_real": float(np.median(pos[ok])) if ok.any() else None,
            "score_median_negative": float(np.median(sco[ok])) if ok.any() else None,
            "spearman_vs_tilt": spearman(sco[ok], [p[1]["tilt_implied_median"]
                                                   for p, k in zip(pairs, ok) if k]),
        }
        grp = np.array([p[1]["flight"] for p, k in zip(pairs, ok) if k])
        boots = bootstrap_paired(pos[ok], sco[ok], grp)
        k_ok = int(np.sum(sco[ok] > thr[ok]))
        res["families"][cond].update({
            "auroc_ci95": boots["auroc_ci"], "rejection_ci95": boots["rejection_ci"],
            "n_flights": boots.get("n_groups"),
            "rejection_wilson95": wilson_interval(k_ok, int(ok.sum())),
        })
        # minimum detectable violation: rejection vs the injected tilt change
        if cond.startswith("tilt_k"):
            k = {"tilt_k0": 0.0, "tilt_k05": 0.5, "tilt_k15": 1.5}[cond]
            mags = np.array([abs(1.0 - k) * p[1]["tilt_implied_median"]
                             for p, okk in zip(pairs, ok) if okk])
            if len(mags):
                res["mdv"][cond] = {"k": k, "n": int(len(mags)),
                                    "violation_median_deg": float(np.median(mags)),
                                    "rejection_rate": float(np.mean(sco[ok] > thr[ok]))}

    # pooled AUROC over the tilt families (one score per negative clip)
    pos, neg_scores, names = [], [], []
    for cond in ("tilt_k0", "tilt_k05", "tilt_k15"):
        for r in records:
            if r["condition"] == cond and r["testable"]:
                p = real.get((r["flight"], r["clip_id"]))
                if p is not None and p["testable"]:
                    pos.append(p["score"])
                    neg_scores.append(r["score"])
                    names.append(cond)
    res["family_auroc_pooled"] = auroc_detect(np.array(pos), np.array(neg_scores)) if pos else None
    if pos:
        # monotonicity: score against the injected tilt change, pooled over the three k
        viol = []
        for cond in ("tilt_k0", "tilt_k05", "tilt_k15"):
            k = {"tilt_k0": 0.0, "tilt_k05": 0.5, "tilt_k15": 1.5}[cond]
            for r in records:
                if r["condition"] == cond and r["testable"]:
                    p = real.get((r["flight"], r["clip_id"]))
                    if p is not None and p["testable"]:
                        viol.append(abs(1.0 - k) * r["tilt_implied_median"])
        res["spearman_score_vs_injected_deg"] = spearman(np.array(neg_scores), np.array(viol))

    # separate report for trackRATM flights
    for label, sel in (("trackratm", is_trackratm),
                       ("other", lambda r: not is_trackratm(r))):
        acc = [r["score"] <= per_clip_threshold.get((r["flight"], r["clip_id"]), np.nan)
               for r in records if r["condition"] == "identity" and r["testable"]
               and sel(r) and np.isfinite(
                   per_clip_threshold.get((r["flight"], r["clip_id"]), np.nan))]
        res[f"heldout_acceptance_{label}"] = float(np.mean(acc)) if acc else None
        res[f"n_{label}"] = len(acc)
    res["diagnostics"] = diagnostics(records)
    res["gate3"] = {
        "real_acceptance_at_least_0.95": res.get("heldout_acceptance") is not None
        and res["heldout_acceptance"] >= 0.95,
        "auroc_above_0.9_tilt_families": all(
            (res["families"].get(c, {}).get("auroc") or 0.0) > 0.9
            for c in ("tilt_k0", "tilt_k05", "tilt_k15")),
        "control_accepted": (res["families"].get("yaw_control", {}).get("rejection_rate")
                             or 0.0) <= 0.10,
    }
    mdv = res["mdv"]
    if mdv:
        pts = sorted((v["violation_median_deg"], v["rejection_rate"]) for v in mdv.values())
        hit = [x for x, y in pts if y >= 0.8]
        res["mdv_smallest_rejected80_deg"] = float(min(hit)) if hit else None
    return res


def identity_vs_phase1(ident_rows: dict, phase1_dir: Path) -> dict:
    """The identity warp must reproduce the Phase 1 run, which never saw a warp.

    Same frames, same estimator, same stride; the only difference is a resample that is
    mathematically the identity. A large disagreement here would mean the edit pipeline
    changed the input, not the physics.
    """
    out = {}
    for flight, rows in ident_rows.items():
        ref_path = phase1_dir / f"{flight}.jsonl"
        if rows is None or not ref_path.exists():
            continue
        ref = load_jsonl(ref_path)
        common = [n for n in rows if n in ref]
        if not common:
            continue
        a = np.array([rows[n]["gravity_up_cam"] for n in common])
        b = np.array([ref[n]["gravity_up_cam"] for n in common])
        ang = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1)
                                           / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)),
                                           -1.0, 1.0)))
        out[flight] = {"n": len(common), "median_deg": float(np.median(ang)),
                       "p90_deg": float(np.percentile(ang, 90)),
                       "frac_gt_1deg": float(np.mean(ang > 1.0))}
    return out


def artifact_floor(spec_paths: dict, ident_rows: dict, yaw_rows: dict) -> dict:
    """How much a pure rotation about world up moves the depicted tilt, per frame.

    A rotation about up cannot change the tilt, so any movement here is the estimator's
    view dependence (plus resampling). It sets the floor under the minimum detectable
    violation, and it is the number that explains a flagged yaw control: the control is
    flagged when this floor is comparable to the threshold.
    """
    from src.perception.frames import camera_to_body_pitched

    out = {}
    for flight, rows in ident_rows.items():
        if rows is None or flight not in yaw_rows or yaw_rows[flight] is None:
            continue
        spec = np.load(spec_paths[flight], allow_pickle=False)
        theta = float(spec["uptilt_deg"])
        z_body_cam = camera_to_body_pitched(np.deg2rad(theta)).T @ np.array([0.0, 0.0, 1.0])
        names = [str(n) for n in spec["names"]]
        diffs, angles = [], []
        for n in names:
            a, b = rows.get(n), yaw_rows[flight].get(n)
            if a is None or b is None:
                continue
            ua = np.asarray(a["gravity_up_cam"], float)
            ub = np.asarray(b["gravity_up_cam"], float)
            ta = np.degrees(np.arccos(np.clip(ua @ z_body_cam / np.linalg.norm(ua), -1, 1)))
            tb = np.degrees(np.arccos(np.clip(ub @ z_body_cam / np.linalg.norm(ub), -1, 1)))
            diffs.append(abs(tb - ta))
            angles.append(float(b.get("angle_deg", 0.0)))
        diffs = np.asarray(diffs)
        angles = np.asarray(angles)
        m = angles > 10.0
        out[flight] = {"n": int(len(diffs)),
                       "median_abs_tilt_change_deg": float(np.median(diffs)) if len(diffs) else None,
                       "p90_abs_tilt_change_deg": float(np.percentile(diffs, 90)) if len(diffs) else None,
                       "n_rotated": int(m.sum()),
                       "median_abs_tilt_change_rotated_deg":
                           float(np.median(diffs[m])) if m.any() else None,
                       "corr_with_rotation": float(np.corrcoef(diffs, angles)[0, 1])
                       if len(diffs) > 3 else None}
    return out


def diagnostics(records: list[dict]) -> dict:
    """Gate 3's diagnosis block: where a failure would come from, computed in one pass."""
    real = [r for r in records if r["condition"] == "identity" and r["testable"]]
    out: dict = {"real_by_speed_bin": {}, "per_condition": {}, "real_by_flight": {}}
    for f in sorted({r["flight"] for r in real}):
        vals = [r["score"] for r in real if r["flight"] == f]
        out["real_by_flight"][f] = {"n": len(vals), "median": float(np.median(vals)),
                                    "p90": float(np.percentile(vals, 90)),
                                    "max": float(np.max(vals))}
    for b in sorted({r["bin"] for r in real}):
        vals = [r["score"] for r in real if r["bin"] == b]
        out["real_by_speed_bin"][b] = {"n": len(vals), "median": float(np.median(vals)),
                                       "p90": float(np.percentile(vals, 90))}
    for cond in ["identity"] + NEGATIVES:
        sel = [r for r in records if r["condition"] == cond and r["testable"]]
        if not sel:
            continue
        ups = [r["nuisances"]["uptilt_offset_deg"] for r in sel if r.get("nuisances", {}).get("ok")]
        lags = [r["nuisances"]["lag_ms"] for r in sel if r.get("nuisances", {}).get("ok")]
        out["per_condition"][cond] = {
            "n": len(sel),
            "score_median": float(np.median([r["score"] for r in sel])),
            "uptilt_offset_median_deg": float(np.median(ups)) if ups else None,
            "uptilt_offset_at_bound_frac": float(np.mean([abs(u) >= 6.999 for u in ups]))
            if ups else None,
            "lag_median_ms": float(np.median(lags)) if lags else None,
            "lag_at_bound_frac": float(np.mean([abs(x) >= 149.0 for x in lags]))
            if lags else None,
            "abstain_frac": float(np.mean([not r["testable"] for r in records
                                           if r["condition"] == cond])),
            "score_vs_valid_frac_spearman": spearman(
                [r["score"] for r in sel], [r.get("valid_frac_mean", 1.0) for r in sel]),
        }
    return out


def quick_delays(spec_paths: dict, ident_rows: dict, clips: list[dict], prereg: dict,
                 delays_ms=(100, 200, 300)) -> dict:
    """Delay negatives scored directly (no pixel edit: the same estimates shifted)."""
    per_delay: dict[str, list[float]] = {str(d): [] for d in delays_ms}
    for flight, rows in ident_rows.items():
        if rows is None:
            continue
        spec = np.load(spec_paths[flight], allow_pickle=False)
        fa = flight_arrays(spec)
        up_all, valid_all = align(spec, rows)
        for clip in clips:
            if clip["flight"] != flight:
                continue
            idx = clip_index(clip)
            up, valid = up_all[idx], valid_all[idx]
            t, imp, yaw = fa["t"][idx], fa["implied"][idx], fa["yaw"][idx]
            tilt = fa["tilt_implied"][idx]
            base = score_clip(up, imp, yaw, t, fa["uptilt"], tilt, valid, prereg)
            if not base["testable"]:
                continue
            for d in delays_ms:
                r = score_clip(up, imp, yaw, t, fa["uptilt"], tilt, valid, prereg,
                               nuisances={"ok": True, "loss": float("nan"),
                                          "uptilt_offset_deg": base["nuisances"]["uptilt_offset_deg"],
                                          "lag_ms": float(d)})
                if r["testable"]:
                    per_delay[str(d)].append(r["score"])
    return {k: {"n": len(v), "score_median": float(np.median(v)) if v else None}
            for k, v in per_delay.items()}


def quick_retime(spec_paths: dict, ident_rows: dict, clips: list[dict], prereg: dict,
                 rates=(0.5, 2.0)) -> dict:
    """Analysis-level negative: the frames played at 0.5x / 2x against the path.

    Frame i keeps its pixels but the path it is compared with is re-indexed as
    ``t0 + (t_i - t0) / rate``, which is what retiming the video does to the
    depicted-vs-implied alignment. No pixel edit, so the estimator is unchanged.
    """
    out: dict[str, list[float]] = {str(r): [] for r in rates}
    for flight, rows in ident_rows.items():
        if rows is None:
            continue
        spec = np.load(spec_paths[flight], allow_pickle=False)
        fa = flight_arrays(spec)
        up_all, valid_all = align(spec, rows)
        for clip in clips:
            if clip["flight"] != flight:
                continue
            idx = clip_index(clip)
            up, valid = up_all[idx], valid_all[idx]
            t, yaw = fa["t"][idx], fa["yaw"][idx]
            imp, tilt = fa["implied"][idx], fa["tilt_implied"][idx]
            base = score_clip(up, imp, yaw, t, fa["uptilt"], tilt, valid, prereg)
            if not base["testable"]:
                continue
            t0 = t[0]
            for rate in rates:
                tq = t0 + (t - t0) / rate
                imp_r = np.column_stack([np.interp(tq, t, imp[:, k]) for k in range(3)])
                tilt_r = np.degrees(np.arccos(np.clip(imp_r[:, 2], -1, 1)))
                r = score_clip(up, imp_r, yaw, t, fa["uptilt"], tilt_r, valid, prereg)
                if r["testable"]:
                    out[str(rate)].append(r["score"])
    return {k: {"n": len(v), "score_median": float(np.median(v)) if v else None}
            for k, v in out.items()}


def plots(out: Path, records: list[dict], ev: dict) -> None:
    real = np.array([r["score"] for r in records if r["condition"] == "identity" and r["testable"]])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cond in ["identity"] + NEGATIVES:
        vals = np.array([r["score"] for r in records if r["condition"] == cond and r["testable"]])
        if len(vals) == 0:
            continue
        xs = np.sort(vals)
        ax.plot(xs, np.linspace(0, 1, len(xs)), label=cond)
    ax.set_xlabel("clip score (median residual, deg)")
    ax.set_ylabel("empirical CDF")
    ax.set_xlim(0, min(90, np.percentile(np.concatenate(
        [np.array([r["score"] for r in records if r["testable"]])]), 99) * 1.1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "score_cdf.png", dpi=140)
    plt.close(fig)

    # ROC per negative family, paired clip by clip
    real_by_clip = {(r["flight"], r["clip_id"]): r["score"] for r in records
                    if r["condition"] == "identity" and r["testable"]}
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for cond in NEGATIVES:
        pairs = [(real_by_clip[(r["flight"], r["clip_id"])], r["score"]) for r in records
                 if r["condition"] == cond and r["testable"]
                 and (r["flight"], r["clip_id"]) in real_by_clip]
        if len(pairs) < 4:
            continue
        pos = np.array([p[0] for p in pairs])
        neg = np.array([p[1] for p in pairs])
        thr = np.unique(np.concatenate([pos, neg]))
        tpr = [(neg > t).mean() for t in thr]
        fpr = [(pos > t).mean() for t in thr]
        ax.plot(fpr, tpr, label=f"{cond} (AUROC {auroc_detect(pos, neg):.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate (real clips rejected)")
    ax.set_ylabel("true positive rate (negatives rejected)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ks, rej = [], []
    for cond, m in ev["mdv"].items():
        ks.append(m["violation_median_deg"])
        rej.append(m["rejection_rate"])
    if ks:
        order = np.argsort(ks)
        ax.plot(np.array(ks)[order], np.array(rej)[order], "o-")
    ax.axhline(0.8, color="k", ls="--", lw=1, label="80% rejection")
    ax.set_xlabel("injected tilt change (deg)")
    ax.set_ylabel("rejection rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mdv_curve.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p3a_v1")
    ap.add_argument("--specs", default="data/processed/p3a")
    ap.add_argument("--geocalib", default="results/phase3a_verdict/geocalib_p3a_v1")
    ap.add_argument("--real-tag", default="identity")
    ap.add_argument("--conditions", default=",".join(NEGATIVES))
    ap.add_argument("--out", default="results/phase3a_verdict")
    ap.add_argument("--ablate", default="both", choices=["both", "none", "no_uptilt", "no_lag"])
    args = ap.parse_args()

    info = verify_prereg()
    prereg = info["doc"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    suffix = "" if args.ablate == "both" else f"_{args.ablate}"
    out = ROOT / args.out / f"{run_id}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    spec_dir = ROOT / args.specs / args.tag
    clips = json.loads((spec_dir / "clips.json").read_text())["clips"]
    geo = ROOT / args.geocalib
    spec_paths = {p.name.replace(".npz", ""): p for p in sorted(spec_dir.glob("flight-*.npz"))}

    conds = [args.real_tag] + [c for c in args.conditions.split(",") if c]
    records: list[dict] = []
    for flight, path in spec_paths.items():
        rows_by_cond = {}
        for c in conds:
            f = geo / f"{flight}__{c}.jsonl"
            if f.exists():
                rows_by_cond[c] = load_jsonl(f)
        if not rows_by_cond:
            print(f"{flight}: no GeoCalib rows, skipped")
            continue
        recs = score_flight(path, rows_by_cond, clips, prereg, ablate=args.ablate)
        records.extend(recs)
        for c in rows_by_cond:
            sub = [r for r in recs if r["condition"] == c]
            if sub:
                print(f"{flight} [{c}]: {sum(r['testable'] for r in sub)}/{len(sub)} testable, "
                      f"score median {np.median([r['score'] for r in sub if r['testable']] or [np.nan]):.2f}",
                      flush=True)

    ev = evaluation(records, prereg)
    ev["threshold_policy"] = prereg["thresholds"].get("threshold_fit_flights", "all_flights")
    for alt in ("all_flights", "all_but_13a"):
        e_alt = evaluation(records, prereg, policy=alt)
        ev[f"alt_threshold_policy_{alt}"] = {
            k: e_alt[k] for k in ("heldout_acceptance", "heldout_uncertified_alt",
                                  "pooled_acceptance", "family_auroc_pooled", "families",
                                  "mdv", "mdv_smallest_rejected80_deg") if k in e_alt}
    ident_rows = {f: load_jsonl(geo / f"{f}__identity.jsonl")
                  for f in spec_paths if (geo / f"{f}__identity.jsonl").exists()}
    if ident_rows:
        ev["delay_negatives"] = quick_delays(spec_paths, ident_rows, clips, prereg)
        ev["retime_negatives"] = quick_retime(spec_paths, ident_rows, clips, prereg)
    ev["diagnostics"] = diagnostics(records)
    ev["identity_vs_phase1"] = identity_vs_phase1(ident_rows, ROOT / "data/raw/geocalib/div10")
    ev["artifact_floor"] = artifact_floor(
        spec_paths, ident_rows,
        {f: (load_jsonl(geo / f"{f}__yaw_control.jsonl")
             if (geo / f"{f}__yaw_control.jsonl").exists() else None) for f in spec_paths})
    ident = [r for r in records if r["condition"] == "identity"]
    ev["abstention_rate"] = {
        "overall": float(np.mean([not r["testable"] for r in ident])) if ident else None,
        "trackratm": float(np.mean([not r["testable"] for r in ident if is_trackratm(r)]))
        if any(is_trackratm(r) for r in ident) else None,
        "autonomous": float(np.mean([not r["testable"] for r in ident if r["control"] == "autonomous"]))
        if any(r["control"] == "autonomous" for r in ident) else None,
        "piloted": float(np.mean([not r["testable"] for r in ident if r["control"] == "piloted"]))
        if any(r["control"] == "piloted" for r in ident) else None,
        "per_flight": {f: float(np.mean([not r["testable"] for r in ident if r["flight"] == f]))
                       for f in sorted({r["flight"] for r in ident})},
    }
    (out / "metrics.json").write_text(json.dumps(
        {"run_id": out.name, "git": git_hash(), "prereg": {k: info[k] for k in
                                                           ("path", "sha256", "git_hash")},
         "config": vars(args), "n_records": len(records),
         "evaluation": ev, "records": records}, indent=1, default=float))
    (out / "config.yaml").write_text(json.dumps(vars(args), indent=1))
    plots(out, records, ev)
    print(json.dumps({k: ev[k] for k in ("heldout_acceptance", "pooled_acceptance",
                                        "family_auroc_pooled", "abstention_rate", "gate3")},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
