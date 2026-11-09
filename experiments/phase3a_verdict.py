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
from src.score.prereg import verify_prereg
from src.score.verdict import clip_index, speed_bin

ROOT = Path(__file__).resolve().parents[1]
NEGATIVES = ["tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll", "yaw_control"]
FAMILY = {"tilt_k0": "tilt_removal", "tilt_k05": "tilt_scaling",
          "tilt_k15": "tilt_scaling", "mirror_roll": "mirrored_roll",
          "yaw_control": "control"}
TRACK_A_FLIGHTS = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}


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
                      "edit_angle_deg": float(np.median(spec[f"angle_{cond}"][idx]))
                      if f"angle_{cond}" in spec.files else 0.0})
            out.append(r)
    return out


def _bin_groups(fit: list[dict], bins: list[list[float]], min_n: int) -> list[dict]:
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
        g["threshold"] = conformal_quantile(vals, 0.05) if vals else float("nan")
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
           and (policy == "all_flights" or r["flight"] not in TRACK_A_FLIGHTS)]
    groups = _bin_groups(fit, bins, min_n) if fit else []
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
            return info["threshold"]
    return th["pooled"]["threshold"]


def evaluation(records: list[dict], prereg: dict,
               policy: str | None = None) -> dict:
    """Held-out acceptance, rejection, AUROC, MDV, Spearman, controls."""
    policy = policy or prereg["thresholds"].get("threshold_fit_flights", "all_flights")
    flights = sorted({r["flight"] for r in records})
    real = {(r["flight"], r["condition"]): r for r in records if r["condition"] == "identity"}
    per_clip_threshold = {}
    heldout_accept = []
    folded = []
    for f in flights:
        th = thresholds_for(records, prereg, holdout=f, policy=policy)
        for r in records:
            if r["flight"] == f and r["condition"] == "identity" and r["testable"]:
                t = threshold_lookup(th, r["bin"])
                heldout_accept.append(r["score"] <= t)
                per_clip_threshold[(f, r["clip_id"])] = t
        folded.append({"flight": f, "thresholds": th})
    pooled_th = thresholds_for(records, prereg, holdout=None, policy=policy)
    pooled_accept = [r["score"] <= threshold_lookup(pooled_th, r["bin"])
                     for r in records if r["condition"] == "identity" and r["testable"]]

    res = {"heldout_acceptance": float(np.mean(heldout_accept)) if heldout_accept else None,
           "heldout_n": len(heldout_accept),
           "pooled_acceptance": float(np.mean(pooled_accept)) if pooled_accept else None,
           "pooled_thresholds": pooled_th, "fold_thresholds": folded,
           "families": {}, "mdv": {}, "ablations": {}}

    # per-family rejection and AUROC against the paired real clip
    for cond in NEGATIVES:
        neg = [r for r in records if r["condition"] == cond and r["testable"]]
        pairs = [(real[(r["flight"], "identity")], r) for r in neg
                 if (r["flight"], "identity") in real
                 and real[(r["flight"], "identity")]["testable"]]
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
            "spearman_vs_magnitude": spearman(sco[ok], [p[1]["tilt_implied_median"]
                                                        for p, k in zip(pairs, ok) if k]),
        }
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
                p = real.get((r["flight"], "identity"))
                if p is not None and p["testable"]:
                    pos.append(p["score"])
                    neg_scores.append(r["score"])
                    names.append(cond)
    res["family_auroc_pooled"] = auroc_detect(np.array(pos), np.array(neg_scores)) if pos else None

    # separate report for trackRATM flights
    for label, sel in (("trackratm", lambda f: f in TRACK_A_FLIGHTS),
                       ("other", lambda f: f not in TRACK_A_FLIGHTS)):
        acc = [r["score"] <= per_clip_threshold.get((r["flight"], r["clip_id"]), np.nan)
               for r in records if r["condition"] == "identity" and r["testable"]
               and sel(r["flight"]) and np.isfinite(
                   per_clip_threshold.get((r["flight"], r["clip_id"]), np.nan))]
        res[f"heldout_acceptance_{label}"] = float(np.mean(acc)) if acc else None
        res[f"n_{label}"] = len(acc)
    mdv = res["mdv"]
    if mdv:
        pts = sorted((v["violation_median_deg"], v["rejection_rate"]) for v in mdv.values())
        hit = [x for x, y in pts if y >= 0.8]
        res["mdv_smallest_rejected80_deg"] = float(min(hit)) if hit else None
    return res


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
            f = geo / (f"{flight}.jsonl" if c == "identity_real" else f"{flight}__{c}.jsonl")
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
    ev_alt = evaluation(records, prereg, policy="all_flights")
    ev["alt_threshold_policy_all_flights"] = {
        k: ev_alt[k] for k in ("heldout_acceptance", "pooled_acceptance", "family_auroc_pooled",
                               "families") if k in ev_alt}
    ident_rows = {f: load_jsonl(geo / f"{f}__identity.jsonl")
                  for f in spec_paths if (geo / f"{f}__identity.jsonl").exists()}
    if ident_rows:
        ev["delay_negatives"] = quick_delays(spec_paths, ident_rows, clips, prereg)
        ev["retime_negatives"] = quick_retime(spec_paths, ident_rows, clips, prereg)
    ev["abstention_rate"] = {
        "overall": float(np.mean([not r["testable"] for r in records if r["condition"] == "identity"]))
        if records else None,
        "trackratm": float(np.mean([not r["testable"] for r in records
                                    if r["condition"] == "identity" and r["flight"] in TRACK_A_FLIGHTS]))
        if records else None,
    }
    ev["gate3"] = {
        "real_acceptance_at_least_0.95": ev.get("heldout_acceptance") is not None
        and ev["heldout_acceptance"] >= 0.95,
        "auroc_above_0.9_tilt_families": all(
            (ev["families"].get(c, {}).get("auroc") or 0.0) > 0.9
            for c in ("tilt_k0", "tilt_k05", "tilt_k15")),
        "null_edit_accepted": (ev["families"].get("identity", {}).get("rejection_rate") is None),
        "control_accepted": (ev["families"].get("yaw_control", {}).get("rejection_rate") or 0.0) <= 0.10,
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
