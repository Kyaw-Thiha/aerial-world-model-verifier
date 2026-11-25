"""X01: tilt fidelity of each generator on the common path set.

Descriptive Stage-1 statistic, computed from stored artefacts only.

For every produced clip:
  * required side: the flatness lean of the commanded path at the clip's frame times, computed
    by ``experiments/x01_common_paths.py`` with the frozen 0.37 s window and k_d = 0.295;
  * depicted side: the frozen reader (GeoCalib 1dedb29, 224 centre crop, f = 112 px), rows in
    ``results/x01/depicted/<model>_<tag>.jsonl``, converted to a lean with
    ``src.verdict.depicted_lean_from_gravity``;
  * tilt-fidelity score: the median over eligible frames of |lean_dep| / |lean_req|, eligible =
    |lean_req| >= 3 deg on an in-envelope path;
  * null: the same statistic with path A's depicted lean paired with path B's required lean
    (fixed cyclic shift), which must be near the "no relationship" value;
  * fraction of in-envelope paths whose median ratio lies in [0.5, 1.5].

Cluster bootstrap over paths (5,000 replicates, ``default_rng(1)``); fewer than 10 clusters is
reported as descriptive only.

    pixi run python experiments/x01_score.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.perception.frames import camera_to_body_pitched  # noqa: E402
from src.physics.flatness import G0  # noqa: E402
from src.verdict import (Clip, FitOptions, assess, depicted_lean_from_gravity,  # noqa: E402
                         fit_gain)
from src.verdict.lean import world_to_yaw_frame  # noqa: E402

MIN_REQ_DEG = 3.0
ABSTENTION = yaml.safe_load((ROOT / "configs/gain_frozen.yaml").read_text())["abstention"]
RATIO_BAND = (0.5, 1.5)
N_BOOT = 5000


def clip_id_to_path(sid: str) -> str | None:
    """Map a generated clip directory name to the path key of ``paths_summary.json``.

    Camera models write ``<path>__S1`` and ``<path>__S2``; ANWM writes ``x01_<path>_<scene>``
    (its ids carry no double underscore, so the suffix alone disambiguates).
    """
    if sid.startswith("x01_"):
        body = sid[4:]
        if body.endswith("_S1"):
            return body[:-3]
        if body.endswith("_S2"):
            return body[:-3] + "__S2"
        return None
    if sid.endswith("__S1"):
        return sid[:-4]
    if sid.endswith("__S2"):
        return sid
    return None


def load_rows(depicted: Path) -> dict[tuple[str, str], dict[str, dict[int, dict]]]:
    """(model, tag) -> clip -> frame index -> reader row, merged over every shard file."""
    out: dict[tuple[str, str], dict[str, dict[int, dict]]] = defaultdict(
        lambda: defaultdict(dict))
    for p in sorted(depicted.rglob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r.get("model", "?"), r.get("tag", p.parent.name))][r["clip_id"]][
                int(Path(r["frame"]).stem)] = r
    return out


def clip_stats(lean_dep: np.ndarray, lean_req: np.ndarray, tilt_req: np.ndarray,
               eligible_min: float = MIN_REQ_DEG) -> dict:
    """Through-origin gain of depicted on required lean, plus the magnitude ratio.

    ``gain`` = sum(dep . req) / sum(req . req) over eligible frames, both components pooled.
    A generator that depicts the commanded lean gives 1, one that ignores it gives 0, one that
    mirrors it gives a negative value.  ``ratio`` = median |dep| / |req| is reported beside it
    because H04 used that form; it is not the primary statistic (a constant depicted lean of
    realistic size gives a ratio near 1 whatever the command).
    """
    norm = np.linalg.norm(lean_req, axis=1)
    ok = np.isfinite(norm) & np.isfinite(lean_dep).all(axis=1) & (norm >= eligible_min / 57.3)
    if ok.sum() < 3:
        return {"n_frames": int(ok.sum()), "gain": float("nan"), "ratio": float("nan"),
                "r2": float("nan")}
    x = lean_req[ok].ravel()
    y = lean_dep[ok].ravel()
    xx = float(x @ x)
    gain = float((x @ y) / xx) if xx > 0 else float("nan")
    resid = y - gain * x
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid @ resid) / ss) if ss > 0 else float("nan")
    dep = np.linalg.norm(lean_dep[ok], axis=1)
    req = norm[ok]
    return {"n_frames": int(ok.sum()), "gain": gain, "r2": r2,
            "ratio": float(np.median(dep / req))}


def fitted_gain(t, pos, yaw, dep_lean, dep_conf, lean_req, vel_req, tilt_req,
                min_frames: int) -> dict:
    """X01 tilt gain: per-axis OLS slope of depicted on required lean, with an intercept.

    The project's frozen gain fitter (``configs/gain_frozen.yaml``) adds a drag regressor and an
    intercept.  On this path set the commanded speed is constant (the design has no forward-speed
    modulation outside the accel/brake family), so the drag regressor is collinear with the
    intercept and the frozen fitter's design is rank-deficient: condition numbers 1.7e5-3.1e5
    against its own 1e3 ceiling, i.e. every clip abstains.  The X01 statistic is therefore the
    two-parameter fit per axis (slope + intercept), which is the same "depicted lean divided by
    required lean" in slope form without the collinear channel.  Recorded in
    ``configs/x01_common_paths_amend1.yaml``.  Eligibility uses the frozen excitation floor
    (2 deg of required-lean spread) and a frame minimum.
    """
    n = min(len(dep_lean), len(lean_req))
    x2 = np.asarray(lean_req, float)[:n]
    y2 = np.asarray(dep_lean, float)[:n]
    ok = np.isfinite(x2).all(axis=1) & np.isfinite(y2).all(axis=1)
    x2, y2 = x2[ok], y2[ok]
    spread = float(np.sqrt(np.linalg.eigvalsh(np.cov(x2.T))[-1])) if len(x2) > 2 else 0.0
    out = {"gain": float("nan"), "gx": float("nan"), "gy": float("nan"), "r2": float("nan"),
           "n_frames": int(len(x2)), "excitation_deg": float(np.degrees(spread)),
           "testable": False, "abstain": []}
    if len(x2) < int(min_frames):
        out["abstain"] = [f"min_frames {len(x2)} < {min_frames}"]
        return out
    if out["excitation_deg"] < 2.0:
        out["abstain"] = [f"excitation {out['excitation_deg']:.2f} < 2.0"]
        return out
    slopes, preds = [], []
    for k in range(2):
        x, y = x2[:, k], y2[:, k]
        A = np.column_stack([x, np.ones_like(x)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        slopes.append(float(beta[0]))
        preds.append(A @ beta)
    resid = np.concatenate([y2[:, k] - preds[k] for k in range(2)])
    total = np.concatenate([y2[:, k] - y2[:, k].mean() for k in range(2)])
    ss_t = float(total @ total)
    out.update({"gain": 0.5 * (slopes[0] + slopes[1]), "gx": slopes[0], "gy": slopes[1],
                "r2": float(1.0 - (resid @ resid) / ss_t) if ss_t > 0 else float("nan"),
                "testable": True})
    return out


def boot_median(values: np.ndarray, clusters: np.ndarray, rng) -> dict:
    """Cluster bootstrap of the median; clusters are paths."""
    uniq = np.unique(clusters)
    if len(uniq) < 3 or not np.isfinite(values).any():
        return {"median": float(np.nanmedian(values)) if len(values) else float("nan"),
                "lo": None, "hi": None, "n_clusters": int(len(uniq)), "descriptive": True}
    idx = {c: np.where(clusters == c)[0] for c in uniq}
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([values[idx[c]] for c in pick])
        draws[b] = np.nanmedian(vals) if len(vals) else np.nan
    return {"median": float(np.nanmedian(values)),
            "lo": float(np.nanpercentile(draws, 2.5)),
            "hi": float(np.nanpercentile(draws, 97.5)),
            "n_clusters": int(len(uniq)),
            "descriptive": bool(len(uniq) < 10)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default=str(ROOT / "results/x01/paths.npz"))
    ap.add_argument("--summary", default=str(ROOT / "results/x01/paths_summary.json"))
    ap.add_argument("--depicted", default=str(ROOT / "results/x01/depicted"))
    ap.add_argument("--out", default=str(ROOT / "results/x01"))
    args = ap.parse_args()
    d = np.load(args.paths)
    summary = json.loads(Path(args.summary).read_text())
    R_cb = camera_to_body_pitched(0.0)
    rng = np.random.default_rng(1)

    by_model_path = {(s["model"], s["path"]): s for s in summary}
    result: dict = {"min_req_deg": MIN_REQ_DEG, "ratio_band": list(RATIO_BAND),
                    "models": {}}
    ratios_by_model: dict[str, np.ndarray] = {}
    clusters_by_model: dict[str, np.ndarray] = {}

    groups = load_rows(Path(args.depicted))
    for (model_raw, tag), rows in sorted(groups.items()):
        if not rows:
            continue
        model = model_raw if tag in ("d42", "full", "x01", "C") else f"{model_raw}_{tag}"
        clips = []
        for sid, frames in sorted(rows.items()):
            path = clip_id_to_path(sid)
            if path is None or (model, path) not in by_model_path:
                continue
            key = f"{model}__{path}"
            if f"{key}__lean_req" not in d:
                continue
            idx = sorted(frames)
            up = np.array([frames[i]["gravity_up_cam"] for i in idx], dtype=float)
            lean_dep = depicted_lean_from_gravity(up, R_cb)
            lean_req = d[f"{key}__lean_req"][:len(idx)]
            tilt_req = d[f"{key}__tilt_req"][:len(idx)]
            if model == "anwm":
                # ANWM's 16 generated frames follow context frame 0 of its 17-step path
                lean_req = d[f"{key}__lean_req"][1:1 + len(idx)]
                tilt_req = d[f"{key}__tilt_req"][1:1 + len(idx)]
            if len(lean_req) != len(lean_dep):
                n = min(len(lean_req), len(lean_dep))
                lean_req, tilt_req, lean_dep = lean_req[:n], tilt_req[:n], lean_dep[:n]
            # the frozen gain statistic on the same frames
            pos = d[f"{key}__pos"][:len(idx)]
            yaw = d[f"{key}__yaw"][:len(idx)]
            vel = d[f"{key}__vel"][:len(idx)]
            if model == "anwm":
                pos, yaw, vel = (d[f"{key}__pos"][1:1 + len(idx)],
                                 d[f"{key}__yaw"][1:1 + len(idx)],
                                 d[f"{key}__vel"][1:1 + len(idx)])
            n = min(len(pos), len(lean_dep), len(lean_req))
            pos, yaw, vel = pos[:n], yaw[:n], vel[:n]
            lean_dep, lean_req, tilt_req = lean_dep[:n], lean_req[:n], tilt_req[:n]
            vel_req = world_to_yaw_frame(vel[:, :2] / G0, yaw)
            conf = np.array([frames[i].get("latitude_confidence_mean", np.nan) for i in idx])
            conf = np.where(np.isfinite(conf), conf, 1.0)[:n]
            try:
                fit = fitted_gain(np.arange(n) / 16.0, pos, yaw, lean_dep, conf, lean_req,
                                  vel_req, tilt_req,
                                  min_frames=10 if model == "anwm" else 24)
            except Exception as exc:                      # noqa: BLE001
                fit = {"gain": float("nan"), "gx": float("nan"), "gy": float("nan"),
                       "lag_ms": float("nan"), "n_frames": int(n), "error": str(exc)}
            st = clip_stats(lean_dep, lean_req, tilt_req)
            st.update({k: fit[k] for k in ("gain", "gx", "gy", "r2", "excitation_deg", "testable", "abstain")})
            st["stat"] = "x01_slope_gain"
            meta = by_model_path[(model, path)]
            st.update({"clip_id": sid, "path": path, "family": meta["family"],
                       "in_envelope": meta["in_envelope"],
                       "requested_peak_deg": meta["requested_peak_deg"],
                       "peak_tilt_req_deg": meta["peak_tilt_deg"],
                       "n_frames_clip": len(idx)})
            clips.append(st)
        if not clips:
            continue
        testable = [c for c in clips if c.get("testable") and np.isfinite(c["gain"])]
        in_env = [c for c in testable if c["in_envelope"]]
        vals = np.array([c["gain"] for c in in_env])
        clus = np.array([c["path"] for c in in_env])
        ratios_by_model[model] = vals
        clusters_by_model[model] = clus
        band_ok = [c for c in in_env if RATIO_BAND[0] <= c["gain"] <= RATIO_BAND[1]]
        nulls = []
        for i, c in enumerate(in_env):
            other = in_env[(i + 7) % len(in_env)]
            key_o = f"{model}__{other['path']}"
            if f"{key_o}__lean_req" not in d:
                continue
            rows_o = rows.get(other["clip_id"], {})
            idx = sorted(rows[c["clip_id"]])
            up = np.array([rows[c["clip_id"]][i2]["gravity_up_cam"] for i2 in idx], float)
            lean_dep = depicted_lean_from_gravity(up, R_cb)
            lean_req = d[f"{key_o}__lean_req"][:len(lean_dep)]
            tilt_req = d[f"{key_o}__tilt_req"][:len(lean_dep)]
            nulls.append(clip_stats(lean_dep, lean_req, tilt_req)["gain"])
        result["models"][model] = {
            "n_clips_produced": len(clips),
            "n_clips_testable": len(testable),
            "n_clips_in_envelope": len(in_env),
            "tilt_fidelity": {**boot_median(vals, clus, rng),
                              "median_ratio": float(np.nanmedian(
                                  [c["ratio"] for c in in_env])),
                              "median_r2": float(np.nanmedian(
                                  [c["r2"] for c in in_env]))},
            "frac_in_band": {"k": len(band_ok), "n": len(in_env),
                             "value": (len(band_ok) / len(in_env)) if in_env else None},
            "null_gain": {"median": float(np.nanmedian(nulls)) if nulls else None,
                           "n": len(nulls)},
            "by_path_class": {},
            "clips": clips,
        }
        for fam in sorted({c["family"] for c in in_env}):
            sub = [c for c in in_env if c["family"] == fam]
            v = np.array([c["gain"] for c in sub])
            cl = np.array([c["path"] for c in sub])
            result["models"][model]["by_path_class"][fam] = {
                "k": int(len(sub)), "n_paths": int(len(set(cl))),
                **boot_median(v, cl, rng)}

    # pairwise model differences on the shared in-envelope paths, path-clustered
    models = sorted(ratios_by_model)
    pairs = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            pa = {c["path"]: c["gain"] for c in result["models"][a]["clips"]
                  if c["in_envelope"] and np.isfinite(c["gain"])}
            pb = {c["path"]: c["gain"] for c in result["models"][b]["clips"]
                  if c["in_envelope"] and np.isfinite(c["gain"])}
            shared = sorted(set(pa) & set(pb))
            if len(shared) < 3:
                continue
            diff = np.array([pa[p] - pb[p] for p in shared])
            uniq = np.array(shared)
            draws = np.empty(N_BOOT)
            for k in range(N_BOOT):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                lut = dict(zip(uniq, diff))
                draws[k] = float(np.median([lut[p] for p in pick]))
            pairs[f"{a}-{b}"] = {"n_paths": len(shared), "median_diff": float(np.median(diff)),
                                 "lo": float(np.percentile(draws, 2.5)),
                                 "hi": float(np.percentile(draws, 97.5))}
    result["pairwise"] = pairs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(result, indent=1))

    lines = ["# X01 tilt fidelity (descriptive, Stage 1)", "",
             f"eligible frames: |lean_req| >= {MIN_REQ_DEG} deg, in-envelope paths only", "",
             "| model | clips | testable | in-env | median gain | 95% CI | paths | gain in [0.5,1.5] | null | median ratio | R2 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in models:
        r = result["models"][m]
        tf = r["tilt_fidelity"]
        ci = ("descriptive" if tf.get("descriptive") or tf["lo"] is None
              else f"{tf['lo']:.2f}-{tf['hi']:.2f}")
        lb = r["frac_in_band"]
        null = r["null_gain"]["median"] or float("nan")
        lines.append(f"| {m} | {r['n_clips_produced']} | {r['n_clips_testable']} | "
                     f"{r['n_clips_in_envelope']} | "
                     f"{tf['median']:.2f} | {ci} | {tf['n_clusters']} | "
                     f"{lb['k']}/{lb['n']} | {null:.2f} | "
                     f"{tf['median_ratio']:.2f} | {tf['median_r2']:.2f} |")
    (out / "score_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for k, v in pairs.items():
        print(f"{k}: diff {v['median_diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] "
              f"over {v['n_paths']} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
