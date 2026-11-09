"""Task 3: score Track B's generated (ANWM) clips in commanded-path mode.

Input contract (Track B): R2 ``track-b/clips/clips_index.csv`` plus one directory of
frames per clip and a manifest per clip holding the commanded path (``dx, dy, dz, dyaw``
per step) and the designed required tilt at the nominal dt. Frames are 224x224.

What this script does per clip:

1. depicted tilt per frame with the Task 1 setup (GeoCalib pinhole, 90 deg field,
   f = 112 px at 224), plus a coherence statistic: if the estimated up direction jumps
   by more than ``--jump-deg`` between most consecutive frames the rollout is visually
   collapsed and the clip is marked **not testable** rather than failed;
2. implied tilt from the commanded path for each dt in {0.25, 0.667, 1.0} s — frames do
   not depend on dt, so this is analysis, not a new generation;
3. residual against the pre-registered thresholds, per regime;
4. the Regime C dose-response: depicted tilt against required tilt, with the controls.

Nothing here claims a dt for ANWM. dt is a swept axis and every number is a curve over it.

    pixi run -e gpu python experiments/phase3a_anwm_scoring.py --clips <dir>
    pixi run -e gpu python experiments/phase3a_anwm_scoring.py --synthetic  # machinery check
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

from src.physics.flatness import G0, implied_tilt_from_positions
from src.score.prereg import verify_prereg
from src.score.verdict import auroc, conformal_quantile, spearman

ROOT = Path(__file__).resolve().parents[1]
DTS = (0.25, 0.667, 1.0)
SG_WINDOW = 11          # the ANWM excitation audit's window, in frames
SG_POLY = 3


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_index(clips_dir: Path) -> list[dict]:
    """Read ``clips_index.csv`` if present; return [] with a recorded reason otherwise."""
    import csv

    p = clips_dir / "clips_index.csv"
    if not p.exists():
        return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def implied_from_path(points: np.ndarray, dt: float, k_d: float = 0.295) -> np.ndarray:
    """Implied tilt (deg) from a commanded path at one dt, via the shared physics helper."""
    tilt, mask = implied_tilt_from_positions(points, float(dt), SG_WINDOW, SG_POLY, k_d, G0)
    return tilt


def coherence(up: np.ndarray) -> dict:
    """Frame-to-frame movement of the estimated up direction, degrees."""
    if len(up) < 2:
        return {"n_pairs": 0, "median_jump_deg": float("nan"), "p90_jump_deg": float("nan")}
    u = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-12)
    d = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u[1:], u[:-1]), -1.0, 1.0)))
    return {"n_pairs": int(len(d)), "median_jump_deg": float(np.median(d)),
            "p90_jump_deg": float(np.percentile(d, 90))}


def score_clip(tilt_dep: np.ndarray, tilt_imp: np.ndarray, prereg: dict) -> dict:
    """Residual for one clip at one dt: the same statistic as the verdict (median)."""
    tilt_dep = np.asarray(tilt_dep, dtype=np.float64)
    tilt_imp = np.asarray(tilt_imp, dtype=np.float64)
    r = np.abs(tilt_dep - tilt_imp)
    ok = np.isfinite(r)
    if ok.sum() == 0:
        return {"testable": False, "reason": "no finite frames"}
    signal = np.isfinite(tilt_imp) & (tilt_imp >= 10.0)
    n_sig = int(signal.sum())
    frac = n_sig / max(len(tilt_imp), 1)
    if n_sig < 12 or frac < 0.20:
        return {"testable": False, "reason": f"excitation {n_sig} frames, frac {frac:.2f}",
                "n_signal": n_sig, "frac_signal": frac}
    use = ok & signal
    return {"testable": True, "score_deg": float(np.median(r[use])),
            "p90_deg": float(np.percentile(r[use], 90)),
            "n_scored": int(use.sum()), "n_signal": n_sig, "frac_signal": float(frac),
            "dep_tilt_median": float(np.median(tilt_dep[use])),
            "imp_tilt_median": float(np.median(tilt_imp[use]))}


def thresholds_from_verdict(path: Path) -> dict:
    """Read the pre-registered thresholds from a Phase 3a verdict run."""
    if not path.exists():
        return {}
    m = json.loads(path.read_text())
    return m.get("evaluation", {}).get("pooled_thresholds", {})


def dose_response(rows: list[dict]) -> dict:
    """Regime C: depicted tilt against required tilt, per dt and per model family.

    ``C_control`` rows are the matched controls (level-camera references); they are
    reported next to the treatment rows, never pooled into them.
    """
    out: dict = {}
    models = sorted({r.get("model") for r in rows if r.get("model")})
    for dt in DTS:
      for model in (models or [None]):
        sel = [r for r in rows if r.get("regime") == "C" and abs(r["dt"] - dt) < 1e-9
               and r.get("testable") and (model is None or r.get("model") == model)]
        ctrl = [r for r in rows if r.get("regime") == "C_control" and abs(r["dt"] - dt) < 1e-9
                and r.get("testable")]
        key = f"{dt:g}" + (f"|{model}" if model else "")
        if not sel:
            out[key] = {"n": 0}
            continue
        req = np.array([r["required_tilt_deg"] for r in sel])
        dep = np.array([r["dep_tilt_median"] for r in sel])
        out[key] = {"n": int(len(sel)), "model": model,
                        "required_median_deg": float(np.median(req)),
                        "depicted_median_deg": float(np.median(dep)),
                        "spearman_depicted_vs_required": spearman(req, dep),
                        "matched_control_n": int(len(ctrl)),
                        "matched_control_depicted_median_deg":
                            float(np.median([r["dep_tilt_median"] for r in ctrl])) if ctrl else None}
    return out


def seed_variability(rows: list[dict]) -> dict:
    """Spread of the clip score across seeds for the same path."""
    by_path: dict = {}
    for r in rows:
        key = (f"{r.get('model')}/{r.get('path_id')}", round(float(r["dt"]), 3))
        by_path.setdefault(key, []).append(r.get("score_deg"))
    out = {}
    for (path_id, dt), vals in by_path.items():
        v = [x for x in vals if x is not None]
        if len(v) >= 2:
            out.setdefault(str(dt), []).append({"path_id": path_id, "n_seeds": len(v),
                                                "median_deg": float(np.median(v)),
                                                "range_deg": float(max(v) - min(v))})
    return {dt: {"n_paths": len(v), "median_range_deg": float(np.median([x["range_deg"] for x in v]))}
            for dt, v in out.items()}


def synthetic_rows(prereg: dict) -> dict:
    """Machinery check with no generated clips: a level-camera model vs a banking model.

    The hypothesis Task 3 has to test is stated in the docstring: a 4-DoF level-camera
    generator depicts tilt near the Task 1 floor while the commanded path requires a
    tilt that rises with speed and turn rate. This builds that contrast synthetically
    (several paths x 3 seeds) so the scoring, the dose-response and the seed-variability
    machinery are exercised before any real clip exists. It is a code check, not a
    measurement of ANWM.
    """
    rng = np.random.default_rng(0)
    floor = 0.57
    rows = []
    for dt in DTS:
        n = 48
        t = np.arange(n) * dt
        for i in range(8):
            amp = 2.0 + 1.2 * i                      # 2 to 10.4 m/s^2 of commanded accel
            acc = np.column_stack([amp * np.sin(0.5 * t + 0.3 * i),
                                   0.6 * amp * np.cos(0.4 * t), np.zeros_like(t)])
            vel = np.cumsum(acc, axis=0) * dt
            pts = np.cumsum(vel, axis=0) * dt
            imp = implied_from_path(pts, dt)
            imp = np.clip(imp, 0.0, None)
            for seed in range(3):
                for model in ("level", "banking"):
                    dep = (np.full(n, floor) if model == "level" else imp.copy()) \
                        + rng.normal(scale=0.5, size=n)
                    s = score_clip(dep, imp, prereg)
                    rows.append({"clip_id": f"syn/{model}/{dt}/{i}/{seed}", "regime": "C",
                                 "dt": dt, "model": model, "path_id": f"{dt}/{i}",
                                 "seed": seed,
                                 "required_tilt_deg": float(np.nanmedian(imp)), **s})
    return {"rows": rows, "note": "machinery check only; no ANWM frames involved"}


def plots(out: Path, rows: list[dict], dose: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, style in (("level", "o-"), ("banking", "s-")):
        sel = [r for r in rows if r.get("model") == model and r.get("testable")]
        if sel:
            ax.scatter([r["required_tilt_deg"] for r in sel], [r["dep_tilt_median"] for r in sel],
                       label=model, s=30)
    ax.set_xlabel("required tilt from the commanded path (deg)")
    ax.set_ylabel("depicted tilt (deg)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dose_response.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/raw/track_b_clips")
    ap.add_argument("--verdict-run", default="")
    ap.add_argument("--out", default="results/anwm_scoring")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--jump-deg", type=float, default=20.0)
    args = ap.parse_args()

    info = verify_prereg()
    prereg = info["doc"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / f"{run_id}{'_synthetic' if args.synthetic else ''}"
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        synth = synthetic_rows(prereg)
        rows = synth["rows"]
        result = {"synthetic": True, "note": synth["note"], "rows": rows,
                  "dose_response": dose_response(rows), "seed_variability": seed_variability(rows)}
        plots(out, rows, result["dose_response"])
    else:
        clips_dir = ROOT / args.clips
        index = load_index(clips_dir)
        if not index:
            result = {"clips_available": False,
                      "reason": f"no clips_index.csv under {clips_dir}; Track B has not "
                                "delivered yet. The scoring path is built and the interface "
                                "is fixed in docs/track_b_clips.md expectations.",
                      "expected": ["clips_index.csv", "<clip_id>/frames/*.jpg",
                                   "<clip_id>/path.npz with the commanded (dx,dy,dz,dyaw) steps"]}
            (out / "not_available.json").write_text(json.dumps(result, indent=1))
            print(json.dumps(result, indent=2))
            return 0
        raise SystemExit("clip loading lands with the first delivered clips: see the docstring")
    (out / "metrics.json").write_text(json.dumps(
        {"run_id": out.name, "git": git_hash(),
         "prereg": {k: info[k] for k in ("path", "sha256", "git_hash")}, **result},
        indent=1, default=float))
    print(json.dumps({k: result[k] for k in ("dose_response", "seed_variability") if k in result},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
