"""Gate 2 priors study on stored extractor poses (no GPU, no pixels).

Inputs: the npz files written by ``remote/c_extractors.py`` (``c2w`` per frame) plus the
prepared truth (``truth.npz``). Both are small; fetch them with
``modal volume get awmv-c-data poses ...``.

Three things the verifier needs:

1. per-clip **scale ratio** (Sim(3) scale against truth), an interval fitted on the fit
   split and coverage measured on the held-out split, split by sequence;
2. the **fps sensitivity**: every result repeated with the time base wrong by 2x;
3. the **apparent gravity** of the recovered path against the *oracle* lean (truth attitude
   supplies the lean). This is the oracle-lean version of the handoff's statistic: it
   isolates extractor and time-base error from depicted-tilt error, which needs GeoCalib on
   this data and is not run here.

    python experiments/pathrec_priors_study.py --poses /tmp/poses --truth /tmp/truth \
        --out results/pathrec_priors/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.pathrec.align import umeyama
from src.pathrec.evaluate import up_axis
from src.pathrec.lean import differentiate, lean_xy
from src.pathrec.priors import interval, lean_regression, scale_ratio, time_rescale


def analyse_clip(est_c2w: np.ndarray, true_pos: np.ndarray, true_quat: np.ndarray,
                 fps: float, window_s: float = 0.5) -> dict:
    """Scale ratio, fps sensitivity and oracle-lean apparent gravity for one clip."""
    est = est_c2w[:, :3, 3]
    n = min(len(est), len(true_pos))
    est, true_pos = est[:n], true_pos[:n]
    rot, scale, trans = umeyama(est, true_pos, with_scale=True)

    up = up_axis(true_pos)
    axis = up["axis"]
    perm = [i for i in range(3) if i != axis] + [axis]

    out = {"n": int(n), "fps": float(fps), "scale_ratio": float(scale),
           "up_axis": axis, "up_unambiguous": up["unambiguous"]}

    # oracle lean: truth attitude gives the body z axis if it is the thrust axis; for a
    # camera-first dataset we use the truth *acceleration* lean instead, which is the same
    # physical quantity without an attitude convention
    dt = 1.0 / fps
    t_t, v_t, a_t = differentiate(true_pos, dt=dt, window_s=window_s)
    t_e, v_e, a_e = differentiate(scale * est @ rot.T + trans, dt=dt, window_s=window_s)
    m = min(len(a_e), len(a_t))
    lean_true = lean_xy(a_t[:m][:, perm])
    a_est_h = a_e[:m][:, perm][:, :2]                  # horizontal acceleration, extractor units
    fit = lean_regression(lean_true, a_est_h)          # l_true ~ a_est / g_app
    # a_e comes from the Sim(3)-aligned track, so it is already in metres: no rescale
    out["g_app_oracle_lean"] = float(fit["g_app"])
    out["r2_oracle_lean"] = fit["r2"]

    for factor in (2.0, 0.5):
        # dt assumed 1/factor of the truth rate -> acceleration scales by factor^2
        fit_w = lean_regression(lean_true, time_rescale(a_est_h, factor))
        out[f"g_app_fps_x{factor:g}"] = float(fit_w["g_app"])
        out[f"r2_fps_x{factor:g}"] = fit_w["r2"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, help="dir with <model>/<seq>.npz")
    ap.add_argument("--truth", required=True, help="dir with <seq>/truth.npz")
    ap.add_argument("--prereg", default="configs/pathrec_prereg.yaml")
    ap.add_argument("--out", default="results/pathrec_priors/v1")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.prereg).read_text())
    fit_seqs = set(cfg["testbeds"]["T1_uzh"]["split"]["fit"])
    eval_seqs = set(cfg["testbeds"]["T1_uzh"]["split"]["eval"])

    rows = []
    for model_dir in sorted(Path(args.poses).glob("*")):
        if not model_dir.is_dir() or model_dir.name.startswith("tii_"):
            continue
        for npz in sorted(model_dir.glob("*.npz")):
            seq = npz.stem
            meta_path = Path(args.truth) / seq / "meta.json"
            truth_path = Path(args.truth) / seq / "truth.npz"
            if not truth_path.exists():
                continue
            fps = json.loads(meta_path.read_text())["fps"] if meta_path.exists() else 15.0
            c2w = np.load(npz)["c2w"]
            tr = np.load(truth_path)
            rec = analyse_clip(c2w, tr["pos"], tr["quat"], fps)
            rec.update({"model": model_dir.name, "seq": seq,
                        "split": "fit" if seq in fit_seqs else ("eval" if seq in eval_seqs else "other")})
            rows.append(rec)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"rows": rows, "intervals": {}, "coverage": {}}
    for model in sorted({r["model"] for r in rows}):
        fit = [r["scale_ratio"] for r in rows if r["model"] == model and r["split"] == "fit"]
        ev = [r["scale_ratio"] for r in rows if r["model"] == model and r["split"] == "eval"]
        iv = interval(fit)
        summary["intervals"][model] = iv
        summary["coverage"][model] = {
            "n_eval": len(ev),
            "in_interval": int(sum(iv["low"] <= x <= iv["high"] for x in ev)) if iv["n"] else 0,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{'model':8s} {'seq':22s} {'split':5s} {'scale':>8s} {'g_app':>8s} {'r2':>6s} "
          f"{'g_x2':>8s} {'g_x0.5':>8s}")
    for r in rows:
        print(f"{r['model']:8s} {r['seq']:22s} {r['split']:5s} {r['scale_ratio']:8.4f} "
              f"{r.get('g_app_oracle_lean', float('nan')):8.2f} {r.get('r2_oracle_lean', float('nan')):6.2f} "
              f"{r.get('g_app_fps_x2', float('nan')):8.2f} {r.get('g_app_fps_x0.5', float('nan')):8.2f}")
    print(json.dumps(summary["intervals"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
