"""H02: frozen commanded-path verdict on SolarWM flatness clips.

For each flatness-arm grid clip: required tilt from the designed path (dt = 1 s, k_d = 0.295,
per-step acceleration), interpolated to the 81 generated frames; depicted tilt from the H02
reader; residual = median |required - depicted|; feasible when <= 6.46 deg. Testable when the
frozen gate keeps the clip, it is not collapsed and at least 12 frames require >= 10 deg
(H02's absolute pre-registered rule; H01's was 12 of 16 frames, and the CSV also carries the
strict 12/16-fraction sensitivity).

    pixi run python experiments/h02_verdict.py --depicted results/h02_solarwm_depicted/depicted.jsonl \
        --out results/h02_solarwm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.a2_anwm_required import frame_required
from experiments.a2_anwm_score import frame_arrays, gate_of

ROOT = Path(__file__).resolve().parents[1]
K_D = 0.295
DT = 1.0
THRESH = 6.46


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def required_profile(spec: dict, n: int = 81) -> np.ndarray:
    """H01's frame-required tilt, interpolated to the generated frames.

    ``frame_required(..., "step")`` evaluates the acceleration that produced each path
    frame (node 1..N-1, as in H01); the endpoints have no predecessor and are NaN.  We
    interpolate only the finite nodes and hold the first/last value at the two ends,
    rather than inserting zero-tilt endpoint nodes, which would understate the required
    tilt on the first and last frames.
    """
    tilt = frame_required(np.asarray(spec["positions"], float), DT, K_D, "step")
    good = np.isfinite(tilt)
    return np.interp(np.linspace(0, len(tilt) - 1, n),
                     np.arange(len(tilt))[good], tilt[good])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depicted", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", default="configs/gate_frozen.yaml")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((ROOT / args.gate).read_text())
    specs = {s["id"]: s for s in json.loads(
        (ROOT / "results/anwm_clips/specs.json").read_text())}
    man = json.loads((ROOT / "data/h02_solarwm_inputs/manifest.json").read_text())
    meta = {}
    for sid in man["samples"]:
        mp = ROOT / "data/h02_solarwm_inputs" / sid / "meta.json"
        if mp.exists():
            meta[sid] = json.loads(mp.read_text())
    rows = load_jsonl(ROOT / args.depicted)
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)

    # depicted tilt per clip from the analysis output if present, else from gravity rows
    from experiments.a2_anwm_score import depicted_tilt_deg
    out = []
    for sid, rs in sorted(by_clip.items()):
        m = meta.get(sid, {})
        if m.get("arm") != "flatness" or "source_clip" not in m:
            continue
        rs = sorted(rs, key=lambda r: int(Path(r["frame"]).stem))
        dep = depicted_tilt_deg(frame_arrays(rs)["up"])
        req = required_profile(specs[m["source_clip"]], len(dep))
        n = min(len(dep), len(req))
        dep, req = dep[:n], req[:n]
        val, ok = gate_of(cfg, rs)
        resid = float(np.nanmedian(np.abs(req - dep)))
        out.append({"clip_id": sid, "source_clip": m["source_clip"],
                    "frame": m.get("frame"), "path": m.get("path"),
                    "n_frames": n, "required_median_deg": float(np.nanmedian(req)),
                    "depicted_median_deg": float(np.nanmedian(dep)),
                    "residual_median_deg": resid,
                    "frac_required_ge10": float(np.nanmean(req >= 10.0)),
                    "gate_signal": float(val) if np.isfinite(val) else None,
                    "gate_kept": bool(ok),
                    # H02 pre-registers an absolute minimum of 12 frames.  H01's rule
                    # was written for 16-frame clips (12/16 = 75 %); applying the same
                    # fraction to 81 frames gives the stricter sensitivity flag.
                    "testable": bool(ok and int(np.sum(req >= 10.0)) >= 12),
                    "testable_strict": bool(ok and int(np.sum(req >= 10.0))
                                            >= int(np.ceil(0.75 * n))),
                    "feasible": bool(resid <= THRESH)})
    o = ROOT / args.out
    o.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out)
    df.to_csv(o / "verdict.csv", index=False)
    summary = {"n_clips": len(df),
               "testable_k": int(df.testable.sum()) if len(df) else 0,
               "feasible_k": int((df.testable & df.feasible).sum()) if len(df) else 0,
               "abstain_gate": int((~df.gate_kept).sum()) if len(df) else 0,
               "strict_testable_k": int(df.testable_strict.sum()) if len(df) else 0,
               "strict_feasible_k": int((df.testable_strict & df.feasible).sum())
               if len(df) else 0,
               "threshold_deg": THRESH}
    (o / "verdict_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(df.to_string(index=False)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
