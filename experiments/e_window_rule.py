"""Track D's label-free required-lean window, implemented independently — c / f95.

Rule (their `configs/window_bandwidth_prereg.yaml`, commit 031eb14, written before this exists):
window_s = c / f95 with c = 0.5, where f95 is the frequency below which 95 % of the commanded
acceleration energy lies. Acceleration for the *spectrum* comes from a fixed Savitzky-Golay
differentiation (0.10 s, polyorder 3), independent of the required-lean window; the spectrum is the
rFFT of the mean-removed acceleration, energy summed over the three axes, DC excluded.

This script only computes windows. It reads nothing but the commanded path, so it cannot see a
condition label, an edit or a verdict outcome.

    python experiments/e_window_rule.py --out results/e_window_rule.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from src.depicted.uzh import all_sequences, is_forward, load_truth

C = 0.5
SPECTRUM_WINDOW_S = 0.10
SPECTRUM_POLY = 3


def f95_of_path(pos: np.ndarray, dt: float) -> dict:
    """f95 of the commanded acceleration energy, and the resulting window."""
    n = len(pos)
    w = int(round(SPECTRUM_WINDOW_S / dt))
    if w % 2 == 0:
        w += 1
    # at coarse rates the spectrum window is clamped to polyorder + 2 samples (their note)
    w = max(SPECTRUM_POLY + 2, min(w, n if n % 2 else n - 1))
    if w % 2 == 0:
        w += 1
    acc = savgol_filter(pos, w, SPECTRUM_POLY, deriv=2, delta=dt, axis=0, mode="interp")
    acc = acc - acc.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(acc, axis=0)
    energy = (np.abs(spec) ** 2).sum(axis=1)
    freqs = np.fft.rfftfreq(n, d=dt)
    keep = freqs > 0.0
    e, f = energy[keep], freqs[keep]
    if e.sum() <= 0:
        return {"f95_hz": float("nan"), "window_s": float("nan"), "spectrum_window_samples": w}
    cum = np.cumsum(e) / e.sum()
    i95 = int(np.searchsorted(cum, 0.95))
    i95 = min(i95, len(f) - 1)
    f95 = float(f[i95])
    return {"f95_hz": f95, "window_s": float(C / f95) if f95 > 0 else float("nan"),
            "spectrum_window_samples": w, "dt_s": float(dt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/uzh")
    ap.add_argument("--out", default="results/e_window_rule.json")
    ap.add_argument("--seqs", default="")
    args = ap.parse_args()
    root = Path(args.root)
    wanted = [s for s in args.seqs.split(",") if s] or all_sequences()
    recs = {}
    for seq in wanted:
        p = root / seq / "truth.npz"
        if not p.exists():
            continue
        gt = np.load(p)["gt"]
        t = gt[:, 0]
        dt = float(np.median(np.diff(t)))
        r = f95_of_path(gt[:, 1:4], dt)
        r.update({"seq": seq, "forward": bool(is_forward(seq)),
                  "truth_span_s": float(t[-1] - t[0]), "truth_rate_hz": 1.0 / dt,
                  # their floor read at the two rates: 8 samples at the truth rate, and at a
                  # 15 Hz clip rate (the schema-v1 reading)
                  "floor_truth_rate_s": 8.0 * dt,
                  "floor_clip_rate_samples": (r["window_s"] / (1 / 15.0)
                                              if np.isfinite(r["window_s"]) else None),
                  "abstain_at_clip_rate": bool(np.isfinite(r["window_s"])
                                               and (r["window_s"] / (1 / 15.0)) < 8)})
        recs[seq] = r
        print(f"{seq:20s} f95={r['f95_hz']:6.3f} Hz  window={r['window_s']:.3f} s "
              f"({r['window_s'] / dt:5.1f} samples at {1/dt:.0f} Hz)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"c": C, "spectrum_window_s": SPECTRUM_WINDOW_S,
                                          "per_sequence": recs}, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
