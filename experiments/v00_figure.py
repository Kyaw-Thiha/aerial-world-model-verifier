"""Track V: the Part 0 evidence figure (CPU, reads stored recon).

Two panels for one real sequence: the depicted camera height above the fitted ground plane
against time, the truth altitude on the same axis (left), and the per-chunk relation between
them (right). It is the picture behind the "the ground normal, not the reconstruction, is the
binding constraint" claim.

Usage
-----
PYTHONPATH=. python experiments/v00_figure.py [unit]
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.vertical import recon_io as RIO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "data/v/recon"
OUT = ROOT / "results/v/v00"


def main(unit: str = "outdoor_forward_1") -> int:
    units = {u["unit"]: u for u in json.loads((ROOT / "data/v/units.json").read_text())["units"]}
    u = units[unit]
    data = RIO.load_unit(RECON, unit)
    tt = np.asarray(u["truth"]["t"], float)
    pp = np.asarray(u["truth"]["pos"], float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, ch in sorted(data["chunks"].items()):
        t = ch["t"]
        c = ch["c2w"][:, :3, 3]
        w = min(data["meta"]["windows"],
                key=lambda w: abs(0.5 * (w["t1"] + w["t2"]) - 0.5 * (t[0] + t[-1])))
        if not w.get("plane"):
            continue
        n = np.asarray(w["plane"]["n"], float)
        d = float(w["plane"]["d"])
        h = c @ n - d
        z = np.interp(t, tt, pp[:, 2])
        ax[0].plot(t, h, "-", lw=1, label=f"chunk {k}" if k < 3 else None)
        ax[1].plot(z - z.mean(), h - h.mean(), ".", ms=2)
    ax[0].plot(tt, (pp[:, 2] - pp[0, 2]), "k-", lw=1.2, label="truth (offset)")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("camera height above fitted ground\n(reconstruction units)")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title(f"{unit}: depicted height vs time")
    ax[1].set_xlabel("truth altitude, centred (m)")
    ax[1].set_ylabel("depicted height, centred (units)")
    ax[1].set_title("per frame: the relation is a per-chunk offset")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fig_depicted_vs_truth__{unit}.png"
    fig.savefig(path, dpi=130)
    print(path)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(*sys.argv[1:]))
