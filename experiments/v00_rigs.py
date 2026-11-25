"""Track V: the exploratory per-rig table (all 14 UZH sequences).

NOT the confirmatory Part 2 (the pre-registered stop fired, so V1 is not run). This is the
mechanism diagnostic the report uses: the same estimator and statistic on every UZH rig, to show
that the failure is a property of the viewing geometry (low camera, forward view) rather than of
one sequence.

Usage
-----
PYTHONPATH=. python experiments/v00_rigs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.vertical import metrics as M
from src.vertical import recon_io as RIO

ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "data/v/recon"
OUT = ROOT / "results/v/v00"


def main() -> int:
    units = [u for u in json.loads((ROOT / "data/v/units.json").read_text())["units"]
             if u["kind"] == "uzh"]
    rows_out = []
    for u in sorted(units, key=lambda x: x["unit"]):
        if not (RECON / f"{u['unit']}.json").exists():
            continue
        data = RIO.load_unit(RECON, u["unit"])
        t = np.asarray(u["truth"]["t"], float)
        pos = np.asarray(u["truth"]["pos"], float)
        rows, _ = RIO.unit_rows(data, t, pos, mode="real")
        ok = [r for r in rows if r.get("ok")]
        if not ok:
            rows_out.append({"sequence": u["unit"], "n_ok": 0, "n_windows": len(rows)})
            continue
        e = np.array([r["abs_e"] for r in ok])
        h1 = np.array([r.get("scale_s", np.nan) * r["h1"] for r in ok], float)
        lev = np.array([r["L_truth"] / h for r, h in zip(ok, h1) if h > 0])
        delta = np.array([np.degrees(abs(r["e"]) * h / r["L_truth"])
                          for r, h in zip(ok, h1) if h > 0 and r["L_truth"] > 0])
        rows_out.append({
            "sequence": u["unit"], "rig": u["unit"].split("_")[0] + "_" +
            ("45" if "45" in u["unit"] else "forward"),
            "n_ok": len(ok), "n_windows": len(rows),
            "coverage": len(ok) / max(1, len(rows)),
            "median_abs_e": float(np.median(e)), "p90_abs_e": float(np.percentile(e, 90)),
            "median_h1_m": float(np.median(h1)), "median_L_over_h": float(np.median(lev)),
            "median_implied_delta_deg": float(np.median(delta)),
            "median_dz_truth": float(np.median([r["dz_truth"] for r in ok]))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rigs.json").write_text(json.dumps(rows_out, indent=1))
    lines = ["# Exploratory: the statistic on every UZH rig (not confirmatory)", "",
             "| sequence | windows | coverage | median abs e | p90 | median h1 (m) | L/h | "
             "implied delta (deg) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows_out:
        lines.append(f"| {r['sequence']} | {r.get('n_ok', 0)}/{r.get('n_windows', 0)} | "
                     f"{r.get('coverage', float('nan')):.2f} | "
                     f"{r.get('median_abs_e', float('nan')):.3f} | "
                     f"{r.get('p90_abs_e', float('nan')):.2f} | "
                     f"{r.get('median_h1_m', float('nan')):.2f} | "
                     f"{r.get('median_L_over_h', float('nan')):.1f} | "
                     f"{r.get('median_implied_delta_deg', float('nan')):.1f} |")
    (OUT / "rigs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
