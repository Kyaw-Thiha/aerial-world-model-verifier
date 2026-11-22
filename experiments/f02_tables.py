"""Render the F02 tables: bake-off scores per extractor and the stage-0 statistics on the best.

    python experiments/f02_tables.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDS = ("real", "null_edit", "yaw_control", "tilt_removal", "mirror_roll",
         "scale_0.5", "scale_1.5")
ORDER = ("dpvo", "vipe", "mast3r")


def fmt(r: dict) -> str:
    if not r or r.get("rate") is None:
        return "--"
    return f"{r['rate']:.3f} ({r['k']}/{r['n']})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/f02_stage1/scores.json")
    ap.add_argument("--stage0", default="results/f02_stage1/stage0_vipe/summary.json")
    args = ap.parse_args()
    scores = json.loads(Path(args.scores).read_text())

    lines = ["| extractor | sequence | direction error (deg) | magnitude ratio | "
             "coverage | ATE Sim(3) (m) | pass |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for method in ORDER:
        block = scores["methods"].get(method, {})
        if "rows" not in block:
            continue
        for seq, r in block["rows"].items():
            lines.append(f"| {method} | {seq} | {r['accel_dir_err_deg_median']:.1f} | "
                         f"{r['accel_mag_ratio']:.2f} | {r['coverage_fraction']:.2f} | "
                         f"{r['ate_sim3_m']:.2f} | {'yes' if r['pass'] else 'no'} |")
        lines.append(f"| **{method} total** | **{block['n_pass']}/{block['n_sequences']}** | "
                     f"**{block['median_dir_err_deg']:.1f}** | "
                     f"**{block['median_mag_ratio']:.2f}** | | | |")
    out = Path(args.scores).parent
    (out / "table_bakeoff.md").write_text("\n".join(lines) + "\n")

    s0 = Path(args.stage0)
    if s0.exists():
        d = json.loads(s0.read_text())
        rows = ["", "Stage-0 statistics with the ViPE path (estimated attitude, primary "
                "8 clips, LOSO thresholds):", "",
                "| variant | real | removal | mirror | x1.5 |", "| --- | --- | --- | --- | --- |"]
        for v in ("V1_preintegration", "V2_shooting", "V3_band_coherence",
                  "V4_velocity_drag", "V5_physics_scale"):
            b = d["summary"][v]["estimated_attitude"]["pixel"]["primary"]["rates"]
            rows.append(f"| {v} | {fmt(b['real'])} | {fmt(b['tilt_removal'])} | "
                        f"{fmt(b['mirror_roll'])} | {fmt(b['scale_1.5'])} |")
        (out / "table_stage0_vipe.md").write_text("\n".join(rows) + "\n")
        print("\n".join(rows))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
