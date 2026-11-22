"""H01: emit the report's tables from the stored result files (no recomputation).

    pixi run python experiments/h01_report_tables.py

Writes ``results/h01_tables/*.csv`` and prints the same tables in Markdown so the numbers in
``reports/H01_generator_pilot.md`` can be copied from a single generated source.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/h01_tables"


def md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            if not np.isfinite(v):
                return "—"
            return floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join([" --- " for _ in df.columns]) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    printed = []

    # ---------------- Part 1
    seed_dir = ROOT / "results/h01_seed_variance"
    if (seed_dir / "summary.json").exists():
        s = json.loads((seed_dir / "summary.json").read_text())
        sp = pd.read_csv(seed_dir / "per_path_spread.csv")
        bn = pd.read_csv(seed_dir / "best_of_n.csv")
        t1 = sp[["base_path", "maneuver", "designed_tilt_deg", "n_seeds",
                 "depicted_median_deg", "depicted_seed_sd_deg", "depicted_seed_range_deg",
                 "residual_median_deg", "residual_seed_range_deg", "n_feasible_seeds"]].copy()
        t1.columns = ["path", "maneuver", "designed_deg", "seeds", "depicted_median_deg",
                      "depicted_seed_sd_deg", "depicted_seed_range_deg", "residual_median_deg",
                      "residual_seed_range_deg", "feasible_seeds"]
        t1.to_csv(OUT / "t1_seed_variance.csv", index=False)
        printed.append("### T1 — per-path seed spread\n\n" + md(t1, "{:.2f}"))
        t2 = bn.copy()
        t2.to_csv(OUT / "t2_best_of_n.csv", index=False)
        printed.append("### T2 — oracle best-of-N\n\n" + md(t2, "{:.2f}"))
        printed.append("```json\n" + json.dumps(
            {k: s[k] for k in ("depicted_spread", "residual_spread",
                               "oracle_reduction_1_to_5", "design_contrast")
             if k in s}, indent=1) + "\n```")

    # ---------------- Part 2
    for tag, path in (("lingbot", "results/h01_lingbot"),
                      ("lingbot_b", "results/h01_lingbot_b"),
                      ("wan", "results/h01_wan")):
        d = ROOT / path
        if not (d / "clips.csv").exists():
            continue
        c = pd.read_csv(d / "clips.csv")
        if "arm" in c and c["arm"].notna().any():
            agg = (c.groupby(["path", "arm"], dropna=False)
                   .agg(n=("clip_id", "size"),
                        depicted_median_deg=("depicted_tilt_median_deg", "median"),
                        injected_median_deg=("injected_tilt_median_deg", "median"),
                        lean_cos=("lean_direction_cos_median", "median"))
                   .reset_index())
            agg.to_csv(OUT / f"t3_{tag}_by_arm.csv", index=False)
            printed.append(f"### T3 — {tag} depicted tilt by arm\n\n" + md(agg, "{:.2f}"))
        summ = json.loads((d / "summary.json").read_text())
        printed.append(f"### {tag} summary\n\n```json\n" + json.dumps(
            {k: summ.get(k) for k in ("by_arm", "flatness_minus_level_deg", "mirror_flip")},
            indent=1) + "\n```")
        c.to_csv(OUT / f"t4_{tag}_clips.csv", index=False)

    print("\n\n".join(printed))
    (OUT / "tables.md").write_text("\n\n".join(printed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
