"""Render Track F stage-0 results into report tables (markdown + LaTeX fragments).

Reads ``results/f01_stage0/summary.json`` and writes:
  results/f01_stage0/table_stage0.md   the report table (markdown)
  report_candidates/tab_e11_pixel-path-verdict_1.tex
  report_candidates/tab_e11_pixel-path-verdict_2.tex

    python experiments/f01_tables.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDS = ("real", "null_edit", "yaw_control", "tilt_removal", "mirror_roll",
         "scale_0.5", "scale_1.5")
SHORT = {"real": "real", "null_edit": "null", "yaw_control": "yaw",
         "tilt_removal": "removal", "mirror_roll": "mirror",
         "scale_0.5": "x0.5", "scale_1.5": "x1.5"}
VARIANTS = ("V1_preintegration", "V2_shooting", "V3_band_coherence",
            "V4_velocity_drag", "V5_physics_scale")
LABEL = {"V1_preintegration": "V1 preintegration",
         "V2_shooting": "V2 shooting (primary)",
         "V3_band_coherence": "V3 band coherence",
         "V4_velocity_drag": "V4 velocity-drag",
         "V5_physics_scale": "V5 physics scale"}
ARMS = (("estimated_attitude", "pixel", "estimated attitude, pixel path"),
        ("oracle_attitude", "pixel", "oracle attitude, pixel path"),
        ("estimated_attitude", "oracle", "estimated attitude, oracle path"),
        ("oracle_attitude", "oracle", "oracle attitude, oracle path"))


def _cell(r: dict) -> str:
    if not r or r.get("rate") is None:
        return "--"
    return f"{r['rate']:.3f} ({r['k']}/{r['n']})"


def _ci(block: dict, cond: str) -> str:
    c = block["clustered"].get(cond) or {}
    if "lo" not in c:
        return "--"
    return f"{c['lo']:.2f}-{c['hi']:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/f01_stage0/summary.json")
    ap.add_argument("--outdir", default="report_candidates")
    args = ap.parse_args()
    data = json.loads(Path(args.summary).read_text())
    summary = data["summary"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("| variant | arm / path | real FRR | null | yaw | removal | mirror | x0.5 | "
              "x1.5 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for variant in VARIANTS:
        for arm, path, label in ARMS:
            block = summary[variant][arm][path]["primary"]
            cells = [_cell(block["rates"][c]) for c in CONDS]
            md.append(f"| {LABEL[variant]} | {label} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("Sequence-clustered 95 % intervals over the primary sequences "
              "(descriptive, 4 clusters):")
    md.append("")
    md.append("| variant | real | removal | mirror |")
    md.append("| --- | --- | --- | --- |")
    for variant in VARIANTS:
        arm, path = "estimated_attitude", "pixel"
        block = summary[variant][arm][path]["primary"]
        md.append(f"| {LABEL[variant]} | {_ci(block, 'real')} | "
                  f"{_ci(block, 'tilt_removal')} | {_ci(block, 'mirror_roll')} |")
    (Path(args.summary).parent / "table_stage0.md").write_text("\n".join(md) + "\n")

    # LaTeX table 1: estimated attitude, pixel path (the gate arm)
    def tex_table(arm: str, path: str, path_label: str, name: str) -> None:
        lines = [r"\footnotesize", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                 r"variant & real & null & yaw & removal & mirror & $0.5\times$ & "
                 r"$1.5\times$ \\", r"\midrule"]
        for variant in VARIANTS:
            block = summary[variant][arm][path]["primary"]
            cells = []
            for c in ("real", "null_edit", "yaw_control", "tilt_removal", "mirror_roll",
                      "scale_0.5", "scale_1.5"):
                r = block["rates"][c]
                cells.append("--" if not r or r.get("rate") is None
                             else f"{r['rate']:.3f} ({r['k']}/{r['n']})")
            lines.append(f"{LABEL[variant]} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (outdir / name).write_text("\n".join(lines) + "\n")

    tex_table("estimated_attitude", "pixel", "pixel", "tab_e11_pixel-path-verdict_1.tex")

    # LaTeX table 2: the oracle ladder, V2 only, with clustered intervals
    lines = [r"\footnotesize", r"\begin{tabular}{llrrr}", r"\toprule",
             r"path & attitude & real & removal & mirror \\", r"\midrule"]
    for arm, path, label in ARMS:
        block = summary["V2_shooting"][arm][path]["primary"]
        row = []
        for c in ("real", "tilt_removal", "mirror_roll"):
            r = block["rates"][c]
            row.append("--" if not r or r.get("rate") is None
                       else f"{r['rate']:.3f} ({r['k']}/{r['n']})")
        lines.append(f"{path} & {arm.replace('_', ' ')} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "tab_e11_pixel-path-verdict_2.tex").write_text("\n".join(lines) + "\n")

    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
