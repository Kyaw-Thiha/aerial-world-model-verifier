"""Render the Gate 3 tables from a verdict run into markdown (no numbers typed by hand).

    pixi run -e gpu python experiments/phase3a_gate3_tables.py results/phase3a_verdict/<run>

Keeping this in code is deliberate: every number in ``docs/track_a_verdict.md`` can be
regenerated from ``metrics.json``, and a typo in a table cannot survive a re-run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    run = Path(args.run)
    m = json.loads((run / "metrics.json").read_text())
    ev = m["evaluation"]
    f = ev["families"]

    lines = []
    lines.append(f"run `{m['run_id']}` · git `{m['git']}` · prereg `{m['prereg']['sha256'][:12]}` "
                 f"({m['prereg']['git_hash']}) · {m['n_records']} clip-condition records\n")
    lines.append("### Real clips\n")
    lines.append(f"- held-out acceptance ({ev['threshold_policy']} threshold policy): "
                 f"**{fmt(ev['heldout_acceptance'], 3)}** over {ev['heldout_n']} clips")
    lines.append(f"- pooled acceptance: {fmt(ev['pooled_acceptance'], 3)}")
    lines.append(f"- acceptance, non-trackRATM: {fmt(ev.get('heldout_acceptance_other'), 3)} "
                 f"(n={ev.get('n_other')}); trackRATM: "
                 f"{fmt(ev.get('heldout_acceptance_trackratm'), 3)} "
                 f"(n={ev.get('n_trackratm')})")
    lines.append(f"- abstention: overall {fmt(ev['abstention_rate']['overall'], 3)}, "
                 f"trackRATM {fmt(ev['abstention_rate']['trackratm'], 3)}\n")
    lines.append("### Thresholds (conformal 95th percentile)\n")
    lines.append("| speed bin group | bins merged | n | threshold (deg) |")
    lines.append("| --- | --- | --- | --- |")
    for k, v in ev["pooled_thresholds"]["per_bin"].items():
        lines.append(f"| {k} | {', '.join(v['bins'])} | {v['n']} | {fmt(v['threshold'])} |")
    lines.append(f"| pooled | - | {ev['pooled_thresholds']['pooled']['n']} | "
                 f"{fmt(ev['pooled_thresholds']['pooled']['threshold'])} |")
    counts = ev["pooled_thresholds"].get("raw_counts", {})
    if counts:
        lines.append("\nUnmerged per-bin counts: " +
                     ", ".join(f"{k} n={v}" for k, v in counts.items()) + "\n")
    lines.append("### Negatives\n")
    lines.append("| condition | family | n | rejection | AUROC | real score | negative score |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for cond, v in f.items():
        if not v.get("n"):
            continue
        lines.append(f"| {cond} | {v['family']} | {v['n']} | {fmt(v['rejection_rate'], 3)} | "
                     f"{fmt(v['auroc'], 3)} | {fmt(v['score_median_real'])} | "
                     f"{fmt(v['score_median_negative'])} |")
    lines.append(f"\nPooled AUROC over the three tilt families: "
                 f"**{fmt(ev.get('family_auroc_pooled'), 3)}**\n")
    if ev.get("mdv"):
        lines.append("### Minimum detectable violation\n")
        lines.append("| edit | k | injected tilt change (deg) | rejection |")
        lines.append("| --- | --- | --- | --- |")
        for cond, v in ev["mdv"].items():
            lines.append(f"| {cond} | {v['k']} | {fmt(v['violation_median_deg'], 1)} | "
                         f"{fmt(v['rejection_rate'], 3)} |")
        lines.append(f"\nSmallest violation rejected 80% of the time: "
                     f"**{fmt(ev.get('mdv_smallest_rejected80_deg'), 1)} deg**\n")
    if ev.get("delay_negatives"):
        lines.append("### Delay negatives (analysis, no pixel edit)\n")
        lines.append("| delay (ms) | n | score median | note |")
        lines.append("| --- | --- | --- | --- |")
        for k, v in ev["delay_negatives"].items():
            note = "inside the +/-150 ms nuisance bound" if int(k) <= 150 else "outside the bound"
            lines.append(f"| {k} | {v['n']} | {fmt(v['score_median'])} | {note} |")
        lines.append("")
    if ev.get("retime_negatives"):
        lines.append("### Retime negatives (analysis)\n")
        lines.append("| rate | n | score median |")
        lines.append("| --- | --- | --- |")
        for k, v in ev["retime_negatives"].items():
            lines.append(f"| {k} | {v['n']} | {fmt(v['score_median'])} |")
        lines.append("")
    d = ev.get("diagnostics", {})
    if d:
        lines.append("### Diagnostics\n")
        lines.append("| speed bin | n | real median | real p90 |")
        lines.append("| --- | --- | --- | --- |")
        for k, v in d.get("real_by_speed_bin", {}).items():
            lines.append(f"| {k} | {v['n']} | {fmt(v['median'])} | {fmt(v['p90'])} |")
        lines.append("")
        lines.append("| condition | n | score median | uptilt offset median | at bound | "
                     "lag median (ms) | at bound | abstain |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for k, v in d.get("per_condition", {}).items():
            lines.append(f"| {k} | {v['n']} | {fmt(v['score_median'])} | "
                         f"{fmt(v['uptilt_offset_median_deg'])} | "
                         f"{fmt(v['uptilt_offset_at_bound_frac'], 3)} | "
                         f"{fmt(v['lag_median_ms'], 1)} | {fmt(v['lag_at_bound_frac'], 3)} | "
                         f"{fmt(v['abstain_frac'], 3)} |")
        lines.append("")
    g = ev.get("gate3", {})
    lines.append("### Gate 3\n")
    for k, v in g.items():
        lines.append(f"- {k}: **{v}**")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"-> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
