"""Track G3: render the result CSVs as the markdown tables used in report G06."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    if not Path(path).is_file():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def f(x, nd=2):
    if x in (None, "", "None", "nan"):
        return "—"
    try:
        v = float(x)
    except ValueError:
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{nd}f}"


def pct(k, n):
    try:
        k, n = int(k), int(n)
    except (TypeError, ValueError):
        return "—"
    return f"{k}/{n} ({100*k/n:.1f} %)" if n else f"{k}/0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/g06/ablation")
    ap.add_argument("--yaw", default="results/g06/yaw_slice/yaw_summary.json")
    ap.add_argument("--consistency", default="results/g06/consistency/consistency.json")
    args = ap.parse_args()
    d = Path(args.dir)
    out = []

    frr = read(d / "false_reject.csv")
    out.append("## FRR (feasible clips) by arm and family\n")
    out.append("| arm | family | flagged / testable | rate | false seg / min |")
    out.append("| --- | --- | --- | --- | --- |")
    for r in frr:
        out.append(f"| {r['arm']} | {r['family']} | {pct(r['k'], r['n_testable'])} | "
                   f"{f(r['rate'],3)} | {f(r['false_segments_per_min'])} |")

    det = read(d / "detection.csv")
    arms = list(dict.fromkeys(r["arm"] for r in det))
    out.append("\n## A1 detection, full-clip defects (fraction 1.0)\n")
    out.append("| arm | family | defect | k/n | rate | IoU (all) | IoU (detected) | F1 | onset (frames) |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in det:
        if r["fraction"] != "1.0":
            continue
        out.append(f"| {r['arm']} | {r['family']} | {r['defect']} | {pct(r['k'], r['n_testable'])} "
                   f"| {f(r['rate'],3)} | {f(r['iou_med'])} | {f(r['iou_med_detected'])} | "
                   f"{f(r['f1_med'])} | {f(r['onset_med'],1)} |")

    out.append("\n## A1 detection, partial intervals\n")
    out.append("| arm | family | defect | fraction | k/n | rate | IoU |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in det:
        if r["fraction"] in ("1.0", "0"):
            continue
        out.append(f"| {r['arm']} | {r['family']} | {r['defect']} | {r['fraction']} | "
                   f"{pct(r['k'], r['n_testable'])} | {f(r['rate'],3)} | {f(r['iou_med'])} |")

    matched = read(d / "detection_matched.csv")
    out.append("\n## Matched false-reject, full-clip defects\n")
    out.append("| arm | family | defect | k/n | rate |")
    out.append("| --- | --- | --- | --- | --- |")
    for r in matched:
        if r["fraction"] != "1.0":
            continue
        out.append(f"| {r['arm']} | {r['family']} | {r['defect']} | "
                   f"{pct(r['k'], r['n_testable'])} | {f(r['rate'],3)} |")

    realised = read(d / "a2_fdr_realised.csv")
    out.append("\n## A2 FDR — realised FDP and false segments\n")
    out.append("| arm | q | mode | FDP | windows flagged | false seg / min |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for r in realised:
        out.append(f"| {r['arm']} | {r['q']} | {r['mode']} | {f(r['fdp'],3)} | "
                   f"{r['windows_flagged']} | {f(r['false_segments_per_min'])} |")

    fdr = read(d / "a2_fdr_rates.csv")
    out.append("\n## A2 detection\n")
    out.append("| arm | q | mode | family | defect | fraction | k/n | rate |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in fdr:
        if r["fraction"] != "1.0" and float(r["fraction"] or 0) not in (0.1, 0.25, 0.5):
            continue
        out.append(f"| {r['arm']} | {r['q']} | {r['mode']} | {r['family']} | {r['defect']} | "
                   f"{r['fraction']} | {pct(r['k'], r['n'])} | {f(r['rate'],3)} |")

    b = read(d / "a3_boundaries.csv")
    if b:
        out.append("\n## A3 boundary refinement\n")
        out.append("| clips | median IoU before | median IoU after | mean ΔIoU | median ΔF1 | median onset before | after |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        import numpy as np
        ib = [float(r["iou_before"]) for r in b]
        ia = [float(r["iou_after"]) for r in b]
        f1b = [float(r["f1_before"]) for r in b]
        f1a = [float(r["f1_after"]) for r in b]
        ob = [float(r["onset_before"]) for r in b]
        oa = [float(r["onset_after"]) for r in b]
        out.append(f"| {len(b)} | {np.median(ib):.2f} | {np.median(ia):.2f} | "
                   f"{np.mean(np.array(ia)-np.array(ib)):+.2f} | "
                   f"{np.median(np.array(f1a)-np.array(f1b)):+.2f} | "
                   f"{np.median(ob):.1f} | {np.median(oa):.1f} |")

    lags = read(d / "a4_lags.csv")
    if lags:
        import numpy as np
        by = defaultdict(list)
        for r in lags:
            by[r["family"]].append(float(r["lag_s"]))
        out.append("\n## A4 chosen lag by family\n")
        out.append("| family | clips | median lag (ms) | IQR (ms) |")
        out.append("| --- | --- | --- | --- |")
        for k, v in sorted(by.items()):
            v = np.array(v)
            out.append(f"| {k} | {len(v)} | {1000*np.median(v):+.0f} | "
                       f"{1000*(np.percentile(v,75)-np.percentile(v,25)):.0f} |")

    al = read(d / "a4_alignment.csv")
    if al:
        out.append("\n## A4 alignment effect\n")
        out.append("| arm | family | defect | fraction | k | n | rate |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in al:
            if r["defect"] in ("false_segments", "mean_iou"):
                out.append(f"| {r['arm']} | {r['family']} | {r['defect']} | | "
                           f"{f(r['k'],3)} | {f(r['n'],1)} | — |")
            else:
                out.append(f"| {r['arm']} | {r['family']} | {r['defect']} | {r['fraction']} | "
                           f"{r['k']} | {r['n']} | {f(r['rate'],3)} |")

    ms = read(d / "a5_multiscale.csv")
    if ms:
        out.append("\n## A5 multi-scale\n")
        out.append("| arm | family | defect | fraction | k | n | rate |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in ms:
            if r["defect"] in ("false_segments", "mean_iou"):
                out.append(f"| {r['arm']} | {r['family']} | {r['defect']} | | "
                           f"{f(r['k'],3)} | {f(r['n'],1)} | — |")
            else:
                out.append(f"| {r['arm']} | {r['family']} | {r['defect']} | {r['fraction']} | "
                           f"{r['k']} | {r['n']} | {f(r['rate'],3)} |")

    mdv = read(d / "mdv.csv")
    if mdv:
        out.append("\n## Minimum detectable step (synthetic, per arm)\n")
        for arm in dict.fromkeys(r["arm"] for r in mdv):
            rows = [r for r in mdv if r["arm"] == arm]
            durs = sorted({float(r["duration_s"]) for r in rows})
            deltas = sorted({float(r["delta_deg"]) for r in rows})
            out.append(f"\n{arm}\n")
            out.append("| Δ (deg) | " + " | ".join(f"{dd:g} s" for dd in durs) + " |")
            out.append("| --- | " + " | ".join("---" for _ in durs) + " |")
            for dl in deltas:
                cells = []
                for dd in durs:
                    m = [r for r in rows if float(r["delta_deg"]) == dl and float(r["duration_s"]) == dd]
                    cells.append(f(m[0]["rate"], 2) if m else "—")
                out.append(f"| {dl:g} | " + " | ".join(cells) + " |")

    mdvf = read(d / "mdv_fdr_exploratory.csv")
    if mdvf:
        out.append("\n## Minimum detectable step with BH (EXPLORATORY)\n")
        durs = sorted({float(r["duration_s"]) for r in mdvf})
        deltas = sorted({float(r["delta_deg"]) for r in mdvf})
        out.append("| Δ (deg) | " + " | ".join(f"{dd:g} s" for dd in durs) + " |")
        out.append("| --- | " + " | ".join("---" for _ in durs) + " |")
        for dl in deltas:
            cells = []
            for dd in durs:
                m = [r for r in mdvf if float(r["delta_deg"]) == dl and float(r["duration_s"]) == dd]
                cells.append(f(m[0]["rate"], 2) if m else "—")
            out.append(f"| {dl:g} | " + " | ".join(cells) + " |")

    y = json.loads(Path(args.yaw).read_text()) if Path(args.yaw).is_file() else {}
    if y:
        out.append("\n## Yaw slice\n")
        out.append(f"- clips {y['n_clips']}, testable null {y['testable']['null']}, yaw {y['testable']['yaw']}")
        out.append(f"- flagged null {y['flagged']['null']}, yaw {y['flagged']['yaw']}")
        ls = y["lean_shift_deg"]
        out.append(f"- reader yaw-induced lean shift: median {f(ls['median_of_clip_medians'],2)} deg, "
                   f"p90 {f(ls['p90_of_clip_p90'],2)}, max {f(ls['max'],2)} (real-clip reference 4.75)")

    c = json.loads(Path(args.consistency).read_text()) if Path(args.consistency).is_file() else {}
    if c:
        out.append("\n## Consistency\n")
        out.append(f"- clips {c['clips_compared']}, frames {c['frames_compared']}, "
                   f"component median {c['component_abs_d_lean']['median']:.2e}, "
                   f"p99 {c['component_abs_d_lean']['p99']:.2e}, "
                   f"max {c['component_abs_d_lean']['max']:.2f}, "
                   f"frac>0.05 {c['component_abs_d_lean']['frac_above_0_05']:.3f}")

    text = "\n".join(out)
    (d / "tables.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
