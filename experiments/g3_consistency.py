"""Track G3: stride-1 versus stride-2 reader consistency check.

Compares the new stride-1 score files against Track I's published score files at every frame
index they share.  Track I mixed stride 1 (8 clips) and stride 2 (78 clips); the new run is
stride 1 for everything.  GeoCalib is deterministic on a given frame and both runs use the same
weights, preprocessing and focal prior, so the expectation is exact equality.  A non-zero
difference is a decode-path difference (seek vs sequential H.264), reported as found.

    pixi run python experiments/g3_consistency.py \
        --new results/g06/scores --old ~/Documents/Projects/awmv-track-i/results/i02/scores \
        --out results/g06/consistency
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[int, dict]:
    d = json.loads(path.read_text())
    return {int(r["frame"]): r for r in d["rows"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="results/g06/scores")
    ap.add_argument("--old",
                    default="/home/kevin/Documents/Projects/awmv-track-i/results/i02/scores")
    ap.add_argument("--out", default="results/g06/consistency")
    ap.add_argument("--repeat-a", default="", help="dir of an independent run of the same clips")
    ap.add_argument("--repeat-b", default="", help="dir of a second independent run")
    args = ap.parse_args()

    new_dir, old_dir = Path(args.new), Path(args.old)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, table, all_dl = [], [], []
    for nf in sorted(new_dir.glob("*_score.json")):
        cid = nf.name.replace("_score.json", "")
        of = old_dir / f"{cid}_score.json"
        if not of.is_file():
            continue
        new, old = load(nf), load(of)
        common = sorted(set(new) & set(old))
        if not common:
            continue
        dl = np.array([[new[i]["lean_est_x"] - old[i]["lean_est_x"],
                        new[i]["lean_est_y"] - old[i]["lean_est_y"]] for i in common])
        all_dl.append(np.abs(dl).ravel())
        dt = np.array([new[i]["tilt_est_deg"] - old[i]["tilt_est_deg"] for i in common])
        du = np.array([new[i]["up_error_deg"] - old[i]["up_error_deg"] for i in common])
        old_stride = 1 if len(old) > 0.75 * len(new) else 2
        rows.append({
            "clip_id": cid, "old_stride": old_stride,
            "n_new": len(new), "n_old": len(old), "n_common": len(common),
            "max_abs_d_lean": float(np.abs(dl).max()),
            "median_abs_d_lean": float(np.median(np.abs(dl))),
            "max_abs_d_tilt_deg": float(np.abs(dt).max()),
            "max_abs_d_up_error_deg": float(np.abs(du).max()),
            "all_exact_1e9": bool(np.abs(dl).max() < 1e-9),
        })

    arr = np.array([r["max_abs_d_lean"] for r in rows])
    by_stride = {}
    for s in (1, 2):
        sub = [r for r in rows if r["old_stride"] == s]
        if sub:
            by_stride[s] = {
                "clips": len(sub),
                "frames_compared": int(sum(r["n_common"] for r in sub)),
                "max_abs_d_lean": float(max(r["max_abs_d_lean"] for r in sub)),
                "clips_outside_1e9": int(sum(not r["all_exact_1e9"] for r in sub)),
            }
    def pair_stats(da: Path, db: Path) -> dict:
        vals = []
        for nf in sorted(da.glob("*_score.json")):
            cid = nf.name.replace("_score.json", "")
            other = db / nf.name
            if not other.is_file():
                continue
            a, b = load(nf), load(other)
            common = sorted(set(a) & set(b))
            if not common:
                continue
            dl = np.array([[a[i]["lean_est_x"] - b[i]["lean_est_x"],
                            a[i]["lean_est_y"] - b[i]["lean_est_y"]] for i in common])
            vals.append(np.abs(dl).ravel())
        if not vals:
            return {}
        v = np.concatenate(vals)
        return {"components": int(v.size), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)),
                "max": float(v.max())}

    repeat = {}
    if args.repeat_a and args.repeat_b:
        repeat = pair_stats(Path(args.repeat_a), Path(args.repeat_b))
        # the same statistic on the new-vs-old pairing, for a like-for-like comparison
        repeat["new_vs_old_median_of_component_abs"] = float(
            np.median(np.concatenate([np.array([r["median_abs_d_lean"]]) for r in rows])))
    v = np.concatenate(all_dl) if all_dl else np.array([])
    summary = {
        "repeat_run_envelope": repeat,
        "component_abs_d_lean": {
            "n": int(v.size), "median": float(np.median(v)) if v.size else None,
            "p90": float(np.percentile(v, 90)) if v.size else None,
            "p99": float(np.percentile(v, 99)) if v.size else None,
            "max": float(v.max()) if v.size else None,
            "frac_above_0_05": float(np.mean(v > 0.05)) if v.size else None,
        },
        "clips_compared": len(rows),
        "frames_compared": int(sum(r["n_common"] for r in rows)),
        "max_abs_d_lean": float(arr.max()) if len(arr) else None,
        "median_of_clip_max_abs_d_lean": float(np.median(arr)) if len(arr) else None,
        "clips_outside_1e9": int(sum(not r["all_exact_1e9"] for r in rows)),
        "by_old_stride": by_stride,
        "median_of_per_clip_medians": float(np.median([r["median_abs_d_lean"] for r in rows]))
        if rows else None,
        "rows": rows,
    }
    (out / "consistency.json").write_text(json.dumps(summary, indent=1))

    lines = ["# stride-1 versus stride-2 reader consistency (Track G3)", "",
             "| old stride | clips | frames compared | max abs d lean | clips outside 1e-9 |",
             "| --- | --- | --- | --- | --- |"]
    for s, v in by_stride.items():
        lines.append(f"| {s} | {v['clips']} | {v['frames_compared']} | "
                     f"{v['max_abs_d_lean']:.3e} | {v['clips_outside_1e9']} |")
    if repeat:
        lines += ["", "Two independent runs of the same clips (the reader's own stochasticity):",
                  f"components {repeat['components']}, median {repeat['median']:.3e}, "
                  f"p90 {repeat['p90']:.3e}, p99 {repeat['p99']:.3e}, max {repeat['max']:.3e}."]
    lines += ["", f"Total: {summary['clips_compared']} clips, "
                  f"{summary['frames_compared']} shared frames, max abs d lean "
                  f"{summary['max_abs_d_lean']:.3e}."]
    (out / "consistency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
