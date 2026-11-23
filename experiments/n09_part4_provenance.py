"""Track N2, N09: Part C provenance -- is the Part 4 comparison information-matched?

Pre-registration ``configs/n09_part4_provenance_prereg.yaml``.

CPU reanalysis of stored H04 poses, reader JSONL and path table. No GPU job.

    PYTHONPATH=. pixi run python experiments/n09_part4_provenance.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.n04_generated_rotation import (POSES, ang, load_reader, true_attitudes, up_of)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n2/n09"
FPS = 16.0
FRAMES = {0: 0, 1: 16, 2: 32, 5: 80}


def rel_err(up, Rt, k):
    """Relative-horizon error at k frames: the arm's up at i+h against its up at i rotated by the
    TRUE relative rotation from i to i+h. Zero for a stream that follows the truth exactly."""
    if k == 0:
        return 0.0
    if k >= len(up):
        return float("nan")
    Rrel = np.einsum("nji,njk->nik", Rt[k:], Rt[:-k])   # camera_i -> camera_j
    pred = np.einsum("nij,nj->ni", Rrel, up[:-k])
    return float(np.median(ang(pred, up[k:])))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = [r for r in csv.DictReader(open(ROOT / "results/n04/clips.csv"))
             if r["arm"] == "A_flatness"]
    reader = load_reader()
    per = []
    for n, r in enumerate(clips):
        cid = r["clip"]
        npz = POSES / "fov90" / f"{cid}.npz"
        if not npz.exists():
            continue
        R = np.asarray(np.load(npz)["c2w"], float)[:, :3, :3]
        Rt = true_attitudes(r["path"], r["arm"])
        m = min(len(R), len(Rt))
        R, Rt = R[:m], Rt[:m]
        ru = reader.get(cid, {})
        if 0 not in ru or (m - 1) not in ru:
            continue
        ut = up_of(Rt)
        Rrel0 = np.einsum("nji,jk->nik", R, R[0])         # R(k)^T R(0)
        up_oracle = np.einsum("nij,j->ni", Rrel0, ut[0])
        up_reader = np.einsum("nij,j->ni", Rrel0, ru[0])
        up_rd = np.stack([ru.get(k, np.full(3, np.nan)) for k in range(m)])
        rec = {"clip": cid, "path": r["path"], "draw": r["draw"], "n": m,
               "frame0_reader_err": float(ang(ru[0], ut[0]))}
        for h, k in FRAMES.items():
            rec[f"abs_oracle_{h}s"] = float(ang(up_oracle[k], ut[k]))
            rec[f"abs_readerstart_{h}s"] = float(ang(up_reader[k], ut[k]))
            rec[f"abs_reader_{h}s"] = float(ang(up_rd[k], ut[k]))
            rec[f"rel_oracle_{h}s"] = rel_err(up_oracle, Rt, k)
            rec[f"rel_readerstart_{h}s"] = rel_err(up_reader, Rt, k)
            rec[f"rel_reader_{h}s"] = rel_err(up_rd, Rt, k)
        rec["abs_median_oracle"] = float(np.nanmedian(ang(up_oracle, ut)))
        rec["abs_median_readerstart"] = float(np.nanmedian(ang(up_reader, ut)))
        rec["abs_median_reader"] = float(np.nanmedian(ang(up_rd, ut)))
        per.append(rec)
        if n % 20 == 0:
            print("clip", n, flush=True)
    with open(OUT / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    out = {"n_clips": len(per), "fps": FPS,
           "note": ("absolute = angle(arm up at frame k, commanded up at k) with the propagated "
                    "arms started at frame 0; relative = N04's horizon form (the arm's up at i+h "
                    "against its up at i rotated by the TRUE relative rotation), computed "
                    "identically for every arm. N04's published 3.54/4.40/6.69 are the oracle "
                    "arm's relative medians over all sliding windows.")}
    for fam in ("abs", "rel"):
        out[fam] = {}
        for arm in ("oracle", "readerstart", "reader"):
            out[fam][arm] = {f"{h}s": float(np.nanmedian([r[f"{fam}_{arm}_{h}s"] for r in per]))
                             for h in FRAMES}
    out["abs_median_over_frames"] = {
        a: float(np.nanmedian([r[f"abs_median_{a}"] for r in per]))
        for a in ("oracle", "readerstart", "reader")}
    r1 = out["abs"]["readerstart"]["1s"]
    ref = out["abs_median_over_frames"]["reader"]
    out["decision"] = {
        "reader_start_abs_1s": r1,
        "reader_per_frame_median": ref,
        "information_matched": bool(r1 < ref),
        "reading": ("the reader-start arm is below the reader's all-frame median at 1 s"
                    if r1 < ref else
                    "the reader-start arm is not below the reader's all-frame median at 1 s"),
        "at_matched_horizon_1s": {
            "readerstart": r1, "reader": out["abs"]["reader"]["1s"],
            "readerstart_better": bool(r1 < out["abs"]["reader"]["1s"])},
        "relative_1s": {a: out["rel"][a]["1s"] for a in ("oracle", "readerstart", "reader")},
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
