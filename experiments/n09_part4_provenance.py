"""Track N2, N09: Part C provenance -- is the Part 4 comparison information-matched?

Pre-registration ``configs/n09_part4_provenance_prereg.yaml``.

CPU reanalysis of stored H04 poses, reader JSONL and path table. No GPU job.

    PYTHONPATH=. pixi run python experiments/n09_part4_provenance.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.n04_generated_rotation import H3, POSES, load_reader, true_attitudes, up_of, ang

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n2/n09"
E3 = np.array([0.0, 0.0, 1.0])
FPS = 16.0
HORIZONS = (0, 1, 2, 5)


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
        d = np.load(npz)
        R = np.asarray(d["c2w"], float)[:, :3, :3]
        Rt = true_attitudes(r["path"], r["arm"])
        m = min(len(R), len(Rt))
        R, Rt = R[:m], Rt[:m]
        ru = reader.get(cid, {})
        if 0 not in ru:
            continue
        ut = up_of(Rt)
        Rrel = np.einsum("nji,jk->nik", R, R[0])          # R(k)^T R(0)
        u0_true = ut[0]
        u0_reader = ru[0]
        up_oracle = np.einsum("nij,j->ni", Rrel, u0_true)
        up_reader = np.einsum("nij,j->ni", Rrel, u0_reader)
        rec = {"clip": cid, "path": r["path"], "draw": r["draw"], "n": m,
               "frame0_reader_err": float(ang(u0_reader, u0_true))}
        for h in HORIZONS:
            k = int(round(h * FPS))
            if k >= m:
                rec[f"oracle_{h}s"] = np.nan
                rec[f"readerstart_{h}s"] = np.nan
                rec[f"reader_{h}s"] = np.nan
                continue
            rec[f"oracle_{h}s"] = float(ang(up_oracle[k], ut[k]))
            rec[f"readerstart_{h}s"] = float(ang(up_reader[k], ut[k]))
            rec[f"reader_{h}s"] = float(ang(ru[k], ut[k])) if k in ru else np.nan
        # per-frame reader median (N04's 9.07 reference)
        errs = [float(ang(ru[k], ut[k])) for k in range(m) if k in ru]
        rec["reader_median_all"] = float(np.median(errs)) if errs else np.nan
        per.append(rec)
        if n % 20 == 0:
            print("clip", n, flush=True)
    with open(OUT / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    out = {"n_clips": len(per), "fps": FPS}
    for arm in ("oracle", "readerstart", "reader"):
        out[arm] = {f"{h}s": float(np.nanmedian([r[f"{arm}_{h}s"] for r in per]))
                    for h in HORIZONS}
    out["reader_median_all"] = float(np.nanmedian([r["reader_median_all"] for r in per]))
    out["frame0_reader_err_median"] = float(np.nanmedian([r["frame0_reader_err"] for r in per]))
    matched = out["readerstart"]["1s"] < out["reader_median_all"]
    out["decision"] = {
        "reader_start_1s": out["readerstart"]["1s"],
        "reader_per_frame": out["reader_median_all"],
        "information_matched": bool(matched),
        "text": ("information-matched: reader start at 1 s beats the reader"
                 if matched else
                 "not information-matched: the propagated advantage needs a start better than "
                 "the reader"),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
