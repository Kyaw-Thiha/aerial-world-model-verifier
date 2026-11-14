"""Build the provisional "schema v2" clip set: clips carrying a dense required lean.

Track D has not published schema v2 (its branch is still at `98a5dfc`, whose `verdict_clip.npz`
carries only ``pos`` and ``yaw``). The defect that motivated the request is measured twice in
`reports/E02`: differentiating 15 Hz ``pos`` twice raises the median-residual floor from 1.54 to
7.40 deg and attenuates a lean-regression gain to about 0.2, because the required thrust axis so
obtained disagrees with the dense-truth one by a median 9.7 deg.

So each v1 clip is copied unchanged and **augmented** with ``lean_req`` (N, 2) computed from the
500 Hz Leica truth, plus the provenance fields a consumer needs to treat it as provisional:

    lean_req            (N, 2)  required lean in the yaw-aligned frame, (x, y) components of the
                                required thrust axis after rotating by -yaw
    lean_req_window_s   scalar  Savitzky-Golay window used on the dense truth
    lean_req_polyorder  scalar  polynomial order
    lean_req_source     string  provenance, repeated in meta.json

No depicted-side array is touched, so **no GPU work is involved** and the existing GeoCalib
estimates stay exactly as exported. ``experiments/e_verdict.py --path-source v2`` reads
``lean_req`` and needs nothing else, which is the T6 property the v1 schema could not offer.

    python experiments/e_clips_v2.py --clips data/clips --out data/clips_v2 --window-s 0.10
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.depicted.clips import path_side_with_gt
from src.depicted.uzh import all_sequences, load_truth

SCHEMA_VERSION = "2-provisional"


def lean_from_axis(axis_world: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Required lean: the horizontal components of the thrust axis in the yaw-aligned frame."""
    a = np.asarray(axis_world, dtype=np.float64)
    c, s = np.cos(-np.asarray(yaw, dtype=np.float64)), np.sin(-np.asarray(yaw, dtype=np.float64))
    return np.column_stack([c * a[:, 0] - s * a[:, 1], s * a[:, 0] + c * a[:, 1]])


def build(root: Path, out_root: Path, window_s: float, poly: int) -> dict:
    made, skipped = 0, []
    for seq in all_sequences():
        src = root / seq
        if not src.exists():
            continue
        truth = load_truth(Path("data/uzh"), seq)
        gt = np.load(Path("data/uzh") / seq / "truth.npz")["gt"]
        path = path_side_with_gt(truth, gt, k_d=0.0, window_s=window_s)
        dst = out_root / seq
        dst.mkdir(parents=True, exist_ok=True)
        for npz in sorted(src.glob("*.npz")):
            meta_path = npz.with_suffix(".meta.json")
            if not meta_path.exists():
                skipped.append(npz.name)
                continue
            meta = json.loads(meta_path.read_text())
            lo, hi = meta["index"]
            d = np.load(npz)
            lean = lean_from_axis(path["required_axis"][lo:hi], d["yaw"])
            if not np.allclose(d["t"], d["t"]):  # pragma: no cover - sanity
                raise ValueError(f"{npz}: non-finite t")
            np.savez_compressed(
                dst / npz.name,
                t=d["t"], pos=d["pos"], yaw=d["yaw"],
                depicted_lean=d["depicted_lean"], depicted_conf=d["depicted_conf"],
                **({"valid_frac": d["valid_frac"]} if "valid_frac" in d.files else {}),
                lean_req=lean,
                lean_req_window_s=np.float64(window_s),
                lean_req_polyorder=np.int64(poly),
                lean_req_source=np.array(f"dense Leica truth ({len(gt)} rows at 500 Hz), "
                                         f"Savitzky-Golay {window_s} s, polyorder {poly}"),
            )
            meta["schema_version"] = SCHEMA_VERSION
            meta["lean_req"] = {
                "field": "lean_req", "window_s": window_s, "polyorder": poly,
                "source": "dense Leica truth, not reconstructed from pos",
                "status": "provisional: computed by Track E, not Track D's v2 schema",
                "definition": "horizontal (x, y) components of the required thrust axis after "
                              "rotating by -yaw; the same quantity as depicted_lean",
            }
            (dst / meta_path.name).write_text(json.dumps(meta, indent=1))
            made += 1
    return {"clips": made, "skipped": skipped, "window_s": window_s, "polyorder": poly,
            "out": str(out_root), "schema_version": SCHEMA_VERSION}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--out", default="data/clips_v2")
    ap.add_argument("--window-s", type=float, default=0.10,
                    help="SG window on the 500 Hz truth (E01/E02 use 0.10 s)")
    ap.add_argument("--poly", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    rec = build(Path(args.clips), out, args.window_s, args.poly)
    (out / "build.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("clips", "window_s", "polyorder", "schema_version")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
