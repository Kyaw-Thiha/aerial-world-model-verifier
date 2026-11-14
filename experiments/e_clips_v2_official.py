"""Official schema v2 arrays for Track E's clips: ``lean_req``, ``vel_req``, ``tilt_req_deg``.

Follows `docs/verdict_clip_schema_v2.md` (Track D, `track-d-verdict-stat` at `851096c`) exactly:

1. take the commanded path and heading at their **native** rate (UZH: the 500 Hz Leica truth);
2. differentiate position twice with a Savitzky-Golay window fixed in **seconds** — 0.37 s,
   polyorder 3 for the headline, 0.10 and 0.185 s as pre-registered sensitivities;
3. ``u0 = (a + g e3) / ||a + g e3||``;
4. ``lean_req = Rz(-yaw) [u0_x, u0_y]``, ``vel_req = Rz(-yaw) [v_x, v_y] / g``,
   ``tilt_req_deg = degrees(arccos(u0_z))``;
5. sample at the **camera frame times** by linear interpolation, with frames outside the native
   SG interior (half a window at each end) set to NaN — never extrapolated.

``depicted_lean`` and ``depicted_conf`` are copied unchanged: nothing on the depicted side is
interpolated (E03's provisional builder did interpolate it, which biased the gain upward; this
supersedes that file).

The v2 fields are added per window into separate trees, so the three windows can be scored
against the same clips without ambiguity:

    data/clips_v3/         the v3 clips as exported (v1 fields only)
    data/clips_v2_w370ms/  + lean_req/vel_req/tilt_req_deg at a 0.37 s window  (headline)
    data/clips_v2_w100ms/  + ... at 0.10 s                                     (sensitivity)
    data/clips_v2_w185ms/  + ... at 0.185 s                                    (sensitivity)

    python experiments/e_clips_v2_official.py --clips data/clips_v3 --out-root data
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

G0 = 9.81
E3 = np.array([0.0, 0.0, 1.0])


def world_to_yaw_frame(vec: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """``Rz(-yaw)`` applied to the horizontal components (same convention as Track D's lean.py)."""
    vec = np.asarray(vec, dtype=np.float64)
    c, s = np.cos(-np.asarray(yaw, dtype=np.float64)), np.sin(-np.asarray(yaw, dtype=np.float64))
    return np.column_stack([c * vec[:, 0] - s * vec[:, 1], s * vec[:, 0] + c * vec[:, 1]])


def odd_window(n: int, dt_s: float, window_s: float, polyorder: int) -> int:
    w = int(round(window_s / dt_s))
    if w % 2 == 0:
        w += 1
    w = max(polyorder + 2, w)
    if w % 2 == 0:
        w += 1
    return min(w, n if n % 2 == 1 else n - 1)


def dense_required(gt: np.ndarray, t_frame: np.ndarray, window_s: float,
                   polyorder: int = 3, g: float = G0) -> dict:
    """The five steps of the v2 definition, on a TUM table ``gt`` (timestamp + pos + quat)."""
    t = np.asarray(gt[:, 0], dtype=np.float64)
    pos = np.asarray(gt[:, 1:4], dtype=np.float64)
    quat = np.asarray(gt[:, 4:8], dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    n = len(t)
    dt = float(np.median(np.diff(t)))
    w = odd_window(n, dt, window_s, polyorder)
    vel = savgol_filter(pos, w, polyorder, deriv=1, delta=dt, axis=0, mode="interp")
    acc = savgol_filter(pos, w, polyorder, deriv=2, delta=dt, axis=0, mode="interp")
    q3 = acc + g * E3
    norm = np.linalg.norm(q3, axis=1)
    u0 = np.full_like(q3, np.nan)
    good = np.isfinite(norm) & (norm > 1e-9)
    u0[good] = q3[good] / norm[good, None]
    yaw_native = Rotation.from_matrix(Rotation.from_quat(quat).as_matrix()).as_euler("ZYX")[:, 0]
    lean_n = world_to_yaw_frame(u0[:, :2], yaw_native)
    vel_n = world_to_yaw_frame(vel[:, :2] / g, yaw_native)
    tilt_n = np.degrees(np.arccos(np.clip(u0[:, 2], -1.0, 1.0)))
    half = w // 2
    for arr in (lean_n, vel_n, tilt_n):
        arr[:half] = np.nan
        arr[n - half:] = np.nan
    t_rel = t - t[0]

    def interp(y):
        if y.ndim == 1:
            return np.interp(t_frame, t_rel, y, left=np.nan, right=np.nan)
        return np.column_stack([np.interp(t_frame, t_rel, y[:, k], left=np.nan, right=np.nan)
                                for k in range(y.shape[1])])

    return {"lean_req": interp(lean_n), "vel_req": interp(vel_n), "tilt_req_deg": interp(tilt_n),
            "window_samples": int(w), "dt_s": dt, "masked_frames": int(np.isnan(lean_n).sum())}


def _ulp(lean: np.ndarray) -> np.ndarray:
    """Lean back to a unit up-vector, for angle comparisons."""
    from src.depicted.clips import up_body_from_lean

    u = up_body_from_lean(lean)
    return u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)


def build(clips_root: Path, out_root: Path, window_s: float, uzh_root: Path) -> dict:
    tag = f"clips_v2_w{int(round(window_s * 1000))}ms"
    out = out_root / tag
    if out.exists():
        shutil.rmtree(out)
    made, rejected, gaps = 0, [], []
    seqs = sorted({p.parent.name for p in clips_root.glob("*/*.npz")})
    for seq in seqs:
        tz = np.load(uzh_root / seq / "truth.npz")
        gt = tz["gt"]
        meta0 = json.loads((clips_root / seq / "00000.meta.json").read_text()) \
            if (clips_root / seq / "00000.meta.json").exists() else None
        t_unix0 = float(meta0["clock_offset_applied_s"]) if False else None
        (out / seq).mkdir(parents=True, exist_ok=True)
        for npz in sorted((clips_root / seq).glob("*.npz")):
            meta = json.loads(npz.with_suffix(".meta.json").read_text())
            d = np.load(npz)
            offset = float(meta.get("clock_offset_applied_s", 0.0))
            t_first_unix = float(tz["t_unix"][meta["index"][0]])
            # The dense grid is put on the clock of the CLIP's first frame, and the frame times
            # are then measured on that same origin. Sampling with d["t"] against a table whose
            # origin is the start of the whole recording silently reads the wrong slice of
            # trajectory for every clip after the first (caught by comparing lean_req against
            # the truth-attitude lean: 18.6 deg apart instead of the 2-6 deg flatness residual).
            # ``dense_required`` measures its own time from gt[0, 0], so the frame times must be
            # expressed on that same origin: (clip start on the truth clock) - (table start).
            t_frame = (t_first_unix + offset - float(gt[0, 0])) + np.asarray(d["t"], dtype=np.float64)
            rec = dense_required(gt, t_frame, window_s)
            if np.isnan(rec["lean_req"]).all():
                rejected.append(npz.name)
                continue
            # Guard against a mis-sampled path side: on a real clip the required lean must sit
            # within the drag-free flatness residual of the truth attitude. E01 measured that
            # residual at 4.0-11.5 deg on these sequences, so 15 deg is a generous ceiling.
            if meta["condition"] == "real":
                import numpy as _np

                from src.depicted.clips import lean_from_up_body as _lfu
                from src.depicted.metrics import angle_deg as _ang
                from src.depicted.uzh import load_truth as _lt
                _tr = _lt(uzh_root, seq)
                _lt_lean = _lfu(_tr.up_body)[meta["index"][0]:meta["index"][1]]
                ok = _np.isfinite(rec["lean_req"]).all(axis=1) & _np.isfinite(_lt_lean).all(axis=1)
                gap = float(_np.median(_ang(_ulp(rec["lean_req"][ok]), _ulp(_lt_lean[ok]))))
                gaps.append({"seq": seq, "clip": npz.stem, "gap_deg": gap})
                if gap > 15.0:
                    raise RuntimeError(f"{seq}/{npz.stem}: lean_req is {gap:.1f} deg from the "
                                       "truth-attitude lean; the path side is mis-sampled")
            np.savez_compressed(
                out / seq / npz.name, t=d["t"], pos=d["pos"], yaw=d["yaw"],
                depicted_lean=d["depicted_lean"], depicted_conf=d["depicted_conf"],
                **({"valid_frac": d["valid_frac"]} if "valid_frac" in d.files else {}),
                lean_req=rec["lean_req"], vel_req=rec["vel_req"],
                tilt_req_deg=rec["tilt_req_deg"],
                required_window_s=np.float64(window_s),
                required_window_samples=np.int64(rec["window_samples"]),
                required_dt_s=np.float64(rec["dt_s"]))
            meta["schema_version"] = 2
            meta["required_lean_source"] = ("dense Leica truth at 500 Hz, Savitzky-Golay "
                                            f"{window_s} s polyorder 3, sampled at frame times")
            meta["required_window_samples"] = rec["window_samples"]
            meta["required_dt_s"] = rec["dt_s"]
            meta["required_window_s"] = window_s
            meta["depicted_side_interpolated"] = False
            (out / seq / npz.with_suffix(".meta.json").name).write_text(json.dumps(meta, indent=1))
            made += 1
    return {"tree": str(out), "window_s": window_s, "clips": made, "rejected": rejected,
            "lean_req_vs_truth_attitude_deg": {
                "n": len(gaps),
                "median": float(np.median([g["gap_deg"] for g in gaps])) if gaps else None,
                "max": float(np.max([g["gap_deg"] for g in gaps])) if gaps else None}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips_v3")
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--uzh-root", default="data/uzh")
    ap.add_argument("--windows", default="0.37,0.10,0.185")
    args = ap.parse_args()
    recs = []
    for w in [float(x) for x in args.windows.split(",") if x]:
        rec = build(Path(args.clips), Path(args.out_root), w, Path(args.uzh_root))
        recs.append(rec)
        print(json.dumps({k: rec[k] for k in ("tree", "window_s", "clips")}), flush=True)
    Path(args.out_root, "clips_v2_build.json").write_text(json.dumps(recs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
