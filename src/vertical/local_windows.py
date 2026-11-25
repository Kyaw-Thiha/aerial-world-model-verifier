"""Window rows with a locally fitted plane (Track V, the 5 s secondary arm).

The 3 s primary arm uses the plane fits stored by ``remote/v_recon.py``. The 5 s secondary
arm (``configs/v10_vertical_prereg_amend1.yaml``) needs a plane over a window the GPU job did
not fit, so it is fitted here from the stored per-frame point clouds, with the same
``src.vertical.plane`` rule. The points are the same decimated clouds the remote fit saw
(2,000 per frame), so the only difference is the sampling of points inside the plane rule.
"""
from __future__ import annotations

import numpy as np

from src.vertical import metrics as M
from src.vertical.plane import COVERAGE_INLIER_FRAC, fit_ground_plane

PLANE_PAD_S = 0.5


def _chunk_for(data: dict, t1: float, t2: float) -> int:
    centres = [0.5 * (c["t"][0] + c["t"][-1]) for c in data["chunks"].values()]
    keys = list(data["chunks"])
    wc = 0.5 * (t1 + t2)
    return keys[int(min(range(len(keys)), key=lambda k: abs(centres[k] - wc)))]


def rows_local(data: dict, truth_t: np.ndarray, truth_pos: np.ndarray, *, mode: str,
               window_s: float, stride_s: float = M.WINDOW_STRIDE_S,
               min_frames: int = M.WINDOW_MIN_FRAMES) -> list[dict]:
    t_all = data["chunks"][list(data["chunks"])[0]]["t"]
    wins = M.windows(t_all, length=window_s, stride=stride_s, min_frames=min_frames)
    rows = []
    for w in wins:
        k = _chunk_for(data, w["t1"], w["t2"])
        ch = data["chunks"][k]
        t = ch["t"]
        pad = (t >= w["t1"] - PLANE_PAD_S - 1e-9) & (t <= w["t2"] + PLANE_PAD_S + 1e-9)
        local = np.flatnonzero(pad)
        cams = ch["c2w"][local, :3, 3]
        sel = np.isin(ch["pts_frame"], local)
        pts = ch["pts"][sel]
        fit = fit_ground_plane(pts, cams, travel=cams[-1] - cams[0])
        plane = fit.get("plane")
        if plane is None or not plane.admissible or plane.inlier_frac < COVERAGE_INLIER_FRAC:
            rows.append({"t1": w["t1"], "t2": w["t2"], "ok": False,
                         "reason": "no_plane" if plane is None else "plane_inadmissible"})
            continue
        inside = (t >= w["t1"] - 1e-9) & (t <= w["t2"] + 1e-9)
        before = np.flatnonzero(t <= w["t1"] + 1e-9)
        after = np.flatnonzero(t >= w["t2"] - 1e-9)
        keep = set(np.flatnonzero(inside).tolist())
        if len(before):
            keep.add(int(before[-1]))
        if len(after):
            keep.add(int(after[0]))
        idx = np.array(sorted(keep), dtype=int)
        if len(idx) < 2:
            rows.append({"t1": w["t1"], "t2": w["t2"], "ok": False,
                         "reason": "too_few_frames"})
            continue
        scale_s = None
        if mode == "real":
            tt = np.asarray(truth_t, float)
            pp = np.asarray(truth_pos, float)
            ins = (t >= tt[0]) & (t <= tt[-1])
            if ins.sum() >= 4:
                pos_i = np.column_stack([np.interp(t[ins], tt, pp[:, c]) for c in range(3)])
                scale_s = M.sequence_scale(ch["c2w"][ins, :3, 3], pos_i)
        rows.append(M.evaluate_window(t[idx], ch["c2w"][idx, :3, 3], plane.n, plane.d,
                                     w["t1"], w["t2"], truth_t, truth_pos, mode=mode,
                                     scale_s=scale_s))
    return rows
