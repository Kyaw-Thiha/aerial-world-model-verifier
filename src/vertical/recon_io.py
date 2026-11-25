"""Load Track V reconstruction output and turn it into window rows (CPU, no GPU).

Shared by ``experiments/v00_feasibility.py`` and ``experiments/v10_score.py`` so both read the
plane fits and camera centres exactly the same way.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.vertical import metrics as M


def load_unit(recon_dir: Path, unit: str) -> dict:
    js = json.loads((recon_dir / f"{unit}.json").read_text())
    chunks = {}
    for c in js["chunks"]:
        d = np.load(recon_dir / f"{unit}__c{c['chunk']}.npz")
        chunks[c["chunk"]] = {"t": np.asarray(d["t"], float),
                              "c2w": np.asarray(d["c2w"], float)}
    return {"meta": js, "chunks": chunks}


def unit_rows(data: dict, truth_t: np.ndarray, truth_pos: np.ndarray, *, mode: str,
              min_inlier_frac: float = 0.10) -> tuple[list[dict], list[dict]]:
    """Per-window rows for one unit (planes from the window's own fit)."""
    tt = np.asarray(truth_t, float)
    pp = np.asarray(truth_pos, float)
    rows, diag = [], []
    for w in data["meta"]["windows"]:
        ch = data["chunks"][w["chunk"]]
        plane = w.get("plane")
        if not plane or not plane.get("admissible"):
            rows.append({"t1": w["t1"], "t2": w["t2"], "ok": False,
                         "reason": "no_plane" if not plane else "plane_inadmissible"})
            continue
        if plane["inlier_frac"] < min_inlier_frac:
            rows.append({"t1": w["t1"], "t2": w["t2"], "ok": False,
                         "reason": "low_inlier_frac"})
            continue
        t_ch = ch["t"]
        sel = (t_ch >= w["t1"] - 1e-9) & (t_ch <= w["t2"] + 1e-9)
        if sel.sum() < 2:
            rows.append({"t1": w["t1"], "t2": w["t2"], "ok": False,
                         "reason": "too_few_frames"})
            continue
        cams = ch["c2w"][sel, :3, 3]
        n = np.asarray(plane["n"], float)
        d = float(plane["d"])
        scale_s = None
        if mode == "real":
            inside = (t_ch >= tt[0]) & (t_ch <= tt[-1])
            if inside.sum() >= 4:
                pos_i = np.column_stack([np.interp(t_ch[inside], tt, pp[:, k])
                                         for k in range(3)])
                scale_s = M.sequence_scale(ch["c2w"][inside, :3, 3], pos_i)
        row = M.evaluate_window(t_ch[sel], cams, n, d, w["t1"], w["t2"], tt, pp,
                                mode=mode, scale_s=scale_s)
        if scale_s is not None:
            row["scale_s"] = scale_s
        rows.append(row)
        diag.append({"t1": w["t1"], "t2": w["t2"], "chunk": w["chunk"],
                     "inlier_frac": plane["inlier_frac"], "facing_deg": plane["facing_deg"],
                     "height_med": float(np.median(cams @ n - d))})
    return rows, diag


def whole_clip_row(data: dict, truth_pos: np.ndarray) -> dict:
    """The generated-clip statistic: one row over the whole clip, scale-free only.

    ``rho_dep = dz_dep / L_dep`` and ``rho_cmd = dz_cmd / L_cmd``; no height ratio is formed
    because the start height of a generated frame is not known (V1 pre-registration, Part 3).
    """
    pos = np.asarray(truth_pos, float)
    dz_cmd = float(pos[-1, 2] - pos[0, 2])
    L_cmd = float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)))
    rho_cmd = dz_cmd / max(L_cmd, 1e-9)
    w = data["meta"]["windows"]
    if len(w) != 1:
        raise ValueError(f"generated clip should have one window, has {len(w)}")
    w = w[0]
    plane = w.get("plane")
    if not plane or not plane.get("admissible") or plane["inlier_frac"] < 0.10:
        return {"ok": False, "reason": "no_plane" if not plane else "plane_inadmissible",
                "rho_cmd": rho_cmd, "dz_cmd": dz_cmd, "L_cmd": L_cmd}
    ch = data["chunks"][w["chunk"]]
    t_ch = ch["t"]
    c = ch["c2w"][:, :3, 3]
    n = np.asarray(plane["n"], float)
    d = float(plane["d"])
    h = c @ n - d
    dperp = c - np.outer(c @ n, n)
    L_dep = float(np.sum(np.linalg.norm(np.diff(dperp, axis=0), axis=1)))
    dz_dep = float(n @ (c[-1] - c[0]))
    rho_dep = dz_dep / max(L_dep, 1e-9)
    q = (rho_dep - rho_cmd) / (abs(rho_cmd) + 0.05)
    return {"ok": True, "rho_cmd": rho_cmd, "rho_dep": rho_dep,
            "dz_cmd": dz_cmd, "L_cmd": L_cmd, "dz_dep": dz_dep, "L_dep": L_dep, "q": q,
            "abs_q": abs(q), "h_first": float(h[0]), "h_last": float(h[-1]),
            "inlier_frac": float(plane["inlier_frac"]), "n_frames": int(len(t_ch))}
