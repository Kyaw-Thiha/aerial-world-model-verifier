"""Unit tests for the recon-output loader (src/vertical/recon_io.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.vertical import recon_io as RIO


def _write_unit(tmp_path: Path, alt="rising") -> Path:
    """A tiny synthetic reconstruction: cameras fly +x at 10 m over a plane at z = 0."""
    n = 30
    t = np.arange(n) * 0.1333
    c = np.column_stack([5.0 * t, np.zeros(n),
                         (10.0 + 0.5 * t) if alt == "rising" else np.full(n, 10.0)])
    poses = np.repeat(np.eye(4)[None], n, 0)
    poses[:, :3, 3] = c
    # points: a plane 10 units below the cameras (reconstruction units, plane z = 0)
    rng = np.random.default_rng(0)
    pts, labels = [], []
    for s in range(n):
        p = np.column_stack([rng.uniform(-20, 60, 500), rng.uniform(-20, 20, 500),
                             0.05 * rng.normal(size=500)])
        pts.append(p)
        labels.append(np.full(len(p), s, dtype=np.int32))
    np.savez_compressed(tmp_path / "unit__c0.npz", t=t, c2w=poses, pts=np.concatenate(pts),
                        pts_frame=np.concatenate(labels), i0=0, i1=n)
    plane = {"n": [0.0, 0.0, 1.0], "d": 0.0, "inlier_frac": 0.9, "facing_deg": 0.0,
             "admissible": True, "n_inliers": 1000, "height_scene_units": 10.0}
    windows = []
    for t1 in np.arange(0.0, t[-1] - 3.0 + 1e-9, 0.5):
        windows.append({"t1": float(t1), "t2": float(t1 + 3.0), "chunk": 0, "i0": 0, "i1": n,
                        "plane": plane, "fit": {"ok": True}})
    (tmp_path / "unit.json").write_text(json.dumps(
        {"unit": "unit", "chunks": [{"chunk": 0, "i0": 0, "i1": n, "n_points": len(pts) * 500,
                                     "t_range": [float(t[0]), float(t[-1])]}],
         "windows": windows}))
    return tmp_path


def test_rows_are_computed_from_the_stored_planes(tmp_path):
    _write_unit(tmp_path)
    data = RIO.load_unit(tmp_path, "unit")
    tt = np.arange(0, 4.1, 0.02)
    pos = np.column_stack([5.0 * tt, np.zeros_like(tt), 10.0 + 0.5 * tt])
    rows, _ = RIO.unit_rows(data, tt, pos, mode="sim")
    ok = [r for r in rows if r.get("ok")]
    assert ok
    # the depicted height ratio is exactly the commanded one by construction
    assert max(abs(r["e"]) for r in ok) < 1e-6


def test_constant_altitude_reconstruction_is_flagged(tmp_path):
    _write_unit(tmp_path, alt="level")
    data = RIO.load_unit(tmp_path, "unit")
    tt = np.arange(0, 4.1, 0.02)
    pos = np.column_stack([5.0 * tt, np.zeros_like(tt), 10.0 + 0.5 * tt])
    rows, _ = RIO.unit_rows(data, tt, pos, mode="sim")
    ok = [r for r in rows if r.get("ok")]
    assert ok and all(r["e"] < -0.01 for r in ok)


def test_whole_clip_excursion_statistic(tmp_path):
    _write_unit(tmp_path)
    data = RIO.load_unit(tmp_path, "unit")
    # a generated clip is reconstructed as ONE window over the whole clip
    w = data["meta"]["windows"][0]
    ch = data["chunks"][0]
    data["meta"]["windows"] = [{"t1": float(ch["t"][0]), "t2": float(ch["t"][-1]),
                                "chunk": 0, "i0": 0, "i1": len(ch["t"]),
                                "plane": w["plane"], "fit": {"ok": True}}]
    tt = np.arange(0, 4.1, 0.02)
    # an excursion: up and back down
    z = 10.0 + 0.2 * np.sin(np.pi * tt / 4.0)
    pos = np.column_stack([5.0 * tt, np.zeros_like(tt), z])
    row = RIO.whole_clip_row(data, pos, kind="excursion")
    assert row["ok"]
    assert 0.0 < row["rho_cmd"] < 0.1
    assert np.isfinite(row["q"])
