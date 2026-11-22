"""Schema v3: a model-required lean, written to new files (never editing v2).

Schema v2's ``lean_req`` is the **drag-free** required lean; a drag coefficient is
applied by the consumer as ``lean_req + k_d * vel_req``, which is a first-order
approximation to the nonlinear flatness axis (K01 measures a median 2.2 deg and a p95
20.4 deg gap on testable frames).  Schema v3 stores the required lean of a named, fitted
airframe model *exactly* at the clip's frame times:

    lean_req_v3(t) = horizontal part of normalise(axis_model(t)) in the path-yaw frame

The semantics are explicit in the metadata: ``lean_req_v3`` is the model-required lean
and must NOT be rescaled by ``(k_d - 0.295)``.  Files are new; the v2 tree is read only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.verdict.lean import world_to_yaw_frame

from .kinematics import TII_FLIGHTS, load_tii_kinematics
from .models import axis_for


def sample_axis(k, axis: np.ndarray, t_frame: np.ndarray,
                yaw_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a native axis at frame times, then express it in the frame's yaw.

    Returns ``(axis, lean, tilt_deg)``.  The axis is kept because a lean vector cannot
    represent an inverted thrust axis (``z < 0``, which occurs on 4,507 TII native frames
    during aggressive vertical manoeuvres); consumers that need the true angle must use
    the axis, and consumers that need a lean must mask ``tilt_deg`` near 90.  Frames
    outside the native Savitzky-Golay interior, or whose bracketing samples are invalid,
    are NaN rather than extrapolated.
    """
    t = k.t
    tt = np.clip(np.asarray(t_frame, float), t[0], t[-1])
    good = k.valid & np.isfinite(axis).all(axis=1)
    out = np.column_stack([np.interp(tt, t, axis[:, j]) for j in range(3)])
    idx = np.clip(np.searchsorted(t, tt), 1, len(t) - 1)
    bad = ~good[idx - 1] | ~good[idx]
    lo, hi = int(np.argmax(good)), len(good) - 1 - int(np.argmax(good[::-1]))
    bad |= (tt < t[lo]) | (tt > t[hi])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.where(np.isfinite(n) & (n > 1e-9), n, np.nan)
    out[bad] = np.nan
    lean = world_to_yaw_frame(out[:, :2], yaw_frame)
    lean[bad] = np.nan
    tilt = np.degrees(np.arccos(np.clip(out[:, 2], -1.0, 1.0)))
    tilt[bad] = np.nan
    return out, lean, tilt


def required_lean_at(k, model: str, params: dict, t_frame: np.ndarray,
                     yaw_frame: np.ndarray) -> dict:
    """Model-required lean and tilt at ``t_frame``, expressed in ``yaw_frame``."""
    axis = axis_for(model, params, k)
    ax, lean, tilt = sample_axis(k, axis, t_frame, yaw_frame)
    return {"axis": ax, "lean": lean, "tilt_deg": tilt}


def write_v3_clip(src_dir: str | Path, dst_dir: str | Path, lean: np.ndarray,
                  tilt: np.ndarray, model: str, params: dict, fold: str,
                  source_note: str, axis: np.ndarray | None = None) -> tuple[Path, str]:
    """Copy a v2 clip's arrays and replace the required side with the v3 model side."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(src_dir / "verdict_clip.npz", allow_pickle=True)
    meta = json.loads((src_dir / "meta.json").read_text())
    if lean.shape != z["depicted_lean"].shape:
        raise ValueError(f"{src_dir}: lean shape {lean.shape} != depicted shape "
                         f"{z['depicted_lean'].shape}")
    payload = {k: z[k] for k in z.files if k not in ("lean_req", "vel_req", "tilt_req_deg")}
    payload["lean_req_v3"] = np.asarray(lean, float)
    payload["tilt_req_v3_deg"] = np.asarray(tilt, float)
    if axis is not None:
        payload["axis_req_v3"] = np.asarray(axis, float)
    # provenance: keep the v2 arrays under names that cannot be mistaken for v3
    if "lean_req" in z.files:
        payload["lean_req_v2_dragfree"] = z["lean_req"]
        payload["vel_req_v2"] = z["vel_req"]
        payload["tilt_req_v2_deg"] = z["tilt_req_deg"]
    np.savez_compressed(dst_dir / "verdict_clip.npz", **payload)
    meta.update({
        "schema_version": 3,
        "required_lean_source": source_note,
        "model_name": model,
        "model_parameters": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in params.items()
                             if k not in ("feat_mean", "feat_sd", "alpha_scores")},
        "fit_fold": fold,
        "lean_req_v3_semantics": ("model-required lean at the model's own parameters; "
                                  "do NOT apply (k_d - 0.295) rescaling"),
        "lean_req_v2_semantics": "drag-free required lean (schema v2)",
    })
    (dst_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    h = hashlib.sha256((dst_dir / "verdict_clip.npz").read_bytes()).hexdigest()
    return dst_dir / "verdict_clip.npz", h


def export_tree(fit: dict, model: str, out_root: str | Path,
                fold: str = "all-8-flights (delivery copy)") -> list[tuple[str, str, str, str]]:
    """Write the whole TII clip tree in schema v3 from a ``fit.json`` full fit.

    Returns ``(flight, clip_id, condition, sha256)`` rows.  The v2 tree is opened
    read-only and is never written.
    """
    from src.airframe.kinematics import ROOT

    root = Path(ROOT)
    params = dict(fit["full_fit"][model]["params"])
    for key in ("W", "feat_mean", "feat_sd"):
        if key in params:
            params[key] = np.asarray(params[key], float)
    out_root = Path(out_root)
    rows = []
    for flight in TII_FLIGHTS:
        k = load_tii_kinematics(flight)
        for d in sorted((root / f"data/processed/track_d/clips/{flight}").iterdir()):
            if not d.is_dir():
                continue
            z = np.load(d / "verdict_clip.npz")
            req = required_lean_at(k, model, params, np.asarray(z["t"], float),
                                   np.asarray(z["yaw"], float))
            cid, cond = d.name.split("__")
            dst = out_root / flight / d.name
            _, h = write_v3_clip(
                d, dst, req["lean"], req["tilt_deg"], model, params, fold,
                f"schema v3: {model} required lean at frame times, fitted on {fold}",
                axis=req["axis"])
            rows.append((flight, cid, cond, h))
    return rows
