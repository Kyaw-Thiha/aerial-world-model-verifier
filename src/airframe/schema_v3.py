"""Schema v3: a model-required lean, written to new files (never editing v2).

Schema v2's ``lean_req`` is the **drag-free** required lean; a drag coefficient is
applied by the consumer as ``lean_req + k_d * vel_req``, which is a first-order
approximation to the nonlinear flatness axis.  Schema v3 stores the required lean of a
named, fitted airframe model *exactly* at the clip's frame times:

    lean_req_v3(t) = horizontal part of normalise(axis_model(t)) in the path-yaw frame

The semantics are explicit in the metadata: ``lean_req_v3`` is the model-required lean
and must NOT be rescaled by ``(k_d - 0.295)``.  Files are new; the v2 tree is read only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.verdict.lean import world_to_yaw_frame

from .kinematics import Kinematics
from .models import axis_for


def required_lean_at(k: Kinematics, model: str, params: dict,
                     t_frame: np.ndarray) -> dict:
    """Model-required lean, tilt and thrust axis sampled at ``t_frame`` (native fit)."""
    axis = axis_for(model, params, k)
    lean = world_to_yaw_frame(axis[:, :2], k.yaw)
    tilt = np.degrees(np.arccos(np.clip(axis[:, 2], -1.0, 1.0)))
    good = k.valid & np.isfinite(lean).all(axis=1) & np.isfinite(tilt)
    t = k.t
    tt = np.clip(np.asarray(t_frame, float), t[0], t[-1])

    def sample(arr, dim):
        out = np.column_stack([np.interp(tt, t, arr[:, j]) for j in range(dim)])
        idx = np.clip(np.searchsorted(t, tt), 1, len(t) - 1)
        bad = ~good[idx - 1] | ~good[idx]
        # also require the requested time itself to be inside the valid interior
        bad |= (tt < t[np.argmax(good)]) | (tt > t[len(good) - 1 - np.argmax(good[::-1])])
        out[bad] = np.nan
        return out

    lean_s = sample(lean, 2)
    tilt_s = sample(tilt[:, None], 1)[:, 0]
    lean_s[~np.isfinite(tilt_s)] = np.nan
    return {"lean": lean_s, "tilt_deg": tilt_s}


def write_v3_clip(src_dir: str | Path, dst_dir: str | Path, lean: np.ndarray,
                  tilt: np.ndarray, model: str, params: dict, fold: str,
                  source_note: str) -> Path:
    """Copy a v2 clip's arrays and replace the required side with the v3 model side."""
    import hashlib
    import json

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
    # keep v2 arrays under renamed keys for provenance, never under their v2 names
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
                             if k not in ("feat_mean", "feat_sd")},
        "fit_fold": fold,
        "lean_req_v3_semantics": ("model-required lean at the model's own parameters; "
                                  "do NOT apply (k_d - 0.295) rescaling"),
        "lean_req_v2_semantics": "drag-free required lean (schema v2)",
    })
    (dst_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    h = hashlib.sha256((dst_dir / "verdict_clip.npz").read_bytes()).hexdigest()
    return dst_dir / "verdict_clip.npz", h
