"""Loader and validator for the shared ``verdict_clip.npz`` schema (handoff 08).

One file per clip and condition, with a sibling ``meta.json``.  The schema is identical
in the Track E handoff, so this loader is the single place that decides what a clip is.

Fields
------
``t``            (N,) seconds, strictly increasing
``pos``          (N, 3) metres, world z-up: the commanded path
``yaw``          (N,) radians, body heading about world z
``depicted_lean`` (N, 2) radians-equivalent lean vector in the yaw-aligned frame,
                  NaN where the estimator failed
``depicted_conf`` (N,) per-frame estimator confidence or NaN; its meaning lives in the
                  metadata (for GeoCalib it is ``latitude_confidence_mean``, higher is
                  better; the frozen gate uses the negative median)
``meta.json``     dataset, sequence/flight, clip id, condition, fps, mount source and
                  rotation/uptilt, estimator name and variant, truth-frame notes.

No mocap-derived quantity other than ``pos`` and ``yaw`` is allowed in a clip: the
required lean is recomputed from those two arrays by :mod:`src.verdict.lean`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONDITIONS = ("real", "tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll",
              "yaw_control", "null_edit", "delay_ms:100", "delay_ms:200", "delay_ms:300")


@dataclass
class Clip:
    t: np.ndarray
    pos: np.ndarray
    yaw: np.ndarray
    depicted_lean: np.ndarray
    depicted_conf: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.t)

    @property
    def condition(self) -> str:
        return str(self.meta.get("condition", ""))

    @property
    def clip_id(self) -> str:
        return str(self.meta.get("clip_id", ""))

    @property
    def flight(self) -> str:
        return str(self.meta.get("flight", self.meta.get("sequence", "")))

    def subset(self, lo: int, hi: int) -> "Clip":
        sl = slice(int(lo), int(hi))
        meta = dict(self.meta)
        meta["parent_index"] = [int(lo), int(hi)]
        return Clip(t=self.t[sl].copy(), pos=self.pos[sl].copy(), yaw=self.yaw[sl].copy(),
                    depicted_lean=self.depicted_lean[sl].copy(),
                    depicted_conf=self.depicted_conf[sl].copy(), meta=meta)


def _as_array(data, key: str, shape: tuple[int, ...], dtype=float) -> np.ndarray:
    if key not in data:
        raise ValueError(f"missing field {key!r}")
    arr = np.asarray(data[key], dtype=dtype)
    if arr.shape != shape:
        raise ValueError(f"{key} shape {arr.shape} != expected {shape}")
    return arr


def validate_clip(clip: Clip) -> None:
    n = len(clip.t)
    if clip.t.ndim != 1 or n < 2:
        raise ValueError(f"t must be a 1-D array with at least 2 samples, got {clip.t.shape}")
    if not np.isfinite(clip.t).all():
        raise ValueError("t contains non-finite values")
    if not np.all(np.diff(clip.t) > 0):
        raise ValueError("t must be strictly increasing")
    if clip.pos.shape != (n, 3) or not np.isfinite(clip.pos).all():
        raise ValueError(f"pos must be finite (N, 3), got {clip.pos.shape}")
    if clip.yaw.shape != (n,) or not np.isfinite(clip.yaw).all():
        raise ValueError(f"yaw must be finite (N,), got {clip.yaw.shape}")
    if clip.depicted_lean.shape != (n, 2):
        raise ValueError(f"depicted_lean must be (N, 2), got {clip.depicted_lean.shape}")
    if not np.isfinite(clip.depicted_lean).any():
        raise ValueError("depicted_lean is entirely NaN: nothing to test")
    if clip.depicted_conf.shape != (n,):
        raise ValueError(f"depicted_conf must be (N,), got {clip.depicted_conf.shape}")


def load_clip(path: str | Path) -> Clip:
    """Load a clip from ``.../verdict_clip.npz`` or its directory."""
    path = Path(path)
    if path.is_dir():
        meta_path = path / "meta.json"
        npz_path = path / "verdict_clip.npz"
    else:
        npz_path = path
        meta_path = path.with_name("meta.json")
        if not meta_path.exists():
            meta_path = path.parent / "meta.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        n = int(np.asarray(data["t"]).size)
        clip = Clip(
            t=_as_array(data, "t", (n,)),
            pos=_as_array(data, "pos", (n, 3)),
            yaw=_as_array(data, "yaw", (n,)),
            depicted_lean=_as_array(data, "depicted_lean", (n, 2)),
            depicted_conf=_as_array(data, "depicted_conf", (n,)),
            meta=json.loads(meta_path.read_text()) if meta_path.exists() else {},
        )
    validate_clip(clip)
    return clip


def save_clip(path: str | Path, clip: Clip) -> Path:
    """Write a clip as ``verdict_clip.npz`` + ``meta.json`` in ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    validate_clip(clip)
    np.savez_compressed(path / "verdict_clip.npz", t=clip.t, pos=clip.pos, yaw=clip.yaw,
                        depicted_lean=clip.depicted_lean,
                        depicted_conf=clip.depicted_conf)
    (path / "meta.json").write_text(json.dumps(clip.meta, indent=1))
    return path / "verdict_clip.npz"
