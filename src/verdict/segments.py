# COPIED UNCHANGED from Track G: src/verdict/segments.py at branch `track-g2-segments`
# commit 4c6eacf, which itself imported it unchanged from `track-g-segments2` 463bfcc.
# Track J imports it for the noise budget (J02) so the window machinery is the published one.
# Do not edit here: a change belongs on the G branch and is re-copied with a new commit id.
"""Segment-level detection: sliding windows, a scan statistic, and its calibration.

The frozen verdict is one clip-level number, so a violation that covers part of a clip is
averaged away and a single bad estimate can reject a good clip. This module slides a window
over a clip, computes the clip-verdict tiers on each window, and calibrates the **maximum
over windows** conformally, which is what keeps the whole-clip false-reject at ``alpha``.

Definitions are the ones pre-registered in ``configs/g01_segment_prereg.yaml``:

* window length ``max(rule window, 8 clip frames)``, hop 4 frames. The label-free rule
  (``0.5 / f95``) returns 0.05-0.22 s at the 274.73 Hz truth rate, i.e. 0.6-2.6 frames at
  the 12 Hz clip rate, so its 8-sample floor is applied at the rate the statistic is
  actually computed on (clip frames) and binds on every TII clip;
* per window: the median per-frame depicted-vs-required thrust-axis angle, the coherence
  correlation score ``1 - min(r_x, r_y)``, and the sign score ``1 - min(s_x, s_y)``;
* per window abstention: fewer than 6 usable frames or fewer than 2 signal frames
  (required tilt >= 10 deg) abstains; a tier whose lean has negligible variance on an axis
  abstains for that tier only;
* clip statistic: the maximum over windows (a scan statistic).  The conformal threshold on
  that maximum gives one family-wise budget per clip.

Nothing here reads mocap beyond what a schema-v2 clip already carries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.depicted.clips import up_body_from_lean
from src.depicted.metrics import angle_deg
from src.depicted.stat_a2 import depicted_axis_from_up_body
from src.verdict.coherence import _pearson, _sign_agreement

TIERS = ("residual", "coherence", "sign")
WINDOW_FRAMES = 8
HOP_FRAMES = 4
MIN_USABLE_FRAMES = 6
MIN_SIGNAL_FRAMES = 2
SIGNAL_TILT_DEG = 10.0
VAR_FLOOR = 0.01          # lean units (~0.57 deg) below which a tier abstains for a window


@dataclass
class WindowStat:
    """One window's tier statistics.  ``nan`` means that tier abstained for this window."""

    start: int
    stop: int
    n_usable: int
    n_signal: int
    residual: float = float("nan")
    coherence: float = float("nan")
    sign: float = float("nan")
    abstained: bool = False
    reasons: list = field(default_factory=list)

    @property
    def t0(self) -> int:
        return self.start

    def get(self, tier: str) -> float:
        return float(getattr(self, tier))


def frame_residual_deg(lean_dep: np.ndarray, lean_req: np.ndarray,
                       yaw: np.ndarray) -> np.ndarray:
    """Per-frame angle between the depicted and required thrust axes, in degrees.

    Same construction as the frozen median-residual tier (``src.depicted.stat_a2``): each lean
    vector is inverted to the body-frame world-up vector with ``up_body_from_lean``, the full
    attitude is rebuilt with the path's yaw, and the residual is the angle between the two
    body z (thrust) axes.  The yaw is passed because the ZYX reconstruction is what the frozen
    tier uses, and using the frozen route is what makes this number comparable with the
    published 14.73 deg threshold.  NaN rows stay NaN.
    """
    ax_dep = depicted_axis_from_up_body(up_body_from_lean(lean_dep), yaw)
    ax_req = depicted_axis_from_up_body(up_body_from_lean(lean_req), yaw)
    return angle_deg(ax_dep, ax_req)


def _windows(n: int, length: int, hop: int) -> list[tuple[int, int]]:
    if n <= length:
        return [(0, n)]
    starts = list(range(0, n - length + 1, hop))
    if starts[-1] + length < n:
        starts.append(n - length)          # always cover the tail
    return [(s, s + length) for s in starts]


def window_statistics(clip, length: int = WINDOW_FRAMES, hop: int = HOP_FRAMES,
                      lag_ms: float = 0.0) -> list[WindowStat]:
    """All window statistics for one clip (see the module docstring)."""
    from src.verdict.fit import _shift_series

    lean_dep = np.asarray(clip.depicted_lean, dtype=np.float64)
    if clip.lean_req is None:
        raise ValueError("segments need schema v2 lean_req; re-export the clip")
    lean_req = _shift_series(np.asarray(clip.lean_req, dtype=np.float64),
                             np.asarray(clip.t, dtype=np.float64), lag_ms / 1000.0)
    resid = frame_residual_deg(lean_dep, lean_req, np.asarray(clip.yaw, dtype=np.float64))
    tilt_req = (np.asarray(clip.tilt_req_deg, dtype=np.float64)
                if clip.tilt_req_deg is not None
                else np.degrees(np.arcsin(np.clip(np.linalg.norm(lean_req, axis=1), 0, 1))))
    out: list[WindowStat] = []
    for lo, hi in _windows(len(clip.t), length, hop):
        dep = lean_dep[lo:hi]
        req = lean_req[lo:hi]
        r = resid[lo:hi]
        usable = np.isfinite(dep).all(axis=1) & np.isfinite(req).all(axis=1)
        n_usable = int(usable.sum())
        n_signal = int(np.sum(usable & (tilt_req[lo:hi] >= SIGNAL_TILT_DEG)))
        w = WindowStat(start=lo, stop=hi, n_usable=n_usable, n_signal=n_signal)
        if n_usable < MIN_USABLE_FRAMES:
            w.abstained = True
            w.reasons.append(f"usable_frames {n_usable} < {MIN_USABLE_FRAMES}")
        if n_signal < MIN_SIGNAL_FRAMES:
            w.abstained = True
            w.reasons.append(f"signal_frames {n_signal} < {MIN_SIGNAL_FRAMES}")
        if not w.abstained:
            w.residual = float(np.nanmedian(r[usable]))
            d = dep[usable]
            q = req[usable]
            if (np.std(d[:, 0]) > VAR_FLOOR and np.std(d[:, 1]) > VAR_FLOOR
                    and np.std(q[:, 0]) > VAR_FLOOR and np.std(q[:, 1]) > VAR_FLOOR):
                rs = [_pearson(d[:, i], q[:, i]) for i in range(2)]
                ss = [_sign_agreement(d[:, i], q[:, i]) for i in range(2)]
                w.coherence = float(1.0 - min(rs))
                w.sign = float(1.0 - min(ss))
        out.append(w)
    return out


def clip_max(stats: list[WindowStat], tier: str) -> float:
    """The scan statistic: maximum over usable windows of one tier."""
    vals = [s.get(tier) for s in stats if not s.abstained]
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def clip_median_whole(clip) -> float:
    """The frozen-style whole-clip median residual (no windows), for the baseline."""
    lean_dep = np.asarray(clip.depicted_lean, dtype=np.float64)
    resid = frame_residual_deg(lean_dep, np.asarray(clip.lean_req, dtype=np.float64),
                               np.asarray(clip.yaw, dtype=np.float64))
    ok = np.isfinite(resid)
    return float(np.nanmedian(resid[ok])) if ok.any() else float("nan")


def conformal_threshold(values, alpha: float = 0.05) -> float:
    """Upper conformal threshold: the ``1 - alpha`` quantile of the fit values."""
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=np.float64)
    if len(v) < 8:
        return float("nan")
    return float(np.quantile(v, 1.0 - alpha))


def flagged_windows(stats: list[WindowStat], tier: str, tau: float) -> np.ndarray:
    """Boolean mask over frames: the union of windows whose tier statistic exceeds ``tau``."""
    n = max((s.stop for s in stats), default=0)
    mask = np.zeros(n, dtype=bool)
    for s in stats:
        v = s.get(tier)
        if not s.abstained and np.isfinite(v) and v > tau:
            mask[s.start:s.stop] = True
    return mask


def merge_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[lo, hi)`` runs of True."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def interval_scores(flag: np.ndarray, truth: np.ndarray) -> dict:
    """Frame-level IoU/F1 and onset/offset error of a flagged mask against the true interval."""
    f = np.asarray(flag, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    inter = float(np.sum(f & t))
    union = float(np.sum(f | t))
    iou = inter / union if union > 0 else float("nan")
    prec = inter / float(f.sum()) if f.sum() else 0.0
    rec = inter / float(t.sum()) if t.sum() else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    det = merge_intervals(f)
    tru = merge_intervals(t)
    onset = float(det[0][0] - tru[0][0]) if det and tru else float("nan")
    off = float(det[-1][1] - tru[-1][1]) if det and tru else float("nan")
    return {"iou": iou, "f1": float(f1), "precision": float(prec), "recall": float(rec),
            "onset_err_frames": onset, "offset_err_frames": off,
            "n_segments": len(det), "detected": bool(det)}


def clip_max_persistent(stats: list[WindowStat], tier: str, n: int = 2) -> float:
    """The persistence scan statistic: max over ``n`` consecutive windows of their minimum.

    A single glitchy window cannot produce a value above the per-window level, so this is the
    pre-registered "persistence" variant (``n = 2``): a violation has to last at least ``n``
    windows.  Windows that abstained break the run (their value is NaN).
    """
    if n <= 1:
        return clip_max(stats, tier)
    vals = [s.get(tier) if not s.abstained else float("nan") for s in stats]
    if len(vals) < n:
        return float("nan")
    out = []
    for i in range(len(vals) - n + 1):
        seg = vals[i:i + n]
        if all(np.isfinite(v) for v in seg):
            out.append(float(min(seg)))
    return float(max(out)) if out else float("nan")


def flagged_windows_persistent(stats: list[WindowStat], tier: str, tau: float,
                               n: int = 2) -> np.ndarray:
    """Union of the runs of ``n`` consecutive windows whose minimum exceeds ``tau``."""
    length = max((s.stop for s in stats), default=0)
    mask = np.zeros(length, dtype=bool)
    if n <= 1:
        return flagged_windows(stats, tier, tau)
    vals = [s.get(tier) if not s.abstained else float("nan") for s in stats]
    for i in range(len(vals) - n + 1):
        seg = vals[i:i + n]
        if all(np.isfinite(v) for v in seg) and min(seg) > tau:
            mask[stats[i].start:stats[i + n - 1].stop] = True
    return mask
