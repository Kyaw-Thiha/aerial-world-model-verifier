"""Gate A: is there enough required lean for the coherence tier to mean anything?

The scale-free coherence tier correlates the depicted lean with the required lean.  If the
required lean barely varies over the clip, the correlation is dominated by the reader's own
noise and the tier's answer is a coin flip that a 2-4 deg reader perturbation (the yaw
control) can flip.  Gate A is a **path-side** abstention: it looks only at the required lean
of the commanded path, never at pixels, and abstains when

* the required lean does not carry enough signal (``weak_signal``); or
* the required tilt passes through the near-vertical region where a *lean vector* is not a
  valid representation of the thrust axis (``near_vertical``) -- above 90 deg the horizontal
  component saturates and the two lean axes stop being independent (K02 records 4,507 TII
  native frames with ``z < 0``).

Signal definition
-----------------
The coherence statistic is ``1 - min(r_x, r_y)``: the **weaker** of the two lean axes decides.
The governing path-side quantity is therefore the smaller of the two per-axis standard
deviations of the required lean over the clip's usable frames, in degrees:

    sigma_min = min( std(l_x), std(l_y) )              (degrees, from sin of the tilt)

with the principal-axis spread ``sigma_p`` and the tilt range reported alongside.  For a
series with per-axis signal std ``sigma_x`` and independent reader noise std ``sigma_n`` the
expected per-axis correlation is ``sigma_x / sqrt(sigma_x**2 + sigma_n**2)``; the threshold
is expressed as a multiple ``k`` of a measured ``sigma_n`` so the pre-registration can state
the SNR it implies.

Window view
-----------
The verdict statistics used by Tracks G and J are computed per 8-frame window and maximised.
The same signal statistics are returned per window (:func:`window_signal`) so a window-tier
gate can be evaluated; the local std of 8 samples is noisy, so the window rule is secondary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Representation limit for a lean vector.  At 90 deg the horizontal component is 1 and the
# two axes degenerate; 80 deg leaves a 10 deg margin where the lean is still informative.
NEAR_VERTICAL_DEG = 80.0
VAR_FLOOR = 0.01          # lean units (~0.57 deg), the published window floor


@dataclass
class SignalStats:
    sigma_x: float
    sigma_y: float
    sigma_min: float
    sigma_p: float
    tilt_med: float
    tilt_max: float
    tilt_range: float
    frac_above_80: float
    n_frames: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _flat_std(x: np.ndarray) -> float:
    return float(np.degrees(np.std(x)))


def signal_stats(lean_req: np.ndarray, tilt_deg: np.ndarray | None = None
                 ) -> SignalStats:
    """Path-side signal statistics for one clip (or one window)."""
    l = np.asarray(lean_req, dtype=np.float64)
    if l.ndim != 2 or l.shape[1] != 2:
        raise ValueError(f"lean_req must be (N, 2), got {l.shape}")
    if tilt_deg is None:
        tilt = np.degrees(np.arcsin(np.clip(np.linalg.norm(l, axis=1), 0.0, 1.0)))
    else:
        tilt = np.asarray(tilt_deg, dtype=np.float64)
    ok = np.isfinite(l).all(axis=1) & np.isfinite(tilt)
    if ok.sum() < 3:
        nan = float("nan")
        return SignalStats(nan, nan, nan, nan, nan, nan, nan, nan, int(ok.sum()))
    x = l[ok]
    sd = x.std(axis=0)
    c = x - x.mean(axis=0)
    ev = np.linalg.eigvalsh(c.T @ c / len(c))
    spread = float(np.degrees(np.sqrt(max(float(ev[-1]), 0.0))))
    return SignalStats(
        sigma_x=_flat_std(x[:, 0]), sigma_y=_flat_std(x[:, 1]),
        sigma_min=float(np.degrees(min(sd[0], sd[1]))), sigma_p=spread,
        tilt_med=float(np.nanmedian(tilt[ok])), tilt_max=float(np.nanmax(tilt[ok])),
        tilt_range=float(np.nanmax(tilt[ok]) - np.nanmin(tilt[ok])),
        frac_above_80=float(np.mean(tilt[ok] > NEAR_VERTICAL_DEG)),
        n_frames=int(ok.sum()))


def windows(n: int, length: int = 8, hop: int = 4) -> list[tuple[int, int]]:
    """The published window scheme (8 frames, hop 4, tail covered)."""
    if n <= length:
        return [(0, n)]
    starts = list(range(0, n - length + 1, hop))
    if starts[-1] + length < n:
        starts.append(n - length)
    return [(s, s + length) for s in starts]


def window_signal(lean_req: np.ndarray, tilt_deg: np.ndarray | None = None,
                  length: int = 8, hop: int = 4) -> list[dict]:
    """Per-window signal statistics, in the published window scheme."""
    l = np.asarray(lean_req, dtype=np.float64)
    out = []
    for lo, hi in windows(len(l), length, hop):
        st = signal_stats(l[lo:hi],
                          None if tilt_deg is None else np.asarray(tilt_deg)[lo:hi])
        d = st.as_dict()
        d.update({"start": int(lo), "stop": int(hi)})
        out.append(d)
    return out


def gate_a(clip_stats: SignalStats, sigma_threshold: float,
           near_vertical_deg: float = NEAR_VERTICAL_DEG) -> dict:
    """Apply Gate A to one clip.  Both conditions are pre-registered and path-side only."""
    reasons = []
    if not np.isfinite(clip_stats.sigma_min) or clip_stats.n_frames < 8:
        reasons.append("weak_signal")
    elif clip_stats.sigma_min < sigma_threshold:
        reasons.append("weak_signal")
    if np.isfinite(clip_stats.tilt_max) and clip_stats.tilt_max > near_vertical_deg:
        # a lean vector stops being a faithful representation as the thrust axis approaches
        # horizontal; the fraction above the limit is reported rather than hidden
        reasons.append("near_vertical")
    return {"testable": not reasons, "reasons": reasons,
            "sigma_min": clip_stats.sigma_min, "tilt_max": clip_stats.tilt_max}


def implied_snr(sigma_min: float, sigma_n: float) -> float:
    """Expected per-axis correlation for the weakest axis, the SNR argument's quantity."""
    if not (np.isfinite(sigma_min) and np.isfinite(sigma_n)) or sigma_n <= 0:
        return float("nan")
    sx2 = sigma_min ** 2
    sn2 = sigma_n ** 2
    return float(np.sqrt(sx2 / (sx2 + sn2)))
