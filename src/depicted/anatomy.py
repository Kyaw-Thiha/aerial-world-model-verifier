"""Reader-error anatomy: what the depicted-tilt error is made of, and how long it lives.

Track J, item 1-5 of handoff 19. Everything here is pure numpy on stored per-frame arrays;
the definitions are fixed in ``configs/j01_anatomy_prereg.yaml`` (and its amendment A).

Conventions reused unchanged from the rest of the project:

* ``up_cam`` is the world-up direction in camera coordinates (GeoCalib's convention).
* ``R_cb`` maps camera -> body, from the flight's recorded uptilt (TII) or the Kalibr
  ``T_cam_imu`` (UZH).
* the lean vector is ``depicted_lean_from_gravity(up_cam, R_cb)``, i.e. the horizontal part of
  the thrust axis in the yaw-aligned frame: axis 0 = pitch (forward), axis 1 = roll (right).

The decomposition is exact rather than a filter approximation. The per-flight mean is removed
first (the bias). Each contiguous run is then split with an orthogonal DCT-II basis at 0.5 Hz,
so the run's energy is exactly the sum of a slow and a fast part, with no cross term and no
edge effect from a causal filter. On this corpus one run is the whole flight, so the split
spans about 0.005 Hz to 0.5 Hz (slow) and 0.5 Hz to Nyquist (fast).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.spatial.transform import Rotation

from src.depicted.metrics import angle_deg
from src.perception.frames import body_to_world, camera_to_body_pitched
from src.verdict.lean import depicted_lean_from_gravity, lean_to_roll_pitch

E3 = np.array([0.0, 0.0, 1.0])
SLOW_CUTOFF_HZ = 0.5


# --------------------------------------------------------------------------- inputs
@dataclass
class FrameTable:
    """One unit's frames: truth, both readers, confidence, and the regressors."""

    unit: str
    names: np.ndarray
    t: np.ndarray
    up_true: np.ndarray
    up_frozen: np.ndarray
    up_ft: np.ndarray
    conf: np.ndarray
    speed: np.ndarray
    tilt_deg: np.ndarray
    yaw: np.ndarray
    clip_index: np.ndarray
    uptilt_deg: float | None
    R_cb: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)


def _clip_index(names: np.ndarray, clips_json: dict) -> tuple[np.ndarray, list[str]]:
    """Clip id per frame, from the p3a clip table (index ranges into the same name array)."""
    out = np.full(len(names), "", dtype=object)
    ids: list[str] = []
    for c in clips_json["clips"]:
        a, b = int(c["index"][0]), int(c["index"][1])
        out[a:b] = c["clip_id"]
        ids.append(c["clip_id"])
    return out, ids


def load_flight(flight: str, *, root: str | Path = ".", p3a_dir: str | None = None,
                folds_dir: str | Path = "results/j_inputs/g03_folds",
                gate_dir: str | Path = "results/gate_a2/tii") -> FrameTable:
    """Join the mocap spec, the two readers' per-frame outputs, and the S3 confidence."""
    root = Path(root)
    p3a_dir = Path(p3a_dir) if p3a_dir else root / "data/processed/p3a/p3a_v1"
    spec = np.load(p3a_dir / f"{flight}.npz", allow_pickle=True)
    fold = json.loads(Path(folds_dir, f"tii__{flight}.json").read_text())
    rows = fold["held_out"]

    names = np.array([r["frame"] for r in rows])
    if len(names) != len(spec["names"]) or not np.array_equal(names, spec["names"]):
        raise ValueError(f"{flight}: reader fold and mocap spec name arrays are not identical")

    gate_path = Path(gate_dir, f"{flight}__identity.jsonl")
    conf = np.full(len(names), np.nan)
    conf_by_name: dict[str, float] = {}
    if gate_path.exists():
        for line in gate_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            conf_by_name[r["frame"]] = float(r.get("latitude_confidence_mean", np.nan))
    conf = np.array([conf_by_name.get(n, np.nan) for n in names])

    R_bw = body_to_world(spec["quat"], "xyzw")
    uptilt = float(spec["uptilt_deg"])
    R_cb = camera_to_body_pitched(np.deg2rad(uptilt))
    clips = json.loads((p3a_dir / "clips.json").read_text())
    clip_index, _ = _clip_index(names, clips)
    return FrameTable(
        unit=flight, names=names, t=np.asarray(spec["t"], float),
        up_true=np.array([r["up_cam"] for r in rows], float),
        up_frozen=np.array([r["up_cam_base"] for r in rows], float),
        up_ft=np.array([r["up_cam_ft"] for r in rows], float),
        conf=conf, speed=np.asarray(spec["speed"], float),
        tilt_deg=np.asarray(spec["body_tilt_deg"], float),
        yaw=Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0],
        clip_index=clip_index, uptilt_deg=uptilt, R_cb=R_cb,
        meta={"reader_source": str(Path(folds_dir, f"tii__{flight}.json")),
              "conf_source": str(gate_path) if gate_path.exists() else None},
    )


def load_uzh_sequence(seq: str, *, root: str | Path = ".",
                      folds_dir: str | Path = "results/j_inputs/g03_folds") -> FrameTable:
    """UZH forward sequence: same join, Kalibr mount, no S3 stored (left NaN)."""
    from src.depicted.uzh import published_mount

    root = Path(root)
    meta = json.loads((root / f"data/uzh/{seq}/meta.json").read_text())
    truth = np.load(root / f"data/uzh/{seq}/truth.npz")
    fold = json.loads(Path(folds_dir, f"uzh__{seq}.json").read_text())
    rows = fold["held_out"]
    names = np.array([r["frame"] for r in rows])
    n = len(names)
    fps = float(meta["fps_measured"])
    R_cb, _ = published_mount(meta)
    quat = np.asarray(truth["quat"], float)[:n]
    R_bw = body_to_world(quat, "xyzw")
    return FrameTable(
        unit=seq, names=names, t=np.arange(n) / fps,
        up_true=np.array([r["up_cam"] for r in rows], float),
        up_frozen=np.array([r["up_cam_base"] for r in rows], float),
        up_ft=np.array([r["up_cam_ft"] for r in rows], float),
        conf=np.full(n, np.nan), speed=np.full(n, np.nan),
        tilt_deg=np.degrees(np.arccos(np.clip(np.einsum("nji,j->ni", R_bw, E3)[:, 2], -1, 1))),
        yaw=Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0],
        clip_index=np.full(n, "", dtype=object), uptilt_deg=None, R_cb=R_cb,
        meta={"fps": fps, "truth": str(root / f"data/uzh/{seq}/truth.npz")},
    )


# --------------------------------------------------------------------------- errors
def body_up(up_cam: np.ndarray, R_cb: np.ndarray) -> np.ndarray:
    """World-up in the body frame (FLU), unit rows."""
    u = np.asarray(up_cam, float) @ np.asarray(R_cb, float).T
    n = np.linalg.norm(u, axis=1, keepdims=True)
    return u / np.where(n > 0, n, np.nan)


def attitude_angles(u_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ZYX (roll, pitch) from world-up in the body frame, radians.

    ``u_b = (-sin(theta), sin(phi) cos(theta), cos(phi) cos(theta))``, the same convention as
    ``depicted_lean_from_gravity``.
    """
    u = np.asarray(u_b, float)
    pitch = -np.arcsin(np.clip(u[:, 0], -1.0, 1.0))
    roll = np.arctan2(u[:, 1], u[:, 2])
    return roll, pitch


def tangent_basis(roll: np.ndarray, pitch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal frame (e_pitch, e_roll) on the sphere of up-directions, plus u itself.

    ``e_pitch = du/dtheta`` and ``e_roll = (1/cos theta) du/dphi`` are orthonormal and both
    orthogonal to ``u``, so a tangent displacement written in this basis splits the error into
    two additive degrees of freedom whose Euclidean length is exactly the angle between the
    two up-directions.
    """
    sth, cth = np.sin(pitch), np.cos(pitch)
    sphi, cphi = np.sin(roll), np.cos(roll)
    u = np.column_stack([-sth, sphi * cth, cphi * cth])
    e_th = np.column_stack([-cth, -sphi * sth, -cphi * sth])
    e_ph = np.column_stack([np.zeros_like(sth), cphi, -sphi])
    return e_th, e_ph, u


def tangent_error(up_est: np.ndarray, up_ref: np.ndarray, R_cb: np.ndarray) -> dict:
    """Exact 2-axis error: the log-map of the estimate at the reference, in the lean basis.

    Returns ``pitch`` and ``roll`` in degrees (the two tangent components) and ``theta``, the
    angle between the two up-directions. By construction ``hypot(pitch, roll) == theta`` to
    numerical precision, so the two axes are an orthogonal decomposition of the scalar E2/G03
    error rather than a second, differently-scaled quantity.
    """
    u_e = body_up(up_est, R_cb)
    u_r = body_up(up_ref, R_cb)
    roll_t, pitch_t = attitude_angles(u_r)
    e_th, e_ph, _ = tangent_basis(roll_t, pitch_t)
    cos = np.clip(np.einsum("ij,ij->i", u_e, u_r), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    perp = u_e - cos[:, None] * u_r
    n = np.linalg.norm(perp, axis=1, keepdims=True)
    unit = np.divide(perp, n, out=np.zeros_like(perp), where=n > 1e-12)
    return {"pitch": ang * np.einsum("ij,ij->i", unit, e_th),
            "roll": ang * np.einsum("ij,ij->i", unit, e_ph),
            "theta": ang}


def axis_errors(tab: FrameTable, reader: str = "frozen") -> dict[str, np.ndarray]:
    """Lean-plane error per axis (degrees) and the scalar thrust-axis angle (degrees)."""
    up_est = tab.up_frozen if reader == "frozen" else tab.up_ft
    return tangent_error(up_est, tab.up_true, tab.R_cb)


def consistency_gap(err: dict[str, np.ndarray]) -> float:
    """Largest 99.9th-percentile difference between |lean error| and the scalar angle."""
    mag = np.hypot(err["roll"], err["pitch"])
    ok = np.isfinite(mag) & np.isfinite(err["theta"])
    if not ok.any():
        return float("nan")
    return float(np.percentile(np.abs(mag[ok] - err["theta"][ok]), 99.9))


def trim_mask(theta: np.ndarray, drop: float = 0.10) -> np.ndarray:
    """E2's rule: drop the worst ``drop`` fraction of frames by the scalar angle."""
    ok = np.isfinite(theta)
    if ok.sum() == 0:
        return ok
    thr = float(np.quantile(theta[ok], 1.0 - drop))
    return ok & (theta <= thr)


# --------------------------------------------------------------------------- runs and split
def contiguous_runs(t: np.ndarray, gap_factor: float = 1.5) -> list[tuple[int, int]]:
    """Half-open index ranges of contiguous runs; a new run starts across a gap."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return [(0, len(t))]
    dt = float(np.median(np.diff(t)))
    brk = np.flatnonzero(np.diff(t) > gap_factor * dt) + 1
    edges = np.concatenate([[0], brk, [len(t)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def dct_split(x: np.ndarray, fs: float, cutoff_hz: float = SLOW_CUTOFF_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal slow/fast split of one run: slow = DCT coefficients below ``cutoff_hz``."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return np.zeros_like(x), x.copy()
    coef = dct(x, type=2, norm="ortho")
    kmax = np.arange(n) * fs / (2.0 * n)
    slow = idct(np.where(kmax < cutoff_hz, coef, 0.0), type=2, norm="ortho")
    return slow, x - slow


def decompose(values: np.ndarray, t: np.ndarray, runs: list[tuple[int, int]],
              keep: np.ndarray | None = None) -> dict:
    """Bias / slow / fast split of one axis for one unit.

    The bias is the mean over the *kept* frames. The slow/fast split is a linear operation on
    the residual series and is therefore computed on **all** finite frames (dropping frames
    before a DCT would create discontinuities); the reported moments are averages over the
    kept frames. ``closure`` records the residual cross term that trimming leaves behind
    (exactly zero on the full frame set, by orthogonality).
    """
    v = np.asarray(values, float)
    fin = np.isfinite(v)
    keep = fin if keep is None else (np.asarray(keep, bool) & fin)
    if keep.sum() < 4:
        return {"n": int(keep.sum()), "bias": float("nan"), "ms_total": float("nan"),
                "share_bias": float("nan"), "share_slow": float("nan"),
                "share_fast": float("nan"), "fast_share_of_centred": float("nan"),
                "n_runs": 0, "slow_cutoff_hz": SLOW_CUTOFF_HZ, "closure": float("nan")}
    b = float(np.mean(v[keep]))
    r = np.where(fin, v - b, np.nan)
    slow = np.zeros_like(r)
    fast = np.zeros_like(r)
    used = 0
    for a, bnd in runs:
        seg = r[a:bnd]
        if not np.isfinite(seg).all() or len(seg) < 8:
            continue
        dt = float(np.median(np.diff(t[a:bnd])))
        s, f = dct_split(seg, 1.0 / dt)
        slow[a:bnd] = s
        fast[a:bnd] = f
        used += 1
    good = keep & np.isfinite(slow)
    ms_total = float(np.mean(v[good] ** 2))
    ms_bias = float(b ** 2)
    ms_slow = float(np.mean(slow[good] ** 2))
    ms_fast = float(np.mean(fast[good] ** 2))
    centred = ms_slow + ms_fast
    return {
        "n": int(good.sum()), "bias": b, "ms_total": ms_total,
        "ms_bias": ms_bias, "ms_slow": ms_slow, "ms_fast": ms_fast,
        "closure": ms_total - (ms_bias + ms_slow + ms_fast),
        "share_bias": ms_bias / ms_total if ms_total > 0 else float("nan"),
        "share_slow": ms_slow / ms_total if ms_total > 0 else float("nan"),
        "share_fast": ms_fast / ms_total if ms_total > 0 else float("nan"),
        "fast_share_of_centred": ms_fast / centred if centred > 0 else float("nan"),
        "n_runs": used, "slow_cutoff_hz": SLOW_CUTOFF_HZ,
    }


def decompose_axes(err: dict[str, np.ndarray], t: np.ndarray, runs: list[tuple[int, int]],
                   keep: np.ndarray | None = None) -> dict:
    """Per-axis decomposition plus the pooled two-axis share the decision rule uses."""
    keep = np.ones(len(t), bool) if keep is None else np.asarray(keep, bool)
    per_axis = {}
    ms = {"bias": 0.0, "slow": 0.0, "fast": 0.0, "total": 0.0}
    n_tot = 0
    for ax in ("pitch", "roll"):
        d = decompose(err[ax], t, runs, keep)
        per_axis[ax] = d
        for k in ms:
            ms[k] += d[f"ms_{k}"] * d["n"]
        n_tot += d["n"]
    pooled = {f"share_{k}": ms[k] / ms["total"] if ms["total"] > 0 else float("nan")
              for k in ("bias", "slow", "fast")}
    pooled["ms_total"] = ms["total"]
    if ms["slow"] + ms["fast"] > 0:
        pooled["fast_share_of_centred"] = ms["fast"] / (ms["slow"] + ms["fast"])
    else:
        pooled["fast_share_of_centred"] = float("nan")
    pooled["n"] = n_tot
    return {"per_axis": per_axis, "pooled": pooled}


# --------------------------------------------------------------------------- time structure
def acf(values: np.ndarray, t: np.ndarray, runs: list[tuple[int, int]],
        max_lag_s: float = 5.0) -> dict:
    """Autocorrelation of the mean-removed series, pairs inside runs only."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 8:
        return {"lags_s": np.array([]), "rho": np.array([]), "tau_int_s": float("nan")}
    v = np.where(ok, v - np.nanmean(v[ok]), np.nan)
    dt = float(np.median(np.diff(t)))
    lags = np.arange(0, int(round(max_lag_s / dt)) + 1)
    rho = np.full(len(lags), np.nan)
    var = float(np.nanmean(v ** 2))
    if var <= 0:
        return {"lags_s": lags * dt, "rho": rho, "tau_int_s": float("nan")}
    for i, k in enumerate(lags):
        acc, cnt = 0.0, 0
        for a, b in runs:
            seg = v[a:b]
            if len(seg) <= k:
                continue
            x, y = seg[: len(seg) - k], seg[k:]
            good = np.isfinite(x) & np.isfinite(y)
            acc += float(np.sum(x[good] * y[good]))
            cnt += int(good.sum())
        if cnt >= 20:
            rho[i] = acc / (cnt * var)
    # integral time, truncated at the first non-positive lag
    neg = np.flatnonzero(rho[1:] <= 0)
    stop = (neg[0] + 1) if len(neg) else len(rho) - 1
    tau = dt * (1.0 + 2.0 * float(np.nansum(rho[1:stop + 1])))
    return {"lags_s": lags * dt, "rho": rho, "tau_int_s": max(tau, 0.0),
            "truncation_lag": int(stop)}


def periodogram(values: np.ndarray, t: np.ndarray, runs: list[tuple[int, int]]) -> dict:
    """One-sided periodogram per run (Hann window, constant detrend), averaged within a unit."""
    from scipy.signal import periodogram as _pg

    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 16:
        return {"freq_hz": np.array([]), "psd": np.array([])}
    v = np.where(ok, v - np.nanmean(v[ok]), np.nan)
    acc, fs_ref = None, None
    for a, b in runs:
        seg = v[a:b]
        if not np.isfinite(seg).all() or len(seg) < 16:
            continue
        fs = 1.0 / float(np.median(np.diff(t[a:b])))
        f, p = _pg(seg, fs=fs, window="hann", detrend="constant", scaling="density")
        if acc is None:
            acc, fs_ref = p, f
        elif len(f) == len(acc):
            acc = acc + p
    if acc is None:
        return {"freq_hz": np.array([]), "psd": np.array([])}
    return {"freq_hz": fs_ref, "psd": acc}


# --------------------------------------------------------------------------- bias structure
def fit_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation R minimising sum |R src_i - dst_i|^2 (Wahba / orthogonal Procrustes)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[ok], dst[ok]
    src = src / np.linalg.norm(src, axis=1, keepdims=True)
    dst = dst / np.linalg.norm(dst, axis=1, keepdims=True)
    h = dst.T @ src
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def rotation_angle_deg(r: np.ndarray) -> float:
    c = (np.trace(np.asarray(r, float)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def small_ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> dict:
    """OLS with one-way cluster-robust standard errors (clusters = units)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x).all(1)
    y, x, clusters = y[ok], x[ok], np.asarray(clusters)[ok]
    if len(y) < 10:
        return {"beta": [], "se": [], "n": int(len(y))}
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    meat = np.zeros((k, k))
    for g in np.unique(clusters):
        m = clusters == g
        s = x[m].T @ resid[m]
        meat += np.outer(s, s)
    g = len(np.unique(clusters))
    scale = g / max(1, g - 1) * (n - 1) / max(1, n - k)
    cov = scale * xtx_inv @ meat @ xtx_inv
    return {"beta": beta.tolist(), "se": np.sqrt(np.diag(cov)).tolist(), "n": int(n), "clusters": int(g)}


# --------------------------------------------------------------------------- tail
def tail_runs(mask: np.ndarray, runs: list[tuple[int, int]]) -> list[int]:
    """Lengths (frames) of maximal runs of True inside each contiguous run."""
    out: list[int] = []
    for a, b in runs:
        seg = np.asarray(mask[a:b], bool)
        i = 0
        while i < len(seg):
            if seg[i]:
                j = i
                while j < len(seg) and seg[j]:
                    j += 1
                out.append(j - i)
                i = j
            else:
                i += 1
    return out


# --------------------------------------------------------------------------- bootstrap
def cluster_bootstrap(groups: list[str], stat_fn, n_boot: int = 5000, seed: int = 1) -> dict:
    """Flight-clustered bootstrap of an arbitrary pooled statistic.

    ``stat_fn(index_array)`` must take an integer index array into the frame arrays and return
    a float. Clusters are resampled with replacement; a draw that cannot be computed is
    dropped and the usable fraction is reported.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[p] for p in pick])
        v = stat_fn(idx)
        if np.isfinite(v):
            out.append(float(v))
    if not out:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0, "frac_usable": 0.0}
    return {"lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "n_boot": len(out), "frac_usable": len(out) / n_boot}
