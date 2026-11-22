"""Track J, item 6: the synthetic noise budget — how small a violation can the verdict see?

Pre-registered in ``configs/j02_noise_budget_prereg.yaml``. The path side of every real clip is
kept; the depicted side is replaced by a perfect depiction plus a controlled reader error. The
clip-level median residual and the G-round window scan statistic are then calibrated
leave-one-flight-out and swept over the size of a uniform tilt reduction.

CPU only. No GPU, no spend.

Usage
-----
PYTHONPATH=. python experiments/j02_noise_budget.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.verdict.schema import load_clip
from src.verdict.segments import frame_residual_deg

TII = [
    "flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
    "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM",
]
OUT = Path("results/j02_noise_budget")
CLIPS = Path("data/processed/track_d/clips")
DELTAS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0])
WINDOWS = [8, 12, 16, 24, 48]
Z = 1.645
MAXN = 49


# --------------------------------------------------------------------------- population
def load_population() -> list[dict]:
    units = []
    for flight in TII:
        for d in sorted((CLIPS / flight).glob("*__real")):
            c = load_clip(d / "verdict_clip.npz")
            if c.lean_req is None:
                continue
            units.append({"flight": flight, "clip": d.name,
                          "t": np.asarray(c.t, float),
                          "lean_req": np.asarray(c.lean_req, float),
                          "yaw": np.asarray(c.yaw, float),
                          "tilt_req_deg": np.asarray(c.tilt_req_deg, float),
                          "n": len(c.t)})
    return units


def axis_angle_deg(lean_a: np.ndarray, lean_b: np.ndarray) -> np.ndarray:
    """Angle (deg) between two thrust axes given their lean vectors, yaw-free."""
    n2a = np.clip(np.sum(lean_a ** 2, axis=-1), 0.0, 1.0)
    n2b = np.clip(np.sum(lean_b ** 2, axis=-1), 0.0, 1.0)
    cos = np.sum(lean_a * lean_b, axis=-1) + np.sqrt((1.0 - n2a) * (1.0 - n2b))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def apply_violation(lean_req: np.ndarray, delta_deg: float) -> np.ndarray:
    """Reduce the required tilt by ``delta_deg`` along its own direction, saturating at zero."""
    mag = np.linalg.norm(lean_req, axis=-1, keepdims=True)
    lhat = np.divide(lean_req, mag, out=np.zeros_like(lean_req), where=mag > 1e-9)
    new = np.clip(mag - np.sin(np.deg2rad(delta_deg)), 0.0, None)
    return lhat * new


# --------------------------------------------------------------------------- error model
def dct_split(x: np.ndarray, fs: float, cutoff: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    from scipy.fft import dct, idct

    n = len(x)
    coef = dct(x - x.mean(), type=2, norm="ortho")
    k = np.arange(n) * fs / (2.0 * n)
    slow = idct(np.where(k < cutoff, coef, 0.0), type=2, norm="ortho")
    return slow, x - x.mean() - slow


def make_error(n: int, dt: float, sigma: float, bias: float, tau_s: float,
               fast_share: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bias vector and a 2-D error series with the requested shares and correlation time."""
    rho = float(np.exp(-dt / max(tau_s, 1e-6)))
    white = rng.standard_normal((n, 2))
    ar = np.zeros_like(white)
    ar[0] = white[0]
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + np.sqrt(1.0 - rho ** 2) * white[i]
    slow = np.zeros_like(ar)
    fast = np.zeros_like(ar)
    for j in range(2):
        s, f = dct_split(ar[:, j], 1.0 / dt)
        slow[:, j], fast[:, j] = s, f

    def scale(a, target):
        sd = float(np.std(a))
        return a * (np.sqrt(target) / sd) if sd > 1e-9 else a

    slow = scale(slow, (1.0 - fast_share) * sigma ** 2)
    fast = scale(fast, fast_share * sigma ** 2)
    ang = rng.uniform(0.0, 2.0 * np.pi)
    b = bias * np.array([np.cos(ang), np.sin(ang)])
    return b, b[None, :] + slow + fast


def realised_params(err: np.ndarray, dt: float) -> dict:
    """Measured shares and tau_int of one generated series (2 axes pooled)."""
    from src.depicted.anatomy import acf, contiguous_runs

    t = np.arange(len(err)) * dt
    slow = np.zeros_like(err)
    fast = np.zeros_like(err)
    for j in range(2):
        slow[:, j], fast[:, j] = dct_split(err[:, j] - err[:, j].mean(), 1.0 / dt)
    centred = float(np.mean(slow ** 2) + np.mean(fast ** 2))
    return {"fast_share_of_centred": float(np.mean(fast ** 2) / centred),
            "tau_int_s": float(np.nanmean([acf(err[:, j], t, contiguous_runs(t))["tau_int_s"]
                                           for j in range(2)]))}


# --------------------------------------------------------------------------- statistics
def window_max(resid: np.ndarray, length: int, hop: int) -> float:
    """Maximum over complete windows of the window-median residual."""
    n = len(resid)
    if n <= length:
        return float(np.median(resid))
    vals = [float(np.median(resid[s:s + length])) for s in range(0, n - length + 1, hop)]
    return float(max(vals)) if vals else float("nan")


def lofo_tau(stats: np.ndarray, flights: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Per-clip leave-one-flight-out 95th percentile of a statistic."""
    out = np.full(len(stats), np.nan)
    for f in np.unique(flights):
        fit = stats[(flights != f) & np.isfinite(stats)]
        if len(fit) >= 5:
            out[flights == f] = float(np.quantile(fit, 1.0 - alpha))
    return out


def mdv_of(curve: dict, threshold: float = 0.5) -> float:
    d = sorted(curve)
    for i, delta in enumerate(d):
        if curve[delta] >= threshold:
            if i == 0:
                return float(delta)
            d0, d1 = d[i - 1], delta
            r0, r1 = curve[d0], curve[d1]
            return float(delta if r1 <= r0 else d0 + (threshold - r0) * (d1 - d0) / (r1 - r0))
    return float("nan")


def run_cell(units: list[dict], flights: np.ndarray, *, sigma: float, bias: float,
             tau_s: float, fast_share: float, draws: int, lengths, rng: np.random.Generator,
             error_fn=None) -> dict:
    """One parameter cell: (draws, delta, clip) statistics per window length."""
    n_clip = len(units)
    T = {L: np.full((draws, len(DELTAS), n_clip), np.nan) for L in lengths}
    for d in range(draws):
        err = np.zeros((n_clip, MAXN, 2))
        for f in np.unique(flights):
            ii = np.flatnonzero(flights == f)
            dt = float(np.median(np.diff(units[ii[0]]["t"])))
            if error_fn is None:
                _, e = make_error(MAXN, dt, sigma, bias, tau_s, fast_share, rng)
            else:
                e = error_fn(units, ii, dt, rng)
            err[ii] = e[None]
        for j, u in enumerate(units):
            n = u["n"]
            req = u["lean_req"]
            e = err[j, :n]
            for L in lengths:
                hop = 1 if L is None else max(1, L // 2)
                for di, delta in enumerate(DELTAS):
                    dep = apply_violation(req, delta) + e
                    r = axis_angle_deg(dep, req)
                    T[L][d, di, j] = (float(np.median(r)) if L is None
                                      else window_max(r, L, hop))
    return T


def analyse(T: dict, flights: np.ndarray) -> dict:
    out = {}
    for L, arr in T.items():
        det = {float(delta): float(np.mean(arr[:, i, :] > tau[None, :]))
               for i, delta in enumerate(DELTAS)
               for tau in [np.array([lofo_tau(arr[dr, 0, :], flights) for dr in range(arr.shape[0])])]}
        # the comprehension above re-derives tau for every delta; compute once instead
        taus = np.stack([lofo_tau(arr[dr, 0, :], flights) for dr in range(arr.shape[0])])
        det = {float(delta): float(np.mean(arr[:, i, :] > taus)) for i, delta in enumerate(DELTAS)}
        h0 = arr[:, 0, :]
        out[str(L)] = {
            "h0_median_deg": float(np.nanmedian(h0)),
            "tau_lofo_median_deg": float(np.nanmedian(taus)),
            "detection": det,
            "mdv_deg": mdv_of(det),
        }
    return out


def analytic_mdv(sigma: float, bias: float, tau_s: float, window_s: float) -> float:
    n_eff = window_s / max(tau_s, 1e-6)
    se = sigma / np.sqrt(n_eff)
    return float(Z * np.sqrt(se ** 2 + bias ** 2) + Z * se)


# --------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    units = load_population()
    flights = np.array([u["flight"] for u in units])
    print(f"clips {len(units)}  flights {len(np.unique(flights))}")

    # correctness: the fast axis formula must equal the frozen YZX reconstruction
    u = units[0]
    delta_lean = np.array([0.01, -0.02])
    fast = axis_angle_deg(u["lean_req"] + delta_lean, u["lean_req"])
    slow = frame_residual_deg(u["lean_req"] + delta_lean, u["lean_req"], u["yaw"])
    worst = float(np.nanmax(np.abs(fast - slow)))

    # realised parameters of the generator, at the centre point
    dt = float(np.median(np.diff(units[0]["t"])))
    rng = np.random.default_rng(1)
    _, e = make_error(MAXN, dt, 4.8, 2.0, 0.5, 0.5, rng)
    realised = realised_params(e, dt)

    draws = 120
    results = {"clips": len(units), "flights": int(len(np.unique(flights))),
               "draws": draws, "deltas": DELTAS.tolist(),
               "formula_check_max_abs_deg": worst,
               "generator_realised": realised,
               "settings": {}}

    # (1) window length at the measured centre point
    centre = dict(sigma=4.8, bias=2.0, tau_s=0.5, fast_share=0.5)
    T = run_cell(units, flights, draws=draws, lengths=[None] + WINDOWS, rng=rng, **centre)
    results["by_window"] = {"params": centre, **analyse(T, flights)}

    # (2) sigma x bias x tau grid, whole-clip median only
    grid = []
    for sigma in (3.0, 4.8, 8.0):
        for bias in (0.0, 2.0, 5.0):
            for tau_s in (0.2, 0.5, 1.0, 2.0):
                Tc = run_cell(units, flights, draws=draws // 2, lengths=[None], rng=rng,
                              sigma=sigma, bias=bias, tau_s=tau_s, fast_share=0.5)
                a = analyse(Tc, flights)["None"]
                grid.append({"sigma": sigma, "bias": bias, "tau_s": tau_s,
                             "mdv_deg": a["mdv_deg"],
                             "h0_median_deg": a["h0_median_deg"],
                             "tau_lofo_median_deg": a["tau_lofo_median_deg"],
                             "analytic_mdv_deg": analytic_mdv(sigma, bias, tau_s, 4.0),
                             "detection": a["detection"]})
    results["grid"] = grid

    # (3) slow-share sensitivity at the centre point
    shares = []
    for fs in (0.2, 0.5, 0.8):
        Tc = run_cell(units, flights, draws=draws // 2, lengths=[None], rng=rng,
                      sigma=4.8, bias=2.0, tau_s=0.5, fast_share=fs)
        a = analyse(Tc, flights)["None"]
        shares.append({"fast_share": fs, "mdv_deg": a["mdv_deg"],
                       "h0_median_deg": a["h0_median_deg"],
                       "tau_lofo_median_deg": a["tau_lofo_median_deg"],
                       "detection": a["detection"]})
    results["by_share"] = shares

    # (4) replay of the MEASURED reader error (validation of the generator)
    from src.depicted.anatomy import load_flight, axis_errors

    tables = {f: load_flight(f) for f in TII}
    per_flight_err = {}
    for f, tab in tables.items():
        err = axis_errors(tab, "frozen")
        per_flight_err[f] = (np.asarray(tab.t), np.column_stack([err["pitch"], err["roll"]]))

    def replay(units_, ii, dt_, rng_):
        f = units_[ii[0]]["flight"]
        t_ref, e_ref = per_flight_err[f]
        out = np.zeros((len(units_), MAXN, 2))
        for k, j in enumerate(ii):
            t_clip = units_[j]["t"]
            idx = np.searchsorted(t_ref, t_clip).clip(0, len(t_ref) - 1)
            out[k, : len(t_clip)] = e_ref[idx]
        return out

    Tr = run_cell(units, flights, draws=1, lengths=[None] + WINDOWS, rng=rng,
                  sigma=0.0, bias=0.0, tau_s=1.0, fast_share=0.5, error_fn=replay)
    replay_out = {}
    for L, arr in Tr.items():
        tau = lofo_tau(arr[0, 0, :], flights)
        replay_out[str(L)] = {
            "h0_median_deg": float(np.nanmedian(arr[0, 0, :])),
            "tau_lofo_median_deg": float(np.nanmedian(tau)),
            "detection": {float(dl): float(np.mean(arr[0, i, :] > tau))
                          for i, dl in enumerate(DELTAS)},
            "mdv_deg": mdv_of({float(dl): float(np.mean(arr[0, i, :] > tau))
                               for i, dl in enumerate(DELTAS)}),
        }
    results["replay_measured_error"] = replay_out

    (OUT / "summary.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("formula_check_max_abs_deg", "generator_realised")}, indent=1))
    print("by_window mdv:", {k: v["mdv_deg"] for k, v in results["by_window"].items()
                             if k != "params"})
    print("replay mdv:", {k: v["mdv_deg"] for k, v in replay_out.items()})
    print("replay h0 median / tau:", {k: (round(v["h0_median_deg"], 2),
                                          round(v["tau_lofo_median_deg"], 2))
                                      for k, v in replay_out.items()})


if __name__ == "__main__":
    main()
