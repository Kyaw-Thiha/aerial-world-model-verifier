"""Track V, Part 0: the feasibility analysis and the pre-registered stop rule.

Reads the per-window plane fits written by ``remote/v_recon.py`` (pulled to
``data/v/recon/``) and computes exactly what
``configs/v00_feasibility_prereg.yaml`` + amendments 1 and 2 define:

* per-window ``e = log(h(t2)/h(t1)) - log(ratio_cmd)`` and its absolute value;
* per-clip medians / p90 / coverage;
* the defect reference (removed, x0.5, x1.5, reversed) from the truth paths;
* the two stop-rule conditions, applied as written.

No threshold is chosen here. Nothing is tuned.

Usage
-----
PYTHONPATH=. python experiments/v00_feasibility.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.vertical import metrics as M
from src.vertical import recon_io as RIO

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "data/v/units.json"
RECON = ROOT / "data/v/recon"
OUT = ROOT / "results/v/v00"


def load_unit(unit: dict) -> dict:
    return RIO.load_unit(RECON, unit["unit"])


def unit_rows(unit: dict, data: dict) -> tuple[list[dict], list[dict]]:
    truth = unit["truth"]
    mode = "sim" if unit["kind"] == "sim" else "real"
    return RIO.unit_rows(data, np.asarray(truth["t"], float),
                         np.asarray(truth["pos"], float), mode=mode)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    units = json.loads(UNITS.read_text())["units"]
    by_unit: dict[str, list[dict]] = {}
    diags: dict[str, list[dict]] = {}
    meta = {}
    unit_population = {u["unit"]: u.get("population", "pilot") for u in units}
    for u in units:
        if u["kind"] not in ("sim", "uzh"):
            continue
        if u["kind"] == "uzh" and u.get("population", "pilot") != "pilot":
            continue
        if not (RECON / f"{u['unit']}.json").exists():
            print(f"MISSING recon for {u['unit']}")
            continue
        data = load_unit(u)
        rows, diag = unit_rows(u, data)
        by_unit[u["unit"]] = rows
        diags[u["unit"]] = diag
        meta[u["unit"]] = {"kind": u["kind"], "n_frames": u["n_frames"],
                           "n_chunks": len(data["chunks"]),
                           "t_span": [float(u["t"][0]), float(u["t"][-1])],
                           "wall_s": data["meta"].get("wall_s")}
        np.savez_compressed(OUT / f"windows__{u['unit']}.npz",
                            **{k: np.array([r.get(k, np.nan) for r in rows], float)
                               for k in ("t1", "t2", "e", "ratio_dep", "ratio_cmd",
                                         "dz_truth", "L_truth", "dz_dep", "rho_cmd", "q")},
                            ok=np.array([bool(r.get("ok")) for r in rows]))

    sim_units = [u for u in by_unit if meta[u]["kind"] == "sim"]
    real_units = [u for u in by_unit if meta[u]["kind"] == "uzh"
                  and unit_population.get(u, "pilot") == "pilot"]
    climb = [u for u in sim_units if u not in ("i02_dev_000", "i02_dev_001")]
    controls = [u for u in sim_units if u in ("i02_dev_000", "i02_dev_001")]

    summaries = {u: M.clip_summary(by_unit[u]) for u in by_unit}
    # coverage: a clip/sequence is covered when >= half of its windows are reconstructable
    coverage = {}
    for u in by_unit:
        ok = [r for r in by_unit[u] if r.get("ok")]
        coverage[u] = {"n_windows": len(by_unit[u]), "n_ok": len(ok),
                       "window_frac": len(ok) / max(1, len(by_unit[u])),
                       "covered": bool(len(ok) >= 0.5 * len(by_unit[u]))}

    # per-clip defect reference on the vertical-motion clips
    defects = {u: M.defect_table(by_unit[u]) for u in climb}

    # the real error distribution, pooled over windows and per sequence
    real_windows = [r for u in real_units for r in by_unit[u] if r.get("ok")]
    real_abs = np.array([r["abs_e"] for r in real_windows])
    real_median = float(np.median(real_abs)) if len(real_abs) else float("nan")
    real_p90 = float(np.percentile(real_abs, 90)) if len(real_abs) else float("nan")
    per_seq = {u: {"n_windows": len([r for r in by_unit[u] if r.get("ok")]),
                   "median_abs_e": summaries[u].get("median_abs_e"),
                   "p90_abs_e": summaries[u].get("p90_abs_e"),
                   "median_abs_q": summaries[u].get("median_abs_q")} for u in real_units}

    # stop rule
    stricter = {}
    for u in climb:
        d = defects[u]
        vals = {f: d[f]["median_abs"] for f in ("scaled_0.5", "scaled_1.5")}
        vals = {f: v for f, v in vals.items() if v is not None}
        stricter[u] = (min(vals.values()) if vals else float("nan"), min(vals, key=vals.get)
                       if vals else None)
    n_exceed = int(sum(1 for u in climb
                       if np.isfinite(stricter[u][0]) and real_median > stricter[u][0]))
    cond_a = bool(len(climb) > 0 and n_exceed > len(climb) / 2)

    cov_climb = [coverage[u]["covered"] for u in climb]
    cov_real = [coverage[u]["covered"] for u in real_units]
    cond_b = bool(np.mean(cov_climb) < 0.5 or np.mean(cov_real) < 0.5)
    stop = bool(cond_a or cond_b)

    # descriptive: the same comparison for the largest (reversed) defect
    reversed_bar = {u: defects[u]["reversed"]["median_abs"] for u in climb}
    n_exceed_rev = int(sum(1 for u in climb
                           if reversed_bar[u] is not None
                           and np.isfinite(reversed_bar[u]) and real_median > reversed_bar[u]))

    boot = bootstrap_real(by_unit, real_units)

    out = {
        "prereg": "configs/v00_feasibility_prereg.yaml + amend1 + amend2",
        "n_units": len(by_unit),
        "coverage": coverage,
        "summaries": summaries,
        "real": {"pooled": {"n_windows": int(len(real_abs)), "median_abs_e": real_median,
                            "p90_abs_e": real_p90,
                            "median_abs_q": float(np.median(np.abs(
                                [r["q"] for r in real_windows if np.isfinite(r["q"])])))
                            if real_windows else None,
                            "median_ratio_cmd": float(np.median(
                                [r["ratio_cmd"] for r in real_windows])) if real_windows else None,
                            "median_dz_truth": float(np.median(
                                [r["dz_truth"] for r in real_windows])) if real_windows else None,
                            "median_h1_m": float(np.median(
                                [r.get("h1_m", np.nan) for r in real_windows
                                 if np.isfinite(r.get("h1_m", np.nan))]))
                            if real_windows else None},
                 "per_sequence": per_seq,
                 "bootstrap": boot},
        "sim": {"climb_units": climb, "control_units": controls,
                "coverage_climb": float(np.mean(cov_climb)) if cov_climb else None,
                "coverage_real": float(np.mean(cov_real)) if cov_real else None,
                "defect_reference": defects,
                "stricter_bar": {u: {"value": stricter[u][0], "family": stricter[u][1]}
                                 for u in climb},
                "n_clips_real_error_exceeds_stricter": n_exceed,
                "n_clips_real_error_exceeds_reversed": n_exceed_rev},
        "stop_rule": {"condition_a_fires": cond_a, "condition_b_fires": cond_b,
                      "STOP": stop,
                      "detail": {"n_climb": len(climb), "n_exceed_stricter": n_exceed,
                                 "coverage_climb": float(np.mean(cov_climb)) if cov_climb else None,
                                 "coverage_real": float(np.mean(cov_real)) if cov_real else None}},
        "meta": meta,
        "diags": {u: diags[u][:5] for u in diags},
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))

    # human-readable table
    lines = ["# Track V Part 0 feasibility — stored results", "",
             f"real pooled |log ratio error|: median {real_median:.3f}, p90 {real_p90:.3f} "
             f"({len(real_abs)} windows)", "",
             "| unit | kind | windows ok/total | median abs e | p90 abs e | median abs q |",
             "| --- | --- | --- | --- | --- | --- |"]
    for u in sorted(by_unit):
        s, cv = summaries[u], coverage[u]
        lines.append(f"| {u} | {meta[u]['kind']} | {cv['n_ok']}/{cv['n_windows']} | "
                     f"{_f(s.get('median_abs_e'))} | {_f(s.get('p90_abs_e'))} | "
                     f"{_f(s.get('median_abs_q'))} |")
    lines += ["", f"STOP = {stop} (A={cond_a}, B={cond_b}; "
                  f"{n_exceed}/{len(climb)} clips with real error above the stricter bar)"]
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    print(f"\nreal median |e| = {real_median:.3f}; STOP={stop} (A={cond_a}, B={cond_b}, "
          f"{n_exceed}/{len(climb)})")
    return 0


def _f(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def bootstrap_real(by_unit, real_units, n_boot: int = 5000):
    """Sequence-clustered bootstrap of the pooled median |e| (3 clusters: descriptive)."""
    rng = np.random.default_rng(1)
    per_unit = {u: np.array([r["abs_e"] for r in by_unit[u] if r.get("ok")], float)
                for u in real_units}
    per_unit = {u: v for u, v in per_unit.items() if len(v)}
    if len(per_unit) < 2:
        return {"n_clusters": len(per_unit), "note": "too few sequences"}
    units = list(per_unit)
    meds = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        vals = np.concatenate([per_unit[u] for u in pick])
        meds.append(float(np.median(vals)))
    return {"n_clusters": len(units), "n_boot": n_boot,
            "median_of_medians": float(np.median(meds)),
            "lo": float(np.percentile(meds, 2.5)), "hi": float(np.percentile(meds, 97.5)),
            "note": "fewer than 10 clusters: descriptive, not a 95 % interval"}


if __name__ == "__main__":
    raise SystemExit(main())
