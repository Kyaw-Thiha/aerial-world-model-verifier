"""Track V, V1: score the commanded-vertical-motion check (Parts 1-3).

Rules: ``configs/v10_vertical_prereg.yaml`` + amendment 1. Nothing here chooses a threshold, a
window, a population or an edit; every number is one the pre-registration names.

Order: Part 2 first, because it produces the thresholds (leave-one-sequence-out conformal
quantile of the window-level |e| over the 14 UZH development sequences, alpha = 0.05, with 0.10
as a pre-registered sensitivity). Then Part 1 (the defect renders and the two control arms) is
scored against those thresholds at the 3 s primary window and the 5 s secondary window. Part 3
(generated clips) is descriptive with the scale-free q only.

Usage
-----
PYTHONPATH=. python experiments/v10_score.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.vertical import metrics as M
from src.vertical import local_windows as LW
from src.vertical import recon_io as RIO

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "data/v/units.json"
RECON = ROOT / "data/v/recon"
OUT = ROOT / "results/v/v10"
ALPHA = 0.05
ALPHA_SENS = 0.10
ARMS = (3.0, 5.0)
CONTROL_FAMILIES = ("null", "yaw")


def _f(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float) and not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


def rows_for(unit: dict) -> dict[float, list[dict]]:
    """Rows per window arm. Generated clips get one whole-clip row under the key 0.0."""
    data = RIO.load_unit(RECON, unit["unit"])
    if unit["kind"] in ("anwm", "solarwm"):
        return {0.0: [RIO.whole_clip_row(data, np.asarray(unit["truth"]["pos"], float),
                                        kind=unit["truth"].get("statistic", "net"))]}
    pos = np.asarray(unit["truth"]["pos"], float)
    t = np.asarray(unit["truth"]["t"], float)
    mode = "sim" if unit["kind"] in ("sim", "defect") else "real"
    out = {3.0: RIO.unit_rows(data, t, pos, mode=mode)[0]}
    out[5.0] = LW.rows_local(data, t, pos, mode=mode, window_s=5.0)
    # predicted effect of a vertical edit, from rendered vs commanded altitude
    if unit["kind"] == "defect" and "rendered_pos" in unit["truth"]:
        rz = np.asarray(unit["truth"]["rendered_pos"], float)[:, 2]
        cz = pos[:, 2]
        for rows in out.values():
            for r in rows:
                if not r.get("ok"):
                    continue
                zc1, zc2 = float(np.interp(r["t1"], t, cz)), float(np.interp(r["t2"], t, cz))
                zr1, zr2 = float(np.interp(r["t1"], t, rz)), float(np.interp(r["t2"], t, rz))
                r["ratio_rendered"] = zr2 / zr1
                r["predicted_e"] = float(np.log(zr2 / zr1) - np.log(zc2 / zc1))
    return out


def coverage(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    return {"n_windows": len(rows), "n_ok": len(ok),
            "window_frac": len(ok) / max(1, len(rows)),
            "covered": bool(len(ok) >= 0.5 * len(rows))}


def clip_stat(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {"ok": False}
    e = np.array([r["e"] for r in ok])
    q = np.array([r["q"] for r in ok if np.isfinite(r["q"])])
    pred = np.array([r["predicted_e"] for r in ok if "predicted_e" in r])
    return {"ok": True, "max_abs_e": float(np.max(np.abs(e))),
            "median_e": float(np.median(e)), "n_ok": len(ok),
            "median_abs_e": float(np.median(np.abs(e))),
            "max_abs_q": float(np.max(np.abs(q))) if len(q) else None,
            "median_dz_truth": float(np.median([r["dz_truth"] for r in ok]))
            if "dz_truth" in ok[0] else None,
            "measured_minus_predicted_e": (float(np.median(e - pred))
                                           if len(pred) == len(e) else None)}


def lofo(rows_by_seq: dict[str, list[dict]], alpha: float) -> dict:
    seqs = sorted(rows_by_seq)
    pooled = {s: np.array([r["abs_e"] for r in rows_by_seq[s] if r.get("ok")], float)
              for s in seqs}
    out = {}
    for s in seqs:
        others = np.concatenate([pooled[o] for o in seqs if o != s] or [np.zeros(0)])
        if len(others) < 20:
            out[s] = {"tau": None, "n_other": int(len(others))}
            continue
        tau = float(np.quantile(others, 1.0 - alpha))
        own = pooled[s]
        out[s] = {"tau": tau, "n_other": int(len(others)), "n_own": int(len(own)),
                  "seq_flagged": bool(len(own) and own.max() > tau),
                  "window_frac_flagged": float(np.mean(own > tau)) if len(own) else None,
                  "own_max": float(own.max()) if len(own) else None,
                  "own_median": float(np.median(own)) if len(own) else None,
                  "own_p90": float(np.percentile(own, 90)) if len(own) else None}
    return out


def detect(rows: list[dict], taus: list[float]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {"n": 0, "flagged_any": False, "flagged_all": False, "flagged_frac": None}
    mx = float(np.max(np.abs([r["e"] for r in ok])))
    flags = [bool(mx > t) for t in taus]
    return {"max_abs_e": mx, "flagged_folds": flags, "flagged_frac": float(np.mean(flags)),
            "flagged_any": bool(any(flags)), "flagged_all": bool(all(flags))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    units = {u["unit"]: u for u in json.loads(UNITS.read_text())["units"]}
    have = {u for u in units if RIO.complete(RECON, u)}
    missing = sorted(set(units) - have)
    if missing:
        print(f"NOTE: {len(missing)} units without recon: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")

    rows_all = {u: rows_for(units[u]) for u in sorted(have)}
    real_units = [u for u in have if units[u]["kind"] == "uzh"]
    sim_units = [u for u in have if units[u]["kind"] == "sim"]
    defect_units = [u for u in have if units[u]["kind"] == "defect"]
    gen_units = [u for u in have if units[u]["kind"] in ("anwm", "solarwm")]

    out = {"prereg": "configs/v10_vertical_prereg.yaml + amend1",
           "n_units_with_recon": len(have), "missing_recon": missing, "arms": {}}

    for arm in ARMS:
        real_rows = {u: rows_all[u][arm] for u in real_units}
        lofo_primary = lofo(real_rows, ALPHA)
        lofo_sens = lofo(real_rows, ALPHA_SENS)
        taus = [v["tau"] for v in lofo_primary.values() if v["tau"] is not None]
        tau_med = float(np.median(taus)) if taus else float("nan")
        per_seq = {}
        for u in real_units:
            ok = [r for r in real_rows[u] if r.get("ok")]
            cov = coverage(real_rows[u])
            if not ok:
                per_seq[u] = {"n_windows": 0, "covered": cov["covered"]}
                continue
            e = np.array([r["abs_e"] for r in ok])
            q = np.array([r["q"] for r in ok if np.isfinite(r["q"])])
            per_seq[u] = {"kind": units[u].get("population", "pilot"), "n_windows": len(ok),
                          "coverage": cov["window_frac"], "covered": cov["covered"],
                          "median_abs_e": float(np.median(e)),
                          "p90_abs_e": float(np.percentile(e, 90)),
                          "max_abs_e": float(e.max()),
                          "median_abs_q": float(np.median(np.abs(q))) if len(q) else None,
                          "median_dz_truth": float(np.median([r["dz_truth"] for r in ok])),
                          "median_ratio_cmd": float(np.median([r["ratio_cmd"] for r in ok]))}
        part2 = {"n_sequences": len(real_units), "alpha": ALPHA, "lofo": lofo_primary,
                 "lofo_sensitivity": lofo_sens,
                 "n_sequences_flagged": int(sum(1 for v in lofo_primary.values()
                                                if v.get("seq_flagged"))),
                 "sequences_covered": int(sum(coverage(real_rows[u])["covered"]
                                              for u in real_units)),
                 "window_fp_rate": float(np.nanmean(
                     [v["window_frac_flagged"] for v in lofo_primary.values()
                      if v.get("window_frac_flagged") is not None])) if real_units else None,
                 "per_sequence": per_seq, "median_tau": tau_med}

        part1 = {"arms": {}, "controls": {}, "sim_false_reject": {}, "calibration": {},
                 "tau_per_fold": taus, "tau_median": tau_med}
        if defect_units and taus:
            arms = defaultdict(list)
            for u in defect_units:
                unit = units[u]
                rows = rows_all[u][arm]
                cs = clip_stat(rows)
                cs.update({"edit_id": u, "family": unit.get("edit_family"),
                           "clip_id": unit.get("clip_id"),
                           "uptilt_deg": unit.get("uptilt_deg"),
                           "coverage": coverage(rows)["window_frac"]})
                cs["det"] = detect(rows, taus)
                if unit.get("edit_family") in CONTROL_FAMILIES:
                    part1["controls"][u] = cs
                else:
                    arms[unit.get("edit_family")].append(cs)
                    pr = [r["predicted_e"] for r in rows if r.get("ok") and "predicted_e" in r]
                    me = [r["e"] for r in rows if r.get("ok") and "predicted_e" in r]
                    if pr:
                        part1["calibration"].setdefault(unit.get("edit_family"), []).append(
                            {"measured_minus_predicted": float(np.median(np.array(me)
                                                                        - np.array(pr))),
                             "sign_agrees": int(np.sum(np.sign(me) == np.sign(pr))),
                             "n": len(pr), "edit_id": u})
            for fam, clips in arms.items():
                part1["arms"][fam] = {
                    "n_clips": len(clips),
                    "n_flagged_max": int(sum(1 for c in clips if c["det"]["flagged_any"])),
                    "n_flagged_all_folds": int(sum(1 for c in clips
                                                   if c["det"]["flagged_all"])),
                    "median_max_abs_e": float(np.median([c["max_abs_e"] for c in clips])),
                    "median_e": float(np.median([c["median_e"] for c in clips])),
                    "per_clip": clips}
            base_ids = {units[u].get("clip_id") for u in defect_units}
            for u in sim_units:
                if units[u]["unit"] in base_ids or u in ("i02_dev_000", "i02_dev_001"):
                    rows = rows_all[u][arm]
                    part1["sim_false_reject"][u] = {**clip_stat(rows),
                                                    "det": detect(rows, taus),
                                                    "coverage": coverage(rows)["window_frac"]}

        part3 = {}
        for u in gen_units:
            r = dict(rows_all[u][0.0][0])
            r.update({"family": units[u]["kind"], "arm": units[u]["truth"].get("arm"),
                      "base_path": units[u]["truth"].get("base_path"),
                      "maneuver": units[u]["truth"].get("maneuver"),
                      "statistic": units[u]["truth"].get("statistic", "net")})
            part3[u] = r

        out["arms"][f"{arm:g}s"] = {"part2_real_false_positives": part2,
                                    "part1_sim_detection": part1,
                                    "part3_generated": part3}
        print(f"[{arm:g}s] real {len(real_units)} seq, "
              f"{part2['n_sequences_flagged']} flagged, tau {tau_med:.3f}, "
              f"window FP {_f(part2['window_fp_rate'])}")
        for fam, a in part1.get("arms", {}).items():
            print(f"   {fam}: {a['n_flagged_max']}/{a['n_clips']} flagged, "
                  f"median max|e| {a['median_max_abs_e']:.3f}, median e {a['median_e']:.3f}")
        for u, c in part1.get("controls", {}).items():
            print(f"   control {u}: flagged={c['det']['flagged_any']} max|e| "
                  f"{_f(c['max_abs_e'])}")
        nfr = part1.get("sim_false_reject", {})
        print(f"   sim base/controls flagged: "
              f"{sum(1 for c in nfr.values() if c['det']['flagged_any'])}/{len(nfr)}")

    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    write_table(out)
    return 0


def write_table(out: dict) -> None:
    lines = ["# Track V1 — stored results", ""]
    for arm, a in out["arms"].items():
        p2, p1, p3 = (a["part2_real_false_positives"], a["part1_sim_detection"],
                      a["part3_generated"])
        lines += [f"## Window {arm}", "",
                  f"Real (LOFO, alpha={p2['alpha']}): {p2['n_sequences']} sequences, "
                  f"{p2['n_sequences_flagged']} flagged, median tau {_f(p2['median_tau'])}, "
                  f"window-level false-reject {_f(p2['window_fp_rate'])}, "
                  f"covered {p2['sequences_covered']}/{p2['n_sequences']}", "",
                  "| sequence | windows | coverage | median abs e | p90 | max | tau | flagged |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for u, s in sorted(p2["per_sequence"].items()):
            lf = p2["lofo"].get(u, {})
            lines.append(f"| {u} | {s.get('n_windows', 0)} | {_f(s.get('coverage'))} | "
                         f"{_f(s.get('median_abs_e'))} | {_f(s.get('p90_abs_e'))} | "
                         f"{_f(s.get('max_abs_e'))} | {_f(lf.get('tau'))} | "
                         f"{lf.get('seq_flagged')} |")
        if p1.get("arms"):
            lines += ["", "| defect arm | flagged | n | median max abs e | median e |",
                      "| --- | --- | --- | --- | --- |"]
            for fam, ar in p1["arms"].items():
                lines.append(f"| {fam} | {ar['n_flagged_max']} | {ar['n_clips']} | "
                             f"{_f(ar['median_max_abs_e'])} | {_f(ar['median_e'])} |")
            lines += ["", "| control | flagged | max abs e |", "| --- | --- | --- |"]
            for u, c in p1["controls"].items():
                lines.append(f"| {u} | {c['det']['flagged_any']} | {_f(c['max_abs_e'])} |")
            lines += ["", "| sim base/control | flagged | median abs e |",
                      "| --- | --- | --- |"]
            for u, c in p1.get("sim_false_reject", {}).items():
                lines.append(f"| {u} | {c['det']['flagged_any']} | "
                             f"{_f(c.get('median_abs_e'))} |")
        if p3:
            lines += ["", "| generated clip | rho_cmd | rho_dep | q | ok |",
                      "| --- | --- | --- | --- | --- |"]
            for u, r in sorted(p3.items()):
                lines.append(f"| {u} | {_f(r.get('rho_cmd'))} | {_f(r.get('rho_dep'))} | "
                             f"{_f(r.get('q'))} | {r.get('ok')} |")
        lines.append("")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
