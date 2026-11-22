"""Render the L02 report's markdown tables from the stored results.

    PYTHONPATH=. python experiments/l02_tables.py > /tmp/l02_tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FOLDS = Path("results/l02/folds")
BASE = Path("results/l02/g03_baselines.json")
OUT = Path("results/l02")
HELD = ["flight-13p-trackRATM", "flight-07a-lemniscate"]


def angle(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.degrees(np.arccos(np.clip((a * b).sum(1) / n, -1, 1)))


def stats(e):
    e = np.sort(np.asarray(e))
    k = max(1, int(round(0.9 * len(e))))
    return {"n": int(len(e)), "median": float(np.median(e)),
            "trimmed_rmse": float(np.sqrt(np.mean(e[:k] ** 2))),
            "gt10": float(np.mean(e > 10))}


def main() -> int:
    base = json.loads(BASE.read_text())
    print("### Held-out real flights (both models trained without them)\n")
    print("| reader | 13p median | 13p trimmed RMSE | 07a median | 07a trimmed RMSE | pooled trimmed RMSE |")
    print("| --- | --- | --- | --- | --- | --- |")
    print(f"| frozen | {base[HELD[0]]['frozen']['median']:.2f} | "
          f"{base[HELD[0]]['frozen']['trimmed_rmse']:.2f} | "
          f"{base[HELD[1]]['frozen']['median']:.2f} | "
          f"{base[HELD[1]]['frozen']['trimmed_rmse']:.2f} | 14.75 |")
    print(f"| G03 fine-tune | {base[HELD[0]]['g03_ft']['median']:.2f} | "
          f"{base[HELD[0]]['g03_ft']['trimmed_rmse']:.2f} | "
          f"{base[HELD[1]]['g03_ft']['median']:.2f} | "
          f"{base[HELD[1]]['g03_ft']['trimmed_rmse']:.2f} | 8.29 |")
    for arm in ("real", "sim", "mixed"):
        p = FOLDS / f"{arm}__none.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        rows = rec.get("held_real") or rec.get("all_real") or []
        per, all_e = {}, []
        for fl in HELD:
            sel = [r for r in rows if r.get("flight") == fl and r.get("up_cam_ft")]
            e = angle(np.array([r["up_cam"] for r in sel]), np.array([r["up_cam_ft"] for r in sel]))
            per[fl] = stats(e)
            all_e.extend(e.tolist())
        s = stats(all_e)
        print(f"| arm {arm} | {per[HELD[0]]['median']:.2f} | {per[HELD[0]]['trimmed_rmse']:.2f} | "
              f"{per[HELD[1]]['median']:.2f} | {per[HELD[1]]['trimmed_rmse']:.2f} | "
              f"{s['trimmed_rmse']:.2f} |")

    print("\n### Development folds (per-flight trimmed RMSE, leave-one-flight-out)\n")
    print("| flight | G03 | frozen | arm a real |")
    print("| --- | --- | --- | --- |")
    for fl in base:
        p = FOLDS / f"real__{fl}.json"
        a = ""
        if p.exists():
            rec = json.loads(p.read_text())
            rows = rec["held_out_real"]
            e = angle(np.array([r["up_cam"] for r in rows]), np.array([r["up_cam_ft"] for r in rows]))
            a = f"{stats(e)['trimmed_rmse']:.2f}"
        print(f"| {fl} | {base[fl]['g03_ft']['trimmed_rmse']:.2f} | "
              f"{base[fl]['frozen']['trimmed_rmse']:.2f} | {a} |")

    print("\n### Simulated scene-family bias spread (out-of-family predictions, degrees)\n")
    s2 = json.loads((OUT / "summary.json").read_text())["criteria"] if (OUT / "summary.json").exists() else {}
    if s2:
        print(f"Frozen reader spread: **{s2.get('s2_frozen_spread', float('nan')):.2f}** deg "
              f"({', '.join(f'{k} {v:.2f}' for k, v in s2.get('s2_frozen', {}).items())})")
        for arm in ("sim", "mixed"):
            key = f"s2_{arm}"
            if key in s2:
                v = s2[key]
                print(f"- arm {arm}: spread **{v['spread']:.2f}** deg "
                      f"({', '.join(f'{k} {x:.2f}' for k, x in v['per_family'].items())}), "
                      f"reduction {100*v['reduction_vs_frozen']:.0f} %")

    d = OUT / "downstream.json"
    if d.exists():
        print("\n### Downstream coherence tier on the gate-kept real clips\n")
        print("| reader | real FRR | removal | mirror | yaw | null |")
        print("| --- | --- | --- | --- | --- | --- |")
        for model, res in json.loads(d.read_text())["models"].items():
            r = res["rates"]

            def cell(k):
                v = r.get(k, {}).get("lofo")
                return f"{v['k']}/{v['n']}" if v else "—"
            print(f"| {model} | {cell('real_null')} | {cell('tilt_k0')} | {cell('mirror_roll')} | "
                  f"{cell('yaw_control')} | {cell('null_edit')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
