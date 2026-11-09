"""Nuisance ablations for the verdict test, in one pass.

Runs the scoring four times over the same per-frame estimates — both nuisances on, uptilt
off, lag off, neither — and reports, per condition: the clip-score median, the rejection
rate, and how much of the *violation* the nuisances absorb. The absorption number is the
point: a nuisance that lowers the real-clip score by more than it lowers the negative's
score is buying acceptance, not removing error.

    pixi run -e gpu python experiments/phase3a_ablations.py --geocalib results/phase3a_verdict/geocalib_p3a_v1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ["both", "no_uptilt", "no_lag", "none"]


def load_module():
    spec = importlib.util.spec_from_file_location("p3a_verdict",
                                                  ROOT / "experiments/phase3a_verdict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p3a_v1")
    ap.add_argument("--specs", default="data/processed/p3a")
    ap.add_argument("--geocalib", default="results/phase3a_verdict/geocalib_p3a_v1")
    ap.add_argument("--conditions", default="tilt_k0,tilt_k05,tilt_k15,mirror_roll,yaw_control")
    ap.add_argument("--out", default="results/phase3a_verdict")
    args = ap.parse_args()

    mod = load_module()
    info = mod.verify_prereg()
    prereg = info["doc"]
    spec_dir = ROOT / args.specs / args.tag
    clips = json.loads((spec_dir / "clips.json").read_text())["clips"]
    geo = ROOT / args.geocalib
    spec_paths = {p.name.replace(".npz", ""): p for p in sorted(spec_dir.glob("flight-*.npz"))}
    conds = ["identity"] + [c for c in args.conditions.split(",") if c]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / f"{run_id}_ablations"
    out.mkdir(parents=True, exist_ok=True)

    per_mode: dict = {}
    for mode in MODES:
        records: list[dict] = []
        for flight, path in spec_paths.items():
            rows_by_cond = {c: mod.load_jsonl(geo / f"{flight}__{c}.jsonl")
                            for c in conds if (geo / f"{flight}__{c}.jsonl").exists()}
            if not rows_by_cond:
                continue
            records.extend(mod.score_flight(path, rows_by_cond, clips, prereg, ablate=mode))
        ev = mod.evaluation(records, prereg)
        per_mode[mode] = {
            "n_records": len(records),
            "heldout_acceptance": ev.get("heldout_acceptance"),
            "family_auroc_pooled": ev.get("family_auroc_pooled"),
            "per_condition": {c: {"score_median": float(np.median(
                [r["score"] for r in records if r["condition"] == c and r["testable"]]))
                if any(r["condition"] == c and r["testable"] for r in records) else None,
                "rejection_rate": ev["families"].get(c, {}).get("rejection_rate"),
                "uptilt_offset_median_deg": (ev["diagnostics"]["per_condition"]
                                             .get(c, {}).get("uptilt_offset_median_deg")),
                "lag_median_ms": (ev["diagnostics"]["per_condition"]
                                  .get(c, {}).get("lag_median_ms"))}
                for c in conds},
        }
        print(f"{mode}: acceptance {ev.get('heldout_acceptance')}, "
              f"AUROC {ev.get('family_auroc_pooled')}", flush=True)

    both = per_mode["both"]["per_condition"]
    table = {}
    for c in conds:
        row = {}
        for mode in MODES:
            v = per_mode[mode]["per_condition"].get(c, {}).get("score_median")
            row[mode] = v
            if v is not None and both.get(c, {}).get("score_median"):
                row[f"{mode}_vs_both_deg"] = v - both[c]["score_median"]
        row["absorption_by_nuisances_deg"] = (
            per_mode["none"]["per_condition"].get(c, {}).get("score_median") - row["both"]
            if row["both"] is not None and per_mode["none"]["per_condition"].get(c, {}).get("score_median") is not None
            else None)
        table[c] = row

    payload = {"run_id": out.name, "git": git_hash(),
               "prereg": {k: info[k] for k in ("path", "sha256", "git_hash")},
               "modes": per_mode, "absorption_table_deg": table}
    (out / "ablations.json").write_text(json.dumps(payload, indent=1, default=float))
    print(json.dumps(table, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
