"""Track N3, Part D: what the verdict is for clips with no anchor (fallbacks).

Pre-registered in `configs/n3a_anchor_prereg.yaml` (`downstream_contract`) and
`configs/n3b_anchored_tiers_prereg.yaml`: Part D is run iff the P1 anchor covers fewer than 60 % of
the dynamic clips, and the answer for a clip with no anchor is ABSTAIN -- no reader start, no level
start, no substitute. This script only counts that.

Run:  PYTHONPATH=. pixi run python experiments/n3d_fallbacks.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n3/n3d"
WINDOWS = ROOT / "results/n3/n3a/windows.csv"
N3B = ROOT / "results/n3/n3b/per_clip.csv"
MANIFEST = ROOT / "results/n2/inputs/manifest.csv"

CONDS = ("feasible", "tilt_removed", "mirrored_roll", "tilt_scaled", "partial", "thrust_limit")
TIERS = ("P1", "P2", "P3")
ARM = {"P1": "anchor1", "P2": "anchor1_P2", "P3": "anchor1_P3"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    man = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST)) if r["split"] == "dev"}
    win = {(r["unit"], r["tier"]): r for r in csv.DictReader(open(WINDOWS)) if r["kind"] == "sim"}
    b = {r["clip_id"]: r for r in csv.DictReader(open(N3B))}

    rows = []
    for cid in sorted(man):
        m = man[cid]
        dyn = float(m["peak_tilt_req_deg"]) >= 15.0
        rec = {"clip_id": cid, "family": m["family"], "mount": float(m["uptilt_deg"]),
               "condition": b[cid]["condition"], "dynamic": int(dyn),
               "peak_tilt_req_deg": float(m["peak_tilt_req_deg"])}
        for tier in TIERS:
            has = win[(cid, tier)]["has_anchor"] == "1"
            rec[f"anchor_{tier}"] = int(has)
            # the verdict for a clip with no anchor: ABSTAIN, by pre-registration
            rec[f"verdict_{tier}"] = "abstain" if not has else "anchored"
        rows.append(rec)
    with open(OUT / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def frac(sub, tier):
        n = len(sub)
        k = sum(r[f"anchor_{tier}"] for r in sub)
        return f"{k}/{n}", (k / n if n else float("nan"))

    summary = {"abstention_rule": "no anchor -> ABSTAIN; no reader start, no level start, no "
                                 "substitute of any kind (pre-registered)"}
    for tier in TIERS:
        block = {"all": frac(rows, tier),
                 "dynamic": frac([r for r in rows if r["dynamic"]], tier),
                 "static": frac([r for r in rows if not r["dynamic"]], tier),
                 "by_family": {}, "by_mount": {}, "by_condition": {},
                 "defects_unreachable": {}}
        for fam in sorted({r["family"] for r in rows}):
            block["by_family"][fam] = frac([r for r in rows if r["family"] == fam], tier)
        for mnt in sorted({r["mount"] for r in rows}):
            block["by_mount"][str(mnt)] = frac([r for r in rows if r["mount"] == mnt], tier)
        for cond in CONDS:
            block["by_condition"][cond] = frac([r for r in rows if r["condition"] == cond], tier)
            # a defect clip without an anchor can never be detected by the anchored arm
            sub = [r for r in rows if r["condition"] == cond]
            block["defects_unreachable"][cond] = f"{sum(1 for r in sub if not r[f'anchor_{tier}'])}" \
                                                f"/{len(sub)}"
        summary[tier] = block
    summary["note"] = ("`defects_unreachable` is the detection ceiling of the anchored arm on that "
                       "condition: a clip with no anchor abstains and cannot be flagged, so the "
                       "max attainable detection count is the complement.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({t: {"all": summary[t]["all"], "dynamic": summary[t]["dynamic"],
                          "forest_feasible": summary[t]["by_family"]["forest_ridge"],
                          "unreachable_removals": summary[t]["defects_unreachable"]["tilt_removed"],
                          "unreachable_mirrors": summary[t]["defects_unreachable"]["mirrored_roll"]}
                      for t in TIERS}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
