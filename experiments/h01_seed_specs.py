"""H01, Part 1: build the ANWM seed-variance spec set (pre-registered in configs/h01_pilot_prereg.yaml).

Twenty Regime C designed paths, seeds 1-4 each (seed 0 already exists in
``results/anwm_clips/specs.json``). The path list is fixed by the pre-registration:

* the start-0 clip of each of the 15 unique (maneuver, designed_tilt, policy) cells;
* plus the start-1 clip of cruise, turn-20-fixed, slalom-10-fixed, accel-20-course and
  climb-20-fixed (two start states on the sharpest cells).

    pixi run python experiments/h01_seed_specs.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/anwm_clips/specs.json"
OUT = ROOT / "results/h01_seed_specs"

EXTRA_START1 = ["C_cruise_ctl_course_1", "C_turn_20_fixed_1", "C_slalom_10_fixed_1",
                "C_accel_brake_20_course_1", "C_climb_turn_20_fixed_1"]
SEEDS = (1, 2, 3, 4)


def select_paths(specs: list[dict]) -> list[dict]:
    c = [s for s in specs if s["regime"] == "C" and not re.search(r"_s\d+$", s["id"])]
    cells: dict[tuple, list[dict]] = {}
    for s in c:
        key = (s["meta"]["maneuver"], s["designed_tilt_deg"], s["meta"]["policy"])
        cells.setdefault(key, []).append(s)
    chosen = []
    for key in sorted(cells, key=lambda k: (str(k[0]), -1 if k[1] is None else k[1], str(k[2]))):
        starts = sorted(cells[key], key=lambda s: s["id"])
        chosen.append(starts[0])
    by_id = {s["id"]: s for s in c}
    for cid in EXTRA_START1:
        chosen.append(by_id[cid])
    ids = [s["id"] for s in chosen]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate path selected")
    if len(chosen) != 20:
        raise RuntimeError(f"expected 20 paths, got {len(chosen)}")
    return chosen


def main() -> int:
    specs = json.loads(SRC.read_text())
    paths = select_paths(specs)
    out_specs = []
    for p in paths:
        for seed in SEEDS:
            s = json.loads(json.dumps(p))
            s["id"] = f"{p['id']}_s{seed}"
            s["seed"] = seed
            s["meta"] = dict(s.get("meta", {}), seed_source="h01_seed_variance",
                             base_path=p["id"])
            out_specs.append(s)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "specs.json").write_text(json.dumps(out_specs))
    (OUT / "paths.json").write_text(json.dumps(
        [{"base_path": p["id"], "maneuver": p["meta"]["maneuver"],
          "policy": p["meta"]["policy"], "designed_tilt_deg": p["designed_tilt_deg"],
          "start_traj": p["start_traj"], "start_time": p["start_time"]} for p in paths],
        indent=1))
    print(f"{len(out_specs)} specs over {len(paths)} paths, seeds {SEEDS}")
    print([p["id"] for p in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
