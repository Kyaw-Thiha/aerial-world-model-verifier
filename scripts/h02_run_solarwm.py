"""Spawn the SolarWM batches on the deployed app (Track H2).

    set -a; . ./.env; set +a
    pixi run python scripts/h02_run_solarwm.py --batch probes
    pixi run python scripts/h02_run_solarwm.py --batch grid
    pixi run python scripts/h02_run_solarwm.py --batch seeds

Each batch is a separate spawned call so a bad sample cannot fail the whole pilot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h02_solarwm_inputs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=["probes", "grid", "seeds", "all"], required=True)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    import modal

    man = json.loads((INPUTS / "manifest.json").read_text())
    samples = man["samples"]
    groups = {
        "probes": [s for s in samples if s.startswith("probe_")],
        "grid": [s for s in samples if s.startswith("grid_")],
        "seeds": [s for s in samples if s.startswith("seed_")],
    }
    chosen = samples if args.batch == "all" else groups[args.batch]
    f = modal.Function.from_name("awmv-h2-solarwm", "infer")
    call = f.spawn(json.dumps(chosen), tag=args.batch, steps=args.steps)
    out = {"batch": args.batch, "n": len(chosen), "call_id": call.object_id,
           "steps": args.steps}
    path = ROOT / "results" / "h02_solarwm_spawns.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old[args.batch] = out
    path.write_text(json.dumps(old, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
