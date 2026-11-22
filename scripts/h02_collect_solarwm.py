"""Block until one SolarWM batch finishes, then print its summary.

    pixi run python scripts/h02_collect_solarwm.py --batch probes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    doc = json.loads((ROOT / "results/h02_solarwm_spawns.json").read_text())[args.batch]
    res = modal.FunctionCall.from_id(doc["call_id"]).get(timeout=4 * 3600)
    (ROOT / "results" / f"h02_solarwm_{args.batch}_summary.json").write_text(
        json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
    print(json.dumps(res.get("clips", {}), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
