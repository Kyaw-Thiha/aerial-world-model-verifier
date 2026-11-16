"""Spawn the H01 ANWM seed-generation job and print its Modal call id.

The job is 80 clips (~5 GPU-hours at the measured 235 s/clip), which is longer than a
single interactive call, so it is spawned and collected later:

    set -a; . ./.env; set +a
    pixi run python scripts/h01_spawn_seedgen.py
    pixi run python scripts/h01_spawn_seedgen.py --collect <call-id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/h01_seed_specs/spawn.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="results/h01_seed_specs/specs.json")
    ap.add_argument("--out-root", default="/r/clips")
    ap.add_argument("--collect", default="")
    args = ap.parse_args()

    import subprocess
    import sys

    import modal

    sys.path.insert(0, str(ROOT / "remote"))
    import h01_anwm

    app = h01_anwm.app
    if args.collect:
        with app.run():
            call = modal.FunctionCall.from_id(args.collect)
            res = call.get(timeout=0)
        (ROOT / "results/h01_seed_specs/summary_seedgen.json").write_text(
            json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "clips"}, indent=1))
        return 0

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    specs = json.loads((ROOT / args.specs).read_text())
    for s in specs:
        s["code_commit"] = commit
    with app.run():
        call = h01_anwm.seeds.spawn(json.dumps(specs), out_root=args.out_root)
    STATE.write_text(json.dumps({"call_id": call.object_id, "n_specs": len(specs),
                                 "code_commit": commit, "out_root": args.out_root}, indent=1))
    print("spawned", call.object_id, "specs", len(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
