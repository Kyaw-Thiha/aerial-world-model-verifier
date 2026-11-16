"""Spawn a LingBot-World grid shard on a deployed app (Track H).

    set -a; . ./.env; set +a
    pixi run python scripts/h01_lingbot_grid.py --app awmv-h-lingbot  --paths turn5,turn10  --tag a
    pixi run python scripts/h01_lingbot_grid.py --app awmv-h-lingbot-b --paths turn20,slalom10 --tag b

Each shard writes to its own Track H volume (`awmv-h-lingbot-results` / `-b`); the analysis
concatenates the reader rows. 20 sampling steps per the pre-registration amendment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/h01_lingbot_inputs"
PROMPTS = {
    "turn5": "A quadrotor drone performs a gentle constant-speed banked turn over a city, "
             "the horizon tilting slightly.",
    "turn10": "A quadrotor drone performs a constant-speed banked turn over a city, the "
              "horizon clearly tilted.",
    "turn20": "A quadrotor drone performs a steep constant-speed banked turn over a city, "
              "the horizon strongly tilted.",
    "slalom10": "A quadrotor drone flies a slalom, banking left then right between buildings "
                "at constant speed.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--frame-num", type=int, default=49)
    args = ap.parse_args()

    import modal

    man = json.loads((INPUTS / "manifest.json").read_text())
    keep = set(args.paths.split(","))
    jobs = []
    for g in man["grid"]:
        if g["path"] not in keep:
            continue
        jobs.append({"id": f"{g['traj']}_{g['frame']:03d}__{g['path']}__{g['arm']}",
                     "pose_dir": g["dir"].replace("data/h01_lingbot_inputs/", ""),
                     "image": g["image"], "path": g["path"], "arm": g["arm"],
                     "traj": g["traj"], "frame": g["frame"],
                     "prompt": PROMPTS[g["path"]], "seed": 0})
    f = modal.Function.from_name(args.app, "generate")
    call = f.spawn(json.dumps(jobs), sampling_steps=args.steps, frame_num=args.frame_num)
    out = {"app": args.app, "tag": args.tag, "call_id": call.object_id,
           "n_clips": len(jobs), "paths": sorted(keep), "steps": args.steps,
           "frame_num": args.frame_num}
    (ROOT / "results/h01_lingbot" / f"spawn_{args.tag}.json").parent.mkdir(
        parents=True, exist_ok=True)
    (ROOT / "results/h01_lingbot" / f"spawn_{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
