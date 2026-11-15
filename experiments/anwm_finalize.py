"""Post-process the generated clips: index, contact sheets, seed variance, R2 sync.

    python experiments/anwm_finalize.py           # local post-processing only
    python experiments/anwm_finalize.py --sync    # also push to R2 track-b/clips/

Assumes the clip tree is already at data/processed/anwm_clips (pull it first with
scripts/pull_volume.py awmv-b-results clips data/processed/anwm_clips).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(mod: str) -> None:
    print(f"== {mod}", flush=True)
    subprocess.run([sys.executable, "-m", mod], check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    run("experiments.anwm_index_clips")
    run("experiments.anwm_clip_sheets")
    run("experiments.anwm_variance")
    # the audit and the clip metadata travel with the clips so a scorer has the paths
    import shutil

    from src.utils.r2 import upload

    dst = ROOT / "data/processed/anwm_clips"
    for extra in ("results/anwm_path_audit/audit.json", "results/anwm_clips/specs.json",
                  "results/anwm_starts/starts.json", "results/anwm_starts/starts_coverage.json"):
        p = ROOT / extra
        if p.exists():
            shutil.copy(p, dst / p.name)
    if args.sync:
        print("== R2 sync", flush=True)
        subprocess.run([sys.executable, "-m", "experiments.anwm_index_clips", "--sync"],
                       check=True, cwd=ROOT)
        for extra in ("audit.json", "specs.json", "starts.json", "starts_coverage.json"):
            p = dst / extra
            if p.exists():
                upload(p, f"track-b/clips/{extra}")
                print("uploaded", extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
