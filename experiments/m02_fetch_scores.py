"""Pull Track M's Gate B score JSONLs from the Modal volume into `data/m02_scores/<tag>/`.

    pixi run python experiments/m02_fetch_scores.py v2 v3
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/m02_scores"


def main() -> None:
    tags = sys.argv[1:] or ["v2", "v3"]
    for tag in tags:
        dst = OUT / tag
        dst.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["modal", "volume", "ls", "awmv-m-data",
                               f"/m02/scores/{tag}"], capture_output=True, text=True)
        names = [Path(w).name for w in proc.stdout.replace("│", " ").split()
                 if w.endswith(".jsonl")]
        for name in sorted(set(names)):
            target = dst / Path(name).name
            subprocess.run(["modal", "volume", "get", "awmv-m-data",
                            f"/m02/scores/{tag}/{name}", str(target), "--force"],
                           check=True, capture_output=True)
            print(f"{tag}/{name}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
