#!/usr/bin/env bash
# Track I: final publish, full sha256 re-hash and docs regeneration once all 300 clips exist.
set -euo pipefail
cd "$(dirname "$0")/.."
p=$(ps -eo cmd | grep "i02_publish.py" | grep -v grep | wc -l)
if [ "$p" != "0" ]; then echo "another publish is running; stopping" >&2; exit 1; fi
PYTHONPATH=. pixi run python experiments/i02_publish.py --all
PYTHONPATH=. pixi run python experiments/i02_build_manifest.py --full-hash
PYTHONPATH=. pixi run python experiments/i02_build_manifest.py --docs
python - <<'EOF'
import csv, collections
rows = list(csv.DictReader(open("results/i02/manifest.csv")))
ids = [r["clip_id"] for r in rows]
print("rows", len(rows), "unique", len(set(ids)))
print(collections.Counter(r["split"] for r in rows))
print("unpublished", sum(not r["r2_video"] for r in rows),
      "bad sha", sum(1 for r in rows if not r["sha256_video"] or not r["sha256_gt"]))
print("frames", sum(int(r["n_frames"]) for r in rows))
EOF
