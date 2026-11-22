#!/usr/bin/env bash
# Track I: render the remaining clips of the frozen 300 (batch 2), publish, verify, docs.
#
#   scripts/i02_finish.sh heldout        # the 97 held-out clips
#   scripts/i02_finish.sh dev            # the 30 missing development clips
#
# One GPU container at a time. The render app skips clips already on the volume, so a
# re-run resumes rather than repeating.
set -euo pipefail
cd "$(dirname "$0")/.."
which=$1
case "$which" in
  dev)     missing=30 ;;
  heldout) missing=200 ;;      # dev are all present by then; this selects the 97 held-out
  *) echo "usage: $0 dev|heldout" >&2; exit 2 ;;
esac
pixi run modal run remote/i_dataset.py::render_clips \
    --missing "$missing" --samples 16 --exposure -3 --commit 955c0b3 \
    > "/tmp/i02_${which}.log" 2>&1
# publish writes the full manifest row (from the volume's manifest_row.json) and adds the R2
# fields in one pass; do NOT run `--pull` after it, because a batch row carries no R2 keys and
# merging it would erase them.
PYTHONPATH=. pixi run python experiments/i02_publish.py --all
PYTHONPATH=. pixi run python experiments/i02_build_manifest.py --verify
PYTHONPATH=. pixi run python experiments/i02_build_manifest.py --docs
python - <<'EOF'
import csv
rows = list(csv.DictReader(open("results/i02/manifest.csv")))
print("manifest rows:", len(rows),
      "| dev:", sum(r["split"] == "dev" for r in rows),
      "| heldout:", sum(r["split"] == "heldout" for r in rows),
      "| unpublished:", sum(not r["r2_video"] for r in rows))
EOF
