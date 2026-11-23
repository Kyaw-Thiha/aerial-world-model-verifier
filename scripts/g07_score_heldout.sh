#!/usr/bin/env bash
# Score the 100 held-out clips at stride 1, once, exactly as the frozen recipe says.
# Idempotent per clip (a clip whose score file is on the volume is skipped), so an
# infrastructure interruption is resumed by re-running this script, never restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
for i in 0 1 2 3; do
  ids=$(cat "results/g07/heldout_chunk${i}.txt")
  echo "shard $i: $(echo "$ids" | tr ',' '\n' | wc -l) clips"
  pixi run modal run remote/g3_score.py::score_l40s \
      --clip-ids "$ids" --out "scores/stride1/heldout_c${i}" --batch 8 \
      > "/tmp/g07_score_c${i}.log" 2>&1 &
done
wait
for i in 0 1 2 3; do tail -2 "/tmp/g07_score_c${i}.log"; done
