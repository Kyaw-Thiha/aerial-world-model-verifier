#!/usr/bin/env bash
# Score the 170 available dev clips at stride 1 on L40S, four balanced shards in parallel.
# Idempotent: a clip whose score file is on the volume is skipped, so a re-run only fills gaps.
set -euo pipefail
cd "$(dirname "$0")/.."
for i in 0 1 2 3; do
  ids=$(cat "results/g06/dev_ids_chunk${i}.txt")
  echo "shard $i: $(echo "$ids" | tr ',' '\n' | wc -l) clips"
  pixi run modal run remote/g3_score.py::score_l40s \
      --clip-ids "$ids" --out "scores/stride1/c${i}" --batch 8 \
      > "/tmp/g3_score_c${i}.log" 2>&1 &
done
wait
for i in 0 1 2 3; do
  echo "--- shard $i ---"
  tail -3 "/tmp/g3_score_c${i}.log"
done
