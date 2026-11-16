#!/usr/bin/env bash
# H01: pull the pilot outputs off the Track H volumes and run the depicted reader.
#
#   bash scripts/h01_collect.sh anwm      # seed clips
#   bash scripts/h01_collect.sh lingbot   # LingBot arms
#   bash scripts/h01_collect.sh wan       # Wan clips
#   bash scripts/h01_collect.sh gate      # pull the reader rows
#
# Requires the repository env (set -a; . ./.env; set +a) and pixi.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
what="${1:?usage: h01_collect.sh anwm|lingbot|wan|gate}"

case "$what" in
  anwm)
    pixi run python scripts/pull_volume.py awmv-h-anwm-results clips data/processed/h01_seed_clips
    ;;
  lingbot)
    pixi run python scripts/pull_volume.py awmv-h-lingbot-results lingbot data/h01_lingbot_out
    ;;
  wan)
    pixi run python scripts/pull_volume.py awmv-h-wan-results wan data/h01_wan_out
    ;;
  gate)
    pixi run python scripts/pull_volume.py awmv-h-gate-results h01_depicted results/h01_depicted_all
    cp -r results/h01_depicted_all/anwm-seeds results/h01_seed_depicted 2>/dev/null || true
    cp -r results/h01_depicted_all/lingbot results/h01_lingbot_depicted 2>/dev/null || true
    cp -r results/h01_depicted_all/wan results/h01_wan_depicted 2>/dev/null || true
    ;;
  *)
    echo "unknown target: $what" >&2; exit 2
    ;;
esac
