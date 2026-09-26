#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yangx/code/new_deform/ai_worker/RTORv2
DATA_ROOT=${IRCADB_DATA_ROOT:-/home/yangx/code/new_deform/RTORv6/output/ircadb1_vis20_30_noise2_rot25_90_seed20260822/dataset/pairs}
RESULT_ROOT=${IRCADB_RESULT_ROOT:-$ROOT/results/rtorv6_fixed_ircadb1_vis20_30_seed20260822}
OUTPUT_DIR=${IRCADB_FIGURE_ROOT:-$RESULT_ROOT/figures}

exec /home/yangx/miniconda3/envs/geo_v2/bin/python \
  "$ROOT/visualization/ircadb_method_comparison.py" \
  --data-root "$DATA_ROOT" --result-root "$RESULT_ROOT" \
  --output-dir "$OUTPUT_DIR" --colormap viridis "$@"
