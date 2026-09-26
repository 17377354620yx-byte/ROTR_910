#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv2"
PYTHON="/home/yangx/miniconda3/envs/geo_v2/bin/python"

exec "$PYTHON" "$PROJECT_ROOT/tools/build_3dircadb_low_overlap.py" "$@"
