#!/usr/bin/env bash
set -euo pipefail
DEPOLL_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEPOLL_PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
DEPOLL_PYTHON="${DEPOLL_PYTHON:-/home/yangx/miniconda3/envs/geo_v2/bin/python}"
DEPOLL_CHECKPOINT="${DEPOLL_SNAPSHOT:-output/geotransformer.p2p_liver.rtor_a3_cooperative_v1/snapshots/best.pth.tar}"
DEPOLL_OUTPUT="${DEPOLL_OUTPUT:-output/depoll_$(date +%Y%m%d_%H%M%S)}"
"$DEPOLL_PYTHON" experiments/geotransformer.p2p_liver/test_depoll.py \
  --root "${DEPOLL_ROOT:-/home/yangx/code/new_deform/DEPOLL}" \
  --snapshot "$DEPOLL_CHECKPOINT" --protocol both --output "$DEPOLL_OUTPUT" "$@"
