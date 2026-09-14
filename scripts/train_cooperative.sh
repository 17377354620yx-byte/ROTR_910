#!/usr/bin/env bash
set -euo pipefail
P2P_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P2P_PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export P2P_RUN_NAME="${P2P_RUN_NAME:-rtor_a3_cooperative_v1}"
exec conda run --no-capture-output -n geo_v2 \
  python experiments/geotransformer.p2p_liver/trainval.py \
  --architecture rtor_a3 --interaction_profile cooperative \
  --max_epoch 150 --lr 1e-4 --log_steps 10 "$@"
