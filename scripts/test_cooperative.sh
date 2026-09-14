#!/usr/bin/env bash
set -euo pipefail
P2P_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P2P_PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export P2P_RUN_NAME="${P2P_RUN_NAME:-rtor_a3_cooperative_v1}"
P2P_RESULT_DIR="output/geotransformer.p2p_liver.${P2P_RUN_NAME}"
P2P_CHECKPOINT="${P2P_SNAPSHOT:-${P2P_RESULT_DIR}/snapshots/best.pth.tar}"
if [[ ! -f "$P2P_CHECKPOINT" ]]; then
  echo "Missing trained checkpoint: $P2P_CHECKPOINT" >&2
  exit 1
fi
for P2P_NOISE in none 2 4; do
  conda run --no-capture-output -n geo_v2 \
    python experiments/geotransformer.p2p_liver/test.py \
    --architecture rtor_a3 --interaction_profile cooperative \
    --snapshot "$P2P_CHECKPOINT" --dataset in_silico --noise "$P2P_NOISE" \
    --output "$P2P_RESULT_DIR/in_silico_noise_${P2P_NOISE}.json"
done
conda run --no-capture-output -n geo_v2 \
  python experiments/geotransformer.p2p_liver/test.py \
  --architecture rtor_a3 --interaction_profile cooperative \
  --snapshot "$P2P_CHECKPOINT" --dataset in_vitro --noise none \
  --output "$P2P_RESULT_DIR/in_vitro.json"
