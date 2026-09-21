#!/usr/bin/env bash
set -euo pipefail

P2P_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P2P_PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
P2P_CONDA_ENV="${P2P_CONDA_ENV:-geo_py310}"
P2P_CHECKPOINT_NAME="${P2P_CHECKPOINT_NAME:-epoch-150.pth.tar}"

profiles=(
  abl1_no_proposal
  abl2_no_soft_weight
  abl3_no_poincare
  abl4_no_a3_geometry
  abl5_no_rtor_descriptor
)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p2p_ablation.sh train|test|all [PROFILE|all] [TRAIN_ARGS...]

Examples:
  bash scripts/run_p2p_ablation.sh train abl1_no_proposal
  bash scripts/run_p2p_ablation.sh test abl1_no_proposal
  bash scripts/run_p2p_ablation.sh all all
  bash scripts/run_p2p_ablation.sh train abl3_no_poincare --resume
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

action="$1"
selector="${2:-all}"
if [[ "$action" != "train" && "$action" != "test" && "$action" != "all" ]]; then
  usage >&2
  exit 2
fi
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi
train_args=("$@")

selected_profiles=()
if [[ "$selector" == "all" ]]; then
  selected_profiles=("${profiles[@]}")
else
  for profile in "${profiles[@]}"; do
    if [[ "$selector" == "$profile" ]]; then
      selected_profiles+=("$profile")
    fi
  done
  if [[ ${#selected_profiles[@]} -eq 0 ]]; then
    echo "Unknown ablation profile: $selector" >&2
    usage >&2
    exit 2
  fi
fi

train_profile() {
  local profile="$1"
  local run_name="rtor_a3_cooperative_${profile}_seed7351"
  echo "[train] profile=${profile} run=${run_name} gpu=${CUDA_VISIBLE_DEVICES}"
  P2P_RUN_NAME="$run_name" conda run --no-capture-output -n "$P2P_CONDA_ENV" \
    python experiments/geotransformer.p2p_liver/trainval.py \
    --architecture rtor_a3 \
    --interaction_profile cooperative \
    --ablation_profile "$profile" \
    --max_epoch 150 \
    --lr 1e-4 \
    --log_steps 10 \
    "${train_args[@]}"
}

test_profile() {
  local profile="$1"
  local run_name="rtor_a3_cooperative_${profile}_seed7351"
  local run_dir="output/geotransformer.p2p_liver.${run_name}"
  local checkpoint="${run_dir}/snapshots/${P2P_CHECKPOINT_NAME}"
  local result_dir="${run_dir}/evaluation_epoch150"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing trained checkpoint: $checkpoint" >&2
    exit 1
  fi
  mkdir -p "$result_dir"
  echo "[test] profile=${profile} checkpoint=${checkpoint} gpu=${CUDA_VISIBLE_DEVICES}"
  for noise in none 2 4; do
    P2P_RUN_NAME="$run_name" conda run --no-capture-output -n "$P2P_CONDA_ENV" \
      python experiments/geotransformer.p2p_liver/test.py \
      --architecture rtor_a3 \
      --interaction_profile cooperative \
      --ablation_profile "$profile" \
      --snapshot "$checkpoint" \
      --dataset in_silico \
      --noise "$noise" \
      --output "${result_dir}/in_silico_noise_${noise}.json"
  done
  P2P_RUN_NAME="$run_name" conda run --no-capture-output -n "$P2P_CONDA_ENV" \
    python experiments/geotransformer.p2p_liver/test.py \
    --architecture rtor_a3 \
    --interaction_profile cooperative \
    --ablation_profile "$profile" \
    --snapshot "$checkpoint" \
    --dataset in_vitro \
    --noise none \
    --output "${result_dir}/in_vitro_noise_none.json"
}

for profile in "${selected_profiles[@]}"; do
  case "$action" in
    train) train_profile "$profile" ;;
    test) test_profile "$profile" ;;
    all)
      train_profile "$profile"
      test_profile "$profile"
      ;;
  esac
done
