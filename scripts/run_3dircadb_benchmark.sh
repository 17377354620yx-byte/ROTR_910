#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yangx/code/new_deform/ai_worker/RTORv2
DATA_ROOT=${IRCADB_DATA_ROOT:-/mnt/data3/yangx/3Dircadb/low_overlap_rigid_40_seed20260822}
RESULT_ROOT=${IRCADB_RESULT_ROOT:-$ROOT/results/3dircadb_low_overlap_40_seed20260822}
GPU=${IRCADB_GPU:-1}
LIMIT=${IRCADB_LIMIT:-0}
DRY_RUN=${IRCADB_DRY_RUN:-0}
WORKERS=${IRCADB_NUM_WORKERS:-0}
GOICP_TIMEOUT=${IRCADB_GOICP_TIMEOUT:-120}

usage() {
  echo "Usage: $0 {ours|geotransformer|castv2|dfat|lepard|lepard_p2p|parenet|livermatch|livermatch_p2p|goicp|all} {020|030|all}" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
METHOD=$1
VIS_KEY=$2
case "$VIS_KEY" in
  020) VISIBILITY=0.20; VIS_SUFFIX=/visibility_020 ;;
  030) VISIBILITY=0.30; VIS_SUFFIX=/visibility_030 ;;
  all) VISIBILITY=all; VIS_SUFFIX= ;;
  *) usage ;;
esac

METHODS=(ours geotransformer castv2 dfat lepard lepard_p2p parenet livermatch livermatch_p2p goicp)
valid=0
for candidate in "${METHODS[@]}" all; do
  [[ "$METHOD" == "$candidate" ]] && valid=1
done
[[ $valid -eq 1 ]] || usage

run_cmd() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_method() {
  local method=$1
  local output="$RESULT_ROOT/$method$VIS_SUFFIX"
  case "$method" in
    ours)
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" P2P_METRICS_ROOT="$ROOT" \
        /home/yangx/miniconda3/envs/geo_v2/bin/python \
        "$ROOT/experiments/geotransformer.p2p_liver/test_3dircadb.py" \
        --data-root "$DATA_ROOT" --output "$output" --visibility "$VISIBILITY" \
        --limit "$LIMIT" --num-workers "$WORKERS" \
        --snapshot "$ROOT/output/geotransformer.p2p_liver.rtor_a3_cooperative/snapshots/epoch-150.pth.tar"
      ;;
    geotransformer)
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" P2P_METRICS_ROOT="$ROOT" \
        /home/yangx/miniconda3/envs/geotransformer/bin/python \
        /home/yangx/code/new_deform/GeoTransformer_base/experiments/geotransformer.p2p_liver/test_3dircadb.py \
        --data-root "$DATA_ROOT" --output "$output" --visibility "$VISIBILITY" \
        --limit "$LIMIT" --num-workers "$WORKERS" \
        --snapshot /home/yangx/code/new_deform/GeoTransformer_base/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar
      ;;
    castv2)
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" P2P_METRICS_ROOT="$ROOT" \
        /home/yangx/miniconda3/envs/overpredator_py310/bin/python \
        /home/yangx/code/Luo/CASTv2/test_3dircadb.py \
        --data-root "$DATA_ROOT" --output "$output" --visibility "$VISIBILITY" \
        --limit "$LIMIT" --device cuda:0 \
        --snapshot /home/yangx/code/Luo/CASTv2/ckpt/p2p_liver/castv2-p2p-liver-epoch-150.pth.tar
      ;;
    dfat)
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" P2P_METRICS_ROOT="$ROOT" \
        /home/yangx/miniconda3/envs/dfat_both/bin/python \
        /home/yangx/code/Luo/DFAT-main/experiments/geotransformer.p2p_liver/test_3dircadb.py \
        --data-root "$DATA_ROOT" --output "$output" --visibility "$VISIBILITY" \
        --limit "$LIMIT" --num-workers "$WORKERS" \
        --snapshot /home/yangx/code/Luo/DFAT-main/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar
      ;;
    lepard|lepard_p2p)
      local solver=ransac
      [[ "$method" == lepard_p2p ]] && solver=p2p
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" P2P_METRICS_ROOT="$ROOT" \
        /home/yangx/miniconda3/envs/livermatch5090/bin/python \
        /home/yangx/code/new_deform/Lepard/evaluate_3dircadb.py \
        --data-root "$DATA_ROOT" --output "$output" --visibility "$VISIBILITY" \
        --limit "$LIMIT" --device cuda:0 --solver "$solver" \
        --checkpoint /home/yangx/code/new_deform/Lepard/snapshot/p2p_liver-paper/p2p_liver_from_scratch/checkpoints/model_best_loss.pth
      ;;
    parenet)
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" P2P_METRICS_ROOT="$ROOT" \
        LD_LIBRARY_PATH="/home/yangx/miniconda3/envs/pare5090/lib/python3.11/site-packages/torch/lib:/home/yangx/miniconda3/envs/pare5090/lib:/home/yangx/miniconda3/envs/pare5090/lib64:${LD_LIBRARY_PATH:-}" \
        /home/yangx/miniconda3/envs/pare5090/bin/python \
        /home/yangx/code/point_clould/PARENet/experiments/P2P/test_3dircadb.py \
        --data-root "$DATA_ROOT" --output "$output" --visibility "$VISIBILITY" \
        --limit "$LIMIT" --num-workers "$WORKERS" \
        --snapshot /home/yangx/code/point_clould/PARENet/output/P2P/snapshots/20260919-194858/epoch-150.pth.tar
      ;;
    livermatch|livermatch_p2p)
      local mode=base
      [[ "$method" == livermatch_p2p ]] && mode=p2p
      run_cmd env CUDA_VISIBLE_DEVICES="$GPU" P2P_METRICS_ROOT="$ROOT" QT_QPA_PLATFORM=offscreen \
        /home/yangx/miniconda3/envs/livermatch5090/bin/python \
        "$ROOT/tools/evaluate_livermatch_3dircadb.py" \
        --mode "$mode" --data-root "$DATA_ROOT" --output "$output" \
        --visibility "$VISIBILITY" --limit "$LIMIT" --device cuda:0 \
        --checkpoint /home/yangx/code/point_clould/LiverMatch/snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth
      ;;
    goicp)
      run_cmd env P2P_METRICS_ROOT="$ROOT" \
        /home/yangx/miniconda3/envs/goicp_py310/bin/python \
        /home/yangx/code/new_deform/go-icp_cython/evaluate_3dircadb.py \
        --data-root "$DATA_ROOT" --output "$output" --visibility "$VISIBILITY" \
        --limit "$LIMIT" --goicp-root /home/yangx/code/new_deform/go-icp_cython \
        --max-points 1000 --trim-fraction 0.7 --dt-size 100 \
        --mse-threshold 0.001 --timeout "$GOICP_TIMEOUT"
      ;;
  esac
}

if [[ "$METHOD" == all ]]; then
  for method in "${METHODS[@]}"; do
    run_method "$method"
  done
  if [[ "$VIS_KEY" == all && "$LIMIT" == 0 ]]; then
    run_cmd /home/yangx/miniconda3/envs/geo_v2/bin/python \
      "$ROOT/tools/summarize_3dircadb_benchmark.py" \
      --data-root "$DATA_ROOT" --result-root "$RESULT_ROOT"
  fi
else
  run_method "$METHOD"
fi
