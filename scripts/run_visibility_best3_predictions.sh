#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yangx/code/new_deform/ai_worker/RTORv2"
RESULT_ROOT="$PROJECT_ROOT/results/visibility_0.2_0.4_benchmark_metrics_v2"
SILICO_ROOT="/mnt/data3/yangx/P2P/in_silico_visibility_0.2_0.4"
VITRO_ROOT="/mnt/data3/yangx/P2P/in_vitro_visibility_0.2_0.4"
OUTPUT_ROOT="${P2P_BEST3_ROOT:-$PROJECT_ROOT/output/visualization/visibility_0.2_0.4_livermatch_style}"
RAW_ROOT="$OUTPUT_ROOT/raw"
SELECTION="$OUTPUT_ROOT/selection.json"
GPU="${P2P_GPU:-1}"
DRY_RUN="${P2P_DRY_RUN:-0}"
FORCE="${P2P_FORCE:-0}"

GEO_ROOT="/home/yangx/code/new_deform/GeoTransformer_base"
CAST_ROOT="/home/yangx/code/Luo/CASTv2"
DFAT_ROOT="/home/yangx/code/Luo/DFAT-main"
LEPARD_ROOT="/home/yangx/code/new_deform/Lepard"
PARE_ROOT="/home/yangx/code/point_clould/PARENet"
GOICP_ROOT="/home/yangx/code/new_deform/go-icp_cython"

OURS_CKPT="$PROJECT_ROOT/output/geotransformer.p2p_liver.rtor_a3_cooperative/snapshots/epoch-150.pth.tar"
GEO_CKPT="$GEO_ROOT/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar"
CAST_CKPT="$CAST_ROOT/ckpt/p2p_liver/castv2-p2p-liver-epoch-150.pth.tar"
DFAT_CKPT="$DFAT_ROOT/output/geotransformer.p2p_liver/snapshots/epoch-150.pth.tar"
LEPARD_CKPT="$LEPARD_ROOT/snapshot/p2p_liver-paper/p2p_liver_from_scratch/checkpoints/model_best_loss.pth"
PARE_CKPT="$PARE_ROOT/output/P2P/snapshots/20260919-194858/epoch-150.pth.tar"
LIVERMATCH_CKPT="/home/yangx/code/point_clould/LiverMatch/snapshot/liver_new_task3_004_002/checkpoints/model_best_loss.pth"

PY_GEO_V2="/home/yangx/miniconda3/envs/geo_v2/bin/python"
PY_GEO="/home/yangx/miniconda3/envs/geotransformer/bin/python"
PY_CAST="/home/yangx/miniconda3/envs/overpredator_py310/bin/python"
PY_DFAT="/home/yangx/miniconda3/envs/dfat_both/bin/python"
PY_LIVER="/home/yangx/miniconda3/envs/livermatch5090/bin/python"
PY_PARE="/home/yangx/miniconda3/envs/pare5090/bin/python"
PY_GOICP="/home/yangx/miniconda3/envs/goicp_py310/bin/python"

METHODS=(ours geotransformer castv2 dfat lepard lepard_p2p parenet livermatch livermatch_p2p goicp)
DATASETS=(in_silico in_vitro)

checkpoint_for() {
  case "$1" in
    ours) echo "$OURS_CKPT" ;;
    geotransformer) echo "$GEO_CKPT" ;;
    castv2) echo "$CAST_CKPT" ;;
    dfat) echo "$DFAT_CKPT" ;;
    lepard|lepard_p2p) echo "$LEPARD_CKPT" ;;
    parenet) echo "$PARE_CKPT" ;;
    livermatch|livermatch_p2p) echo "$LIVERMATCH_CKPT" ;;
    goicp) echo "$GOICP_ROOT" ;;
  esac
}

python_for() {
  case "$1" in
    ours) echo "$PY_GEO_V2" ;;
    geotransformer) echo "$PY_GEO" ;;
    castv2) echo "$PY_CAST" ;;
    dfat) echo "$PY_DFAT" ;;
    lepard|lepard_p2p|livermatch|livermatch_p2p) echo "$PY_LIVER" ;;
    parenet) echo "$PY_PARE" ;;
    goicp) echo "$PY_GOICP" ;;
  esac
}

if [[ "$DRY_RUN" == "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      extra="--save_predictions"
      [[ "$method" == "lepard_p2p" || "$method" == "livermatch_p2p" ]] && extra="$extra --k 5"
      [[ "$method" == "goicp" ]] && extra="--goicp-root $GOICP_ROOT"
      [[ "$method" == "parenet" ]] && extra="$extra artifact=$OUTPUT_ROOT/raw/$dataset/parenet_predictions"
      [[ "$method" == "castv2" ]] && extra="$extra P2P_IN_VITRO_ROOT=$OUTPUT_ROOT/subsets/in_vitro"
      printf 'MODEL_CMD %s %s %s checkpoint=%s %s output=%s/raw/%s/%s.json\n' \
        "$method" "$dataset" "$(python_for "$method")" "$(checkpoint_for "$method")" \
        "$extra" "$OUTPUT_ROOT" "$dataset" "$method"
    done
  done
  exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$RAW_ROOT"
"$PY_GEO_V2" "$PROJECT_ROOT/tools/prepare_visibility_best3.py" \
  --result-root "$RESULT_ROOT" \
  --in-silico-root "$SILICO_ROOT" --in-vitro-root "$VITRO_ROOT" \
  --output-root "$OUTPUT_ROOT" --top-k 3

SILICO_SUBSET="$OUTPUT_ROOT/subsets/in_silico"
VITRO_SUBSET="$OUTPUT_ROOT/subsets/in_vitro"

run_logged() {
  local output="$1"
  shift
  mkdir -p "$(dirname "$output")"
  printf 'Running:'
  printf ' %q' "$@"
  printf '\n'
  "$@" 2>&1 | tee "${output%.json}.log"
}

cache_complete() {
  local method="$1"
  local dataset="$2"
  local summary="$OUTPUT_ROOT/predictions/$dataset/${method}_summary.json"
  [[ "$FORCE" != "1" && -s "$summary" ]] || return 1
  "$PY_GEO_V2" - "$summary" "$(checkpoint_for "$method")" <<'PY'
import json, pathlib, sys
payload=json.loads(pathlib.Path(sys.argv[1]).read_text())
rows=payload.get("samples", [])
expected=str(pathlib.Path(sys.argv[2]).resolve())
valid=len(rows)==3 and all(pathlib.Path(r["prediction_path"]).is_file() for r in rows)
if sys.argv[2].endswith("go-icp_cython"):
    valid = valid and all(not r.get("checkpoint") for r in rows)
else:
    valid = valid and all(str(pathlib.Path(r.get("checkpoint", "")).resolve()) == expected for r in rows)
raise SystemExit(0 if valid else 1)
PY
}

collect() {
  local method="$1" dataset="$2" summary="$3" artifact_kind="${4:-}" artifact_path="${5:-}"
  local args=(
    "$PY_GEO_V2" "$PROJECT_ROOT/tools/collect_visibility_predictions.py"
    --method "$method" --dataset "$dataset" --selection "$SELECTION"
    --summary "$summary" --output-root "$OUTPUT_ROOT"
  )
  if [[ "$method" != "goicp" ]]; then
    args+=(--expected-checkpoint "$(checkpoint_for "$method")")
  fi
  [[ "$artifact_kind" == "prediction_dir" ]] && args+=(--prediction-dir "$artifact_path")
  [[ "$artifact_kind" == "transforms_path" ]] && args+=(--transforms-path "$artifact_path")
  "${args[@]}"
}

run_method() {
  local method="$1" dataset="$2"
  if cache_complete "$method" "$dataset"; then
    echo "Cached: $method/$dataset"
    return
  fi
  local noise="none" data_root output prediction_dir mode solver transforms_path
  output="$RAW_ROOT/$dataset/$method.json"
  data_root="$SILICO_SUBSET"
  [[ "$dataset" == "in_vitro" ]] && data_root="$VITRO_SUBSET"
  case "$method" in
    ours)
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_RUN_NAME=visibility_best3_ours \
        P2P_DATA_ROOT="$SILICO_SUBSET" P2P_IN_VITRO_ROOT="$VITRO_SUBSET" \
        P2P_TRAIN_DATA_ROOT=/mnt/data3/yangx/P2P/Dataset \
        "$PY_GEO_V2" "$PROJECT_ROOT/experiments/geotransformer.p2p_liver/test.py" \
        --snapshot "$OURS_CKPT" --dataset "$dataset" --noise "$noise" \
        --architecture rtor_a3 --interaction_profile cooperative --save_predictions --output "$output"
      prediction_dir="$PROJECT_ROOT/output/geotransformer.p2p_liver.visibility_best3_ours/registration"
      collect "$method" "$dataset" "$output" prediction_dir "$prediction_dir"
      ;;
    geotransformer)
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_DATA_ROOT="$SILICO_SUBSET" \
        P2P_IN_VITRO_ROOT="$VITRO_SUBSET" P2P_TRAIN_DATA_ROOT=/mnt/data3/yangx/P2P/Dataset \
        P2P_METRICS_ROOT="$PROJECT_ROOT" "$PY_GEO" \
        "$GEO_ROOT/experiments/geotransformer.p2p_liver/test_standard.py" \
        --snapshot "$GEO_CKPT" --dataset "$dataset" --noise "$noise" \
        --save_predictions --output "$output"
      collect "$method" "$dataset" "$output" prediction_dir \
        "$GEO_ROOT/output/geotransformer.p2p_liver/registration"
      ;;
    castv2)
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_DATA_ROOT="$data_root" \
        P2P_IN_VITRO_ROOT="$VITRO_SUBSET" \
        P2P_METRICS_ROOT="$PROJECT_ROOT" "$PY_CAST" "$CAST_ROOT/test_p2p_liver.py" \
        --config "$CAST_ROOT/config/p2p_liver.json" --snapshot "$CAST_CKPT" \
        --dataset "$dataset" --noise "$noise" --save_predictions --output "$output"
      transforms_path="${output%.json}_transforms.npy"
      collect "$method" "$dataset" "$output" transforms_path "$transforms_path"
      ;;
    dfat)
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_DATA_ROOT="$SILICO_SUBSET" \
        P2P_IN_VITRO_ROOT="$VITRO_SUBSET" P2P_TRAIN_DATA_ROOT=/mnt/data3/yangx/P2P/Dataset \
        P2P_METRICS_ROOT="$PROJECT_ROOT" "$PY_DFAT" \
        "$DFAT_ROOT/experiments/geotransformer.p2p_liver/test.py" \
        --snapshot "$DFAT_CKPT" --dataset "$dataset" --noise "$noise" \
        --save_predictions --output "$output"
      collect "$method" "$dataset" "$output" prediction_dir \
        "$DFAT_ROOT/output/geotransformer.p2p_liver/registration"
      ;;
    lepard|lepard_p2p)
      solver="ransac"; [[ "$method" == "lepard_p2p" ]] && solver="p2p"
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" "$PY_LIVER" \
        "$LEPARD_ROOT/evaluate_p2p.py" --config "$LEPARD_ROOT/configs/train/p2p_liver.yaml" \
        --checkpoint "$LEPARD_CKPT" --dataset "$dataset" --noise "$noise" --solver "$solver" \
        --in-silico-root "$SILICO_SUBSET/Deform_mesh_npz_test/Test" \
        --in-silico-list "$SILICO_SUBSET/Deform_mesh_npz_test/list.npz" \
        --in-vitro-root "$VITRO_SUBSET" --in-vitro-list "$VITRO_SUBSET/rigid_list.npy" \
        --output "$output" --device cuda:0
      collect "$method" "$dataset" "$output"
      ;;
    parenet)
      local pare_torch_lib
      pare_torch_lib="$($PY_PARE -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" P2P_RUN_NAME=visibility_best3 \
        P2P_DATA_ROOT="$SILICO_SUBSET" P2P_IN_VITRO_ROOT="$VITRO_SUBSET" \
        LD_LIBRARY_PATH="$pare_torch_lib:/home/yangx/miniconda3/envs/pare5090/lib:/home/yangx/miniconda3/envs/pare5090/lib64:${LD_LIBRARY_PATH:-}" \
        "$PY_PARE" "$PARE_ROOT/experiments/P2P/test_standard.py" \
        --snapshot "$PARE_CKPT" --dataset "$dataset" --noise "$noise" \
        --num_workers 0 --save_predictions --output "$output"
      collect "$method" "$dataset" "$output" prediction_dir \
        "${output%.json}_predictions"
      ;;
    livermatch|livermatch_p2p)
      mode="base"; [[ "$method" == "livermatch_p2p" ]] && mode="p2p"
      run_logged "$output" env CUDA_VISIBLE_DEVICES="$GPU" QT_QPA_PLATFORM=offscreen \
        PYVISTA_OFF_SCREEN=true "$PY_LIVER" "$PROJECT_ROOT/tools/evaluate_livermatch_visibility.py" \
        --mode "$mode" --dataset "$dataset" --noise "$noise" --data-root "$data_root" \
        --checkpoint "$LIVERMATCH_CKPT" --k 5 --output "$output" --device cuda:0
      collect "$method" "$dataset" "$output"
      ;;
    goicp)
      run_logged "$output" env P2P_METRICS_ROOT="$PROJECT_ROOT" "$PY_GOICP" \
        "$PROJECT_ROOT/tools/evaluate_goicp_visibility.py" --dataset "$dataset" --noise "$noise" \
        --data-root "$data_root" --goicp-root "$GOICP_ROOT" --max-points 1000 \
        --trim-fraction 0.7 --dt-size 100 --mse-threshold 0.001 --timeout 120 \
        --output "$output"
      collect "$method" "$dataset" "$output"
      ;;
  esac
}

verify_ours() {
  local dataset="$1"
  "$PY_GEO_V2" - "$SELECTION" "$OUTPUT_ROOT/predictions/$dataset/ours_summary.json" <<'PY'
import json, pathlib, sys
selection=json.loads(pathlib.Path(sys.argv[1]).read_text())["datasets"]
dataset=json.loads(pathlib.Path(sys.argv[2]).read_text())["dataset"]
expected={r["sample"]: float(r["ours_rms_tre_mm"]) for r in selection[dataset]["cases"]}
actual=json.loads(pathlib.Path(sys.argv[2]).read_text())["samples"]
for row in actual:
    delta=abs(float(row["rms_tre_mm"])-expected[row["sample"]])
    if delta > 0.05:
        raise SystemExit(f"Ours RMS-TRE mismatch {dataset}/{row['sample']}: "
                         f"selected={expected[row['sample']]:.6f}, rerun={row['rms_tre_mm']:.6f}, delta={delta:.6f}")
print(f"Ours consistency passed for {dataset} (tolerance 0.05 mm)")
PY
}

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_method "$method" "$dataset"
  done
  verify_ours "$dataset"
done

echo "All selected predictions are under $OUTPUT_ROOT/predictions"
