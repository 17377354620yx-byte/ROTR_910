# RTOR+A3 cooperative 实验候选：训练与测试

本版是待验证的 coarse-to-fine 接口实验，尚不能认定为负交互修复或最优模型，使用单骨干、软 RTOR overlap 和逐步混合 predicted/GT proposals。保持原 loss 权重和 LGR 设置。完整训练后的效果尚待验证，不能声称已经达到全局最优。

GT 训练 / predicted 测试是 GeoTransformer/DFAT 共有设计，不足以支持启用混合训练；以下命令仅供显式实验使用。

快捷命令：在项目根目录执行 `bash scripts/train_cooperative.sh`，完成后执行 `bash scripts/test_cooperative.sh`。两个脚本固定使用 `geo_v2`，测试自动读取该 run 的 `best.pth.tar`，覆盖 in-silico 三个噪声设置和 in-vitro。

## 实验候选从头训练

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0 P2P_RUN_NAME=rtor_a3_cooperative_v1 \
conda run --no-capture-output -n geo_v2 \
python experiments/geotransformer.p2p_liver/trainval.py \
  --architecture rtor_a3 \
  --interaction_profile cooperative \
  --max_epoch 150 \
  --lr 1e-4 \
  --log_steps 10
```

默认 seed 为 7351。最优训练侧 validation checkpoint 保存为：

```text
output/geotransformer.p2p_liver.rtor_a3_cooperative_v1/snapshots/best.pth.tar
```

## 中断后续训

使用相同命令并增加 `--resume`，从该 run 的 `snapshots/snapshot.pth.tar` 恢复。

## 测试

在上述项目目录执行：

```bash
P2P_SNAPSHOT=output/geotransformer.p2p_liver.rtor_a3_cooperative_v1/snapshots/best.pth.tar

for P2P_NOISE in none 2 4; do
  CUDA_VISIBLE_DEVICES=0 P2P_RUN_NAME=rtor_a3_cooperative_v1 \
  conda run --no-capture-output -n geo_v2 \
  python experiments/geotransformer.p2p_liver/test.py \
    --architecture rtor_a3 \
    --interaction_profile cooperative \
    --snapshot "$P2P_SNAPSHOT" \
    --dataset in_silico --noise "$P2P_NOISE" \
    --output "output/geotransformer.p2p_liver.rtor_a3_cooperative_v1/in_silico_noise_${P2P_NOISE}.json"
done

CUDA_VISIBLE_DEVICES=0 P2P_RUN_NAME=rtor_a3_cooperative_v1 \
conda run --no-capture-output -n geo_v2 \
python experiments/geotransformer.p2p_liver/test.py \
  --architecture rtor_a3 \
  --interaction_profile cooperative \
  --snapshot "$P2P_SNAPSHOT" \
  --dataset in_vitro --noise none \
  --output output/geotransformer.p2p_liver.rtor_a3_cooperative_v1/in_vitro.json
```

每项测试生成 JSON、逐样本 CSV 和 RMS-TRE NPY。`best.pth.tar` 按训练侧 validation 选择；最终测试用于报告，不再用于选择 profile、epoch 或 loss weight。

## 原模型对照

将训练与测试中的 `--interaction_profile cooperative` 改为 `--interaction_profile legacy`，并使用独立 `P2P_RUN_NAME`。该实验候选不需要 `--dual_encoder` 或 `--registration_profile tight`。

[机制诊断及图表](diagnostics/interaction/REPORT.md)
