# P2P RTOR+A3 递进消融实验

五组实验以 `cooperative` RTOR+A3 为共同起点，按以下顺序累积删除组件：

| 阶段 | `--ablation_profile` | 新增删除项 | 同时保持删除的前序项 |
|---|---|---|---|
| 1 | `abl1_no_proposal` | 25% predicted proposal exposure | 无 |
| 2 | `abl2_no_soft_weight` | RTOR overlap 软权重 | 阶段 1 |
| 3 | `abl3_no_poincare` | RTOR Poincare 距离分支 | 阶段 1–2 |
| 4 | `abl4_no_a3_geometry` | A3 geometry bias | 阶段 1–3；保留普通 cross-attention |
| 5 | `abl5_no_rtor_descriptor` | RTOR 描述子残差更新 | 阶段 1–4；保留 overlap 预测与监督 |

所有实验固定为单编码器、`architecture=rtor_a3`、`interaction_profile=cooperative`、原始 LGR、seed 7351、150 epochs，并从头训练。不要给首次训练命令添加 `--snapshot` 或 `--warm_start`。

## 推荐：逐阶段训练并测试

在项目根目录执行：

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv5
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_p2p_ablation.sh all abl1_no_proposal
bash scripts/run_p2p_ablation.sh all abl2_no_soft_weight
bash scripts/run_p2p_ablation.sh all abl3_no_poincare
bash scripts/run_p2p_ablation.sh all abl4_no_a3_geometry
bash scripts/run_p2p_ablation.sh all abl5_no_rtor_descriptor
```

脚本固定使用 conda 环境 `geo_py310`。每个阶段完成训练后，使用 `epoch-150.pth.tar` 测试 in-silico 的 none/2/4 mm 噪声和 in-vitro。输出目录示例：

```text
output/geotransformer.p2p_liver.rtor_a3_cooperative_abl1_no_proposal_seed7351/
├── snapshots/epoch-150.pth.tar
└── evaluation_epoch150/
    ├── in_silico_noise_none.json
    ├── in_silico_noise_2.json
    ├── in_silico_noise_4.json
    └── in_vitro_noise_none.json
```

## 一次运行全部阶段

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv5
CUDA_VISIBLE_DEVICES=0 bash scripts/run_p2p_ablation.sh all all
```

这会严格按阶段 1→5 串行训练并测试，运行时间较长。

## 分离训练和测试

```bash
bash scripts/run_p2p_ablation.sh train abl3_no_poincare
bash scripts/run_p2p_ablation.sh test abl3_no_poincare
```

中断后续训：

```bash
bash scripts/run_p2p_ablation.sh train abl3_no_poincare --resume
```

若实际保存或希望评估的是 `best.pth.tar`：

```bash
P2P_CHECKPOINT_NAME=best.pth.tar \
  bash scripts/run_p2p_ablation.sh test abl3_no_poincare
```

为了与现有 epoch-150 对照保持一致，主消融表应使用默认的 `epoch-150.pth.tar`，不要混用 best checkpoint。

## 单条底层命令示例

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv5
CUDA_VISIBLE_DEVICES=0 \
P2P_RUN_NAME=rtor_a3_cooperative_abl1_no_proposal_seed7351 \
conda run --no-capture-output -n geo_py310 \
python experiments/geotransformer.p2p_liver/trainval.py \
  --architecture rtor_a3 \
  --interaction_profile cooperative \
  --ablation_profile abl1_no_proposal \
  --max_epoch 150 --lr 1e-4 --log_steps 10
```
