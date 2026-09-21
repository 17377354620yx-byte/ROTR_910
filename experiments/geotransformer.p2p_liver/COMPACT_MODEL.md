# RTOR+A3 compact 模型

该模型关闭 cooperative predicted proposal exposure、Poincaré 距离分支与 A3 geometry bias。

| 组件 | 状态 |
|---|---|
| cooperative 25% predicted proposal exposure | 移除 |
| RTOR overlap 软权重 | 保留 |
| RTOR Poincaré 距离分支 | 移除 |
| A3 geometry bias | 移除，保留普通 cross-attention |
| RTOR 描述子残差更新 | 保留 |

配置名为 `compact_no_proposal_poincare_geometry`，使用单编码器 RTOR+A3、cooperative interaction、原始 LGR、seed 7351 和 150 epochs。必须从头训练，不能加载已有累积消融模型作为 warm start。

训练并完成四组标准测试：

```bash
cd /home/yangx/code/new_deform/ai_worker/RTORv5
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_p2p_compact.sh all
```

只训练：

```bash
bash scripts/run_p2p_compact.sh train
```

中断后续训：

```bash
bash scripts/run_p2p_compact.sh train --resume
```

只测试 epoch 150：

```bash
bash scripts/run_p2p_compact.sh test
```

输出目录：

```text
output/geotransformer.p2p_liver.rtor_a3_cooperative_compact_no_proposal_poincare_geometry_seed7351/
├── snapshots/epoch-150.pth.tar
└── evaluation_epoch150/
    ├── in_silico_noise_none.json
    ├── in_silico_noise_2.json
    ├── in_silico_noise_4.json
    └── in_vitro_noise_none.json
```
