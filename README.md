# RTOR+A3 for P2P liver registration

本项目在 GeoTransformer 上实现 RTOR 和 A3，输出单个刚体变换 `(R, t)`。

保留的实验候选为 `--interaction_profile cooperative`：保留单骨干和原 RTOR/A3，使用 soft overlap proposals，并在 warm-up 后逐步混入预测 coarse candidates，检验候选训练策略的作用（尚未证实提升）。原 loss 权重、Sinkhorn 和 LGR 设置保持不变。

- [训练和测试命令](TRAIN_TEST.md)
- [负交互诊断、证据与限制](diagnostics/interaction/REPORT.md)
- `--interaction_profile legacy` 提供原结构控制组。
- `--architecture geotransformer|rtor_only|a3_only|rtor_a3` 提供四种严格参数结构。

使用 conda 环境 `geo_v2`，工作目录为 `/home/yangx/code/new_deform/ai_worker/RTORv2`。数据读取自 `/mnt/data3/yangx/P2P/`，新代码、日志和模型输出均位于本项目。

已完成训练侧机制诊断和小规模训练验证；候选模型尚需完整训练，未宣称已达到最佳测试精度。
