# P2P RTOR+A3 experiment

请使用项目根目录的 [TRAIN_TEST.md](../../TRAIN_TEST.md) 中的 `geo_v2` 命令。

`cooperative` 是未证实提升的实验候选，profile 保留原单骨干、RTOR、A3 和 LGR，取消 source 硬筛选，逐步引入 predicted coarse training proposals。`legacy` profile 用作原模型控制组。

- `config.py`：架构与 interaction profile。
- `trainval.py`：训练、验证、best checkpoint。
- `test.py`：严格加载、RMS-TRE 和逐样本导出。
- `../../diagnostics/interaction/`：固定训练输入的结构、loss、梯度诊断。

完整诊断和适用边界见 [REPORT.md](../../diagnostics/interaction/REPORT.md)。
