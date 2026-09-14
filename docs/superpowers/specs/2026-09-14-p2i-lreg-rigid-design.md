# P2I-LReg Synthetic Rigid Registration 设计

## 范围

在 RTORv2 中新增独立的 `experiments/geotransformer.p2i_lreg/` 实验，只支持 P2I-LReg synthetic rigid registration。不得修改 `experiments/geotransformer.p2p_liver/`，不得实现或调用 SRSA、NDP、形变场、Dice、non-rigid Chamfer、ICP、GT/oracle 推理分支。

## 数据协议

- 数据根目录通过 `P2I_LREG_ROOT` 或命令行参数指定。
- 使用官方 `train_syn.txt` 和 `test_syn.txt`；不把 test 当 validation 选择 checkpoint。
- source：`<patient>/model/reconstructed_mesh_world_m.obj` 的完整术前肝脏。
- reference：`<patient>/syn/liverPcds/<frame>.ply` 的 synthetic 局部肝脏。
- `camPose.yml` 给出 C2W；GT 为其逆矩阵 W2C，即 `src -> ref`。
- OBJ、网络内部坐标和所有阈值均为 meter；PLY 与 `cam_t_c2w` 从 mm 除以 1000。
- 遵循官方 target 清洗顺序：every-8、去重、统计离群点剔除、去零点；之后 source/ref 均做 0.001 m voxel downsample。
- 每云固定 8192 点：多于 8192 时训练随机无放回采样、验证/测试按 sample seed 确定性采样；不足时确定性/随机 wrap sampling。返回常数 1 feature。
- 不启用官方会破坏 SO(3) 的 scale augmentation。训练可配置刚性旋转与 meter-space jitter，但默认关闭，以保持首个主协议清晰。
- 每个 sample 返回 `ref_points`、`src_points`、`ref_feats`、`src_feats`、`transform`、`patient_id`、`frame_id`、`case_id`。

## 模型协议

直接复用本项目 `geotransformer.modules.liver.registration_model.GeoTransformer`：

- `architecture=geotransformer`：`rtor_enabled=False`、`a3_enabled=False`。
- `architecture=rtor`：`rtor_enabled=True`、`a3_enabled=False`。
- 两者使用相同 KPConvFPN、GeoTransformer、Sinkhorn、训练损失、数据、采样和评估代码。
- 训练阶段允许 `transform` 仅用于构造监督 correspondence/loss；测试 forward 在构造模型输入前移除 `transform`，输出后才交给 evaluator，保证 GT 不进入推理。

## Correspondence 与 pose estimator

- 两个模型统一输出 `src_corr_points`、`ref_corr_points`、`corr_scores`。
- 统一按 score 降序、稳定 tie-break 选取 top-k；默认支持 `2000,1500,1000,500,250`，少于 k 时使用全部有效 correspondence 并记录实际数量。
- `weighted_svd`：对同一 top-k correspondence 按归一化 score 求 source→ref SE(3)。
- `ransac_50k`：Open3D point-to-point RANSAC，50,000 iterations，最大 correspondence distance 0.01 m，无 ICP refinement。
- `lgr`：复用项目 LocalGlobalRegistration 的原生估计路径；两个 architecture 共享完全相同的 LGR 配置。测试结果中明确区分 native LGR 与对 flatten top-k 重算的两类 estimator。
- 任一 estimator 少于 3 对有效 correspondence 时返回显式失败状态，不以 identity 当成功结果。

## 指标协议

所有阈值集中在 `config.py`，当前 `released_corrected_v1` 协议为：

- IR：GT 对齐后的预测 correspondence 欧氏距离 `< 0.01 m` 的比例。
- FMR：样本 IR `> 0.05` 的样本比例。
- RR：全部 source 点上 `mean(||T_est(x)-T_gt(x)||) < 0.01 m` 的样本比例。
- RRE：`acos(clamp((trace(R_gt^T R_est)-1)/2))`，报告 degree。
- RTE：`||t_gt-t_est||`，内部 meter，报告 mm。
- estimator failure 的 RR=0；RRE/RTE 记为 NaN，并分别报告 valid-pose 数量，避免以 identity 稀释误差。

若后续取得不同的论文精确阈值，新增独立 `paper` protocol，不修改或覆盖 `released_corrected_v1`。

## 文件边界

- `config.py`：路径、模型开关、训练参数、所有 metric/estimator 阈值。
- `dataset.py`：官方 synthetic 数据读取、单位/GT、voxel 与 8192 点采样、dataloader。
- `metrics.py`：IR/FMR/RR/RRE/RTE 的纯函数和 accumulator。
- `estimators.py`：稳定 top-k、weighted SVD、RANSAC-50k、native LGR 选择。
- `model.py`：复用源模型并执行 architecture 开关。
- `loss.py`：共享 coarse/fine/RTOR loss；baseline 的 RTOR loss 为零。
- `trainval.py`：训练与验证入口、manifest/checkpoint；默认正式配置 120 epoch，但 smoke 参数只跑 1 epoch。
- `test.py`：无 GT forward 后共享 evaluator，多 estimator/top-k 表格输出。
- `check_dataset.py`：至少随机 100 个官方 test sample 的 GT 前后 asymmetric/symmetric Chamfer 检查。
- `tests/test_p2i_lreg_*.py`：数据、方向、单位、采样、指标、estimator 和 inference contract 测试。

## Sanity 与验收

`check_dataset.py --num-samples 100` 对每个样本计算：

- `ref -> raw src` 单向最近邻均值；
- `ref -> T_gt(src)` 单向最近邻均值；
- symmetric Chamfer before/after。

至少 90% 样本必须同时满足 after < before，并要求总体 median improvement ratio 明显小于 1；否则非零退出并停止 smoke training。

验收顺序：测试先失败 → 最小实现 → 测试通过 → 100-sample sanity → baseline forward → RTOR forward → 两者 loss/backward → 各自 1 epoch smoke → synthetic metric/estimator 测试。不得启动 120 epoch 正式训练。
