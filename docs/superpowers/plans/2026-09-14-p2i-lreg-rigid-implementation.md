# P2I-LReg Synthetic Rigid Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改既有 P2P 实验的前提下，新增可训练、可公平评测的 P2I-LReg synthetic rigid GeoTransformer 与 RTOR 实验。

**Architecture:** 新实验复用项目的 `GeoTransformer`、KPConvFPN、RTOR 和 LGR 公共模块。独立数据层将官方 C2W pose 转成 meter-space src→ref GT；统一 estimator/metric 层只消费模型预测 correspondence，测试 forward 不接收 GT。

**Tech Stack:** Python 3.10、PyTorch、NumPy、SciPy、Open3D、项目现有 GeoTransformer/KPConv CUDA extension、pytest/unittest，conda `geo_v2`。

**Spec:** `docs/superpowers/specs/2026-09-14-p2i-lreg-rigid-design.md`

## Global Constraints

- 只新增 `experiments/geotransformer.p2i_lreg/` 和对应测试，不修改 `experiments/geotransformer.p2p_liver/`。
- 只使用 P2I-LReg synthetic rigid 数据；禁止 SRSA、NDP、non-rigid、ICP、oracle 和 GT 推理输入。
- source 为完整术前肝脏，reference 为 synthetic 局部肝脏，GT 恒为 src→ref。
- 内部单位 meter，voxel size 0.001 m，每云 8192 点。
- 当前协议固定 IR=0.01 m、FMR=0.05、RR=0.01 m、RANSAC=50,000 iterations/0.01 m；阈值只定义于 `config.py`。
- 不启动 120 epoch 正式训练，只运行测试、100-sample sanity 和 1-epoch smoke。

---

### Task 1: 数据协议与采样

**Files:**
- Create: `tests/test_p2i_lreg_dataset.py`
- Create: `experiments/geotransformer.p2i_lreg/config.py`
- Create: `experiments/geotransformer.p2i_lreg/dataset.py`
- Create: `experiments/geotransformer.p2i_lreg/check_dataset.py`

**Interfaces:**
- Produces: `P2ILRegDataset(root, split, cfg, limit=None)`, `voxel_downsample`, `sample_fixed_points`, `load_source_to_reference_transform`, `train_valid_data_loader`, `test_data_loader`。

- [ ] 写测试：合成 fixture 验证 C2W 取逆、mm→m、src/ref 映射、1 mm voxel、8192 点、确定性 test sampling、split 无交集和 Chamfer after<before。
- [ ] 运行 `conda run -n geo_v2 pytest -q tests/test_p2i_lreg_dataset.py`，确认因模块不存在而失败。
- [ ] 实现 config 和 dataset 最小接口，数据根默认 `/mnt/data3/publicData/P2I-LReg/Liver_regis`，训练/验证仅从 train split 派生确定性 disjoint validation，测试严格用 test split。
- [ ] 实现 checker：固定 seed 抽样至少 100，输出 JSON 汇总；改善样本占比低于 0.9 时退出非零。
- [ ] 重跑数据测试并执行 `check_dataset.py --num-samples 100`。

### Task 2: 统一指标

**Files:**
- Create: `tests/test_p2i_lreg_metrics.py`
- Create: `experiments/geotransformer.p2i_lreg/metrics.py`

**Interfaces:**
- Produces: `correspondence_inlier_ratio`, `registration_errors`, `registration_recall`, `MetricAccumulator`。

- [ ] 写测试：精确构造 9/10 inlier、FMR 边界、10 mm RR 边界、90° RRE、10 mm RTE、失败 pose NaN 统计。
- [ ] 运行指标测试并确认缺失模块失败。
- [ ] 用 Torch/NumPy 纯函数实现欧氏距离阈值、degree/mm 输出和 macro accumulator。
- [ ] 重跑指标测试。

### Task 3: 统一 estimator 与 top-k

**Files:**
- Create: `tests/test_p2i_lreg_estimators.py`
- Create: `experiments/geotransformer.p2i_lreg/estimators.py`

**Interfaces:**
- Produces: `select_topk_correspondences`, `weighted_svd`, `ransac_50k`, `estimate_pose(method, output, cfg, topk)` 和显式 `PoseEstimate` 状态。

- [ ] 写测试：稳定 top-k、已知刚体 weighted SVD、少于 3 对失败、RANSAC 已知变换、LGR 只接受模型 native estimate 且不读 GT。
- [ ] 运行 estimator 测试并确认缺失模块失败。
- [ ] 实现 score-only stable top-k、CPU/GPU weighted SVD、Open3D RANSAC 50k 和 native LGR wrapper；所有输出 src→ref。
- [ ] 重跑 estimator 测试。

### Task 4: 模型、loss 与无 GT inference contract

**Files:**
- Create: `tests/test_p2i_lreg_model.py`
- Create: `experiments/geotransformer.p2i_lreg/model.py`
- Create: `experiments/geotransformer.p2i_lreg/loss.py`

**Interfaces:**
- Produces: `create_model(cfg)`、`OverallLoss(cfg)`、`forward_without_ground_truth(model, batch)`。

- [ ] 写测试：baseline 不含 RTOR/A3 参数；RTOR 只含 RTOR；inference helper 从复制 batch 中删除 transform 且原 batch 保留；loss 可反向传播。
- [ ] 运行模型测试并确认失败。
- [ ] 复用源模型，architecture 只允许 `geotransformer|rtor`；复用 coarse/fine loss 并仅在 RTOR 模式加入 topology overlap loss。
- [ ] 在 synthetic 小批次上运行 baseline/RTOR forward 与 loss/backward。

### Task 5: 训练、测试入口与 smoke

**Files:**
- Create: `experiments/geotransformer.p2i_lreg/trainval.py`
- Create: `experiments/geotransformer.p2i_lreg/test.py`
- Create: `experiments/geotransformer.p2i_lreg/README.md`

**Interfaces:**
- Produces: CLI `trainval.py --architecture ...`、`test.py --architecture ... --estimator ... --topk ...`。

- [ ] 写 CLI/parser 和 protocol manifest 测试，确认正式默认 120 epoch、smoke 覆盖为 1 epoch、所有阈值来自 cfg。
- [ ] 实现训练入口，checkpoint metadata 固化 architecture、split、threshold、neighbor limits、seed；validation 仅来自 train split。
- [ ] 实现测试入口：先无 GT forward，再统一 top-k/estimator/metrics，输出 per-sample CSV 和论文表格 JSON。
- [ ] 分别执行 baseline 与 RTOR `--max_epoch 1 --train_limit ... --validation_size ...` smoke，不启动正式训练。
- [ ] 执行测试套件和静态编译检查，确认既有 P2P 文件未变。

### Task 6: 最终验证

**Files:**
- Verify all files above.

- [ ] 运行 `conda run -n geo_v2 pytest -q tests/test_p2i_lreg_dataset.py tests/test_p2i_lreg_metrics.py tests/test_p2i_lreg_estimators.py tests/test_p2i_lreg_model.py`。
- [ ] 运行 100-sample 数据 sanity，并保存真实结果路径。
- [ ] 核对两个 smoke run 的日志、checkpoint、loss/backward 与退出码。
- [ ] 搜索新目录中的 `ICP|SRSA|NDP|deform|oracle` 和测试 forward 的 `transform` 数据流，确认禁止项未进入实现。
- [ ] 汇报文件、协议、阈值、sanity、smoke 和两条正式训练命令。
