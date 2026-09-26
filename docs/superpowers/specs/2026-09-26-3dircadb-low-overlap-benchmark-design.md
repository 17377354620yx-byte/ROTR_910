# 3D-IRCADb 低重叠刚性配准基准设计

日期：2026-09-26  
状态：已实施并验证

## 目标

在不使用 3D-IRCADb 数据训练或微调模型的前提下，使用现有 P2P 肝脏配准权重，对 3D-IRCADb 的 20 个病例建立统一、可复现的低重叠刚性配准测试集，并比较以下十种方法（九种学习方法和一种传统方法）：

1. Ours
2. GeoTransformer
3. CASTV2
4. DFAT
5. Lepard
6. Lepard+P2P
7. PARENet
8. LiverMatch
9. LiverMatch+P2P
10. Go-ICP

每种方法只输出一个从 source 到 target 的全局刚性变换 `T=[R|t]`。

## 数据源和样本规模

- 原始数据：`/mnt/data3/yangx/3Dircadb/3Dircadb1`
- 病例：`3Dircadb1.1` 至 `3Dircadb1.20`
- 表面模型：每例的 `MESHES_VTK/liver.stl`
- 坐标单位：毫米
- 可见率：20% 和 30%
- 每个病例生成 1 个嵌套 pair，并分别保存 20% 和 30% 两档
- 总样本数：`20 病例 × 1 pair × 2 可见率 = 40`
- 全局随机种子：`20260822`

数据集输出目录：

```text
/mnt/data3/yangx/3Dircadb/low_overlap_rigid_40_seed20260822/
├── visibility_0.20/
├── visibility_0.30/
├── manifest.json
└── split.npz
```

## 数据生成协议

### 表面点云

从 `liver.stl` 的三角形表面按面积进行确定性均匀采样。数据生成器负责固定采样数量、浮点类型和点顺序，并在清单中记录源 STL、点数、包围盒和生成参数。所有模型读取同一批已落盘点云，不允许各项目重新采样 STL。

### 嵌套平面裁剪

每个病例的每个 pair 独立采样一个单位平面法向。按所有候选点在该法向上的有符号投影排序，并按固定点数选取可见集合：

- 30% 样本取排序后的 30% 点；
- 20% 样本从同一排序中取 20% 点；
- 因而严格保证同一病例同一 pair 的 `20% ⊂ 30%`。

使用排序和固定点数而不是浮点阈值近似，保证可见点数以及跨平台复现性。裁剪法向、阈值和保留索引均写入样本元数据。

### 刚性扰动

同一病例、同一 pair ID 的 20% 和 30% 样本共享同一个刚性变换：

- 随机旋转轴在单位球面采样；
- 旋转角在 `[25°, 90°]` 均匀采样；
- x、y、z 三轴平移分别在 `[-20, 20] mm` 均匀采样；
- GT 变换方向统一定义为 source 到 target：`p_target = R p_source + t`。

### 噪声

只对经过裁剪和刚性变换后的 target 添加独立各向同性高斯噪声，标准差 `σ=2 mm`。`clean_ref_points` 保存加噪前的目标点，`ref_points` 保存实际输入模型的含噪目标点。

为保证嵌套比较，30% 目标点的噪声先按固定顺序生成，20% 样本复用其对应子集的噪声。

### 样本格式

每个 `.npz` 至少保存：

```text
src_points
ref_points
clean_ref_points
transform
rotation
translation
visibility
case_id
pair_id
noise_sigma_mm
rotation_angle_deg
crop_normal
crop_threshold
crop_indices
units
```

`transform` 为 4×4 齐次矩阵，所有物理坐标和误差采用毫米。清单记录生成器版本和每个文件的相对路径；校验程序检查 40 个样本、有限数值、旋转矩阵合法性、变换方向以及 20%/30% 嵌套关系。

## 模型适配架构

采用“统一数据集 + 原生项目适配器”。RTORv2 负责数据生成、统一输出协议、汇总与可视化；各模型在自己的 Conda 环境和代码库中加载原训练配置及检查点。

计划增加以下入口：

```text
RTORv2/experiments/geotransformer.p2p_liver/test_3dircadb.py
GeoTransformer_base/experiments/geotransformer.p2p_liver/test_3dircadb.py
CASTv2/test_3dircadb.py
DFAT-main/experiments/geotransformer.p2p_liver/test_3dircadb.py
Lepard/evaluate_3dircadb.py
PARENet/experiments/P2P/test_3dircadb.py
RTORv2/tools/evaluate_livermatch_3dircadb.py
go-icp_cython/evaluate_3dircadb.py
```

Lepard 和 LiverMatch 的入口分别通过参数切换基础求解器与 P2P 求解器。适配器允许模型内部执行训练时一致的中心化、缩放、体素化或点数限制，但必须把最终变换反算回原始毫米坐标系，并保存 source 到 target 的 4×4 变换。

Go-ICP 作为无需训练的传统全局刚性配准基线，在独立 `goicp_py310` 环境运行。沿用既定参数：最大输入点数 1000、trim fraction 0.7、distance transform size 100、MSE threshold 0.001、每个样本超时 120 秒。其随机降采样必须由样本标识派生固定种子；超时和求解失败必须显式记录，不得用单位阵替代。

不得在 3D-IRCADb 上执行训练、微调、阈值搜索或按测试结果选择超参数。

## 统一预测输出

预测根目录：

```text
results/3dircadb_low_overlap_40_seed20260822/
├── ours/
├── geotransformer/
├── castv2/
├── dfat/
├── lepard/
├── lepard_p2p/
├── parenet/
├── livermatch/
├── livermatch_p2p/
└── goicp/
```

每个样本预测文件保存：

```text
estimated_transform
gt_transform
case_id
pair_id
visibility
runtime_seconds
status
```

每个方法另输出汇总 JSON/CSV。失败样本必须显式记录状态和原因，不能用单位阵静默替代。

## 指标

统一在原始毫米坐标系计算：

- RRE（度）
- RTE（毫米）
- RMSE（毫米）
- SR@5 mm
- RR@5°/5 mm：同时满足 `RRE < 5°` 且 `RTE < 5 mm`
- 推理耗时

指标实现集中放在 RTORv2，避免不同项目产生公式和单位差异。结果分别按 20%、30%、病例和全体样本汇总。

## 可视化

新增 `visualization/ircadb_method_comparison.py`，复用 `paper_method_comparison.py` 的相机拟合、点云渲染、GT 对齐误差和版式逻辑。

每幅图为两行十二列：

```text
Initial | Ours | GeoTransformer | CASTV2 | DFAT | Lepard |
Lepard+P2P | PARENet | LiverMatch | LiverMatch+P2P | Go-ICP | Ground Truth
```

- 第一行（奇数行）：配准后的 source 与含噪 target 叠加；
- 第二行（偶数行）：配准 source 相对 GT source 位置的逐点距离热图；
- 默认并显式支持 `--colormap viridis`；
- 每个热图标注 `[min, max] mm`；
- 一张图内所有方法共用同一颜色范围；
- 色标上限默认由已配准方法误差的稳健分位数确定并清楚标出截断值；
- 输出 PNG、PDF、JSON 统计和总 manifest；
- 支持按病例、pair ID、可见率和结果质量筛选。

## 运行入口

统一脚本提供以下接口：

```bash
./scripts/build_3dircadb_low_overlap.sh
./scripts/run_3dircadb_benchmark.sh ours all
./scripts/run_3dircadb_benchmark.sh geotransformer all
./scripts/run_3dircadb_benchmark.sh goicp all
./scripts/run_3dircadb_benchmark.sh all all
./scripts/render_3dircadb_comparison.sh ...
```

脚本使用指定的七个 Conda 环境顺序运行模型，避免不同 CUDA 扩展在同一进程中冲突。支持 `limit` 或环境变量进行单样本冒烟测试，正式结果必须在完整 40 个样本上生成。

## 验收标准

1. 数据构建器固定种子重复运行产生一致清单和数组。
2. 正好生成 40 个样本，20% 和 30% 各 20 个。
3. 每个病例恰有 1 个 pair，且该 pair 的 20% 点集合严格嵌套于 30%。
4. 旋转角、平移范围、噪声标准差、单位和变换方向通过自动测试。
5. 九种学习方法均可用指定检查点运行至少一个冒烟样本；Go-ICP 也须完成至少一个样本；十种方法均输出合法 4×4 变换或显式失败状态。
6. 推理过程不写入或更新任何模型权重。
7. 可视化可读取十种预测，使用 viridis 生成两行比较图和误差范围。
8. 最终交付每个模型的独立命令、全量运行命令和可视化命令。
