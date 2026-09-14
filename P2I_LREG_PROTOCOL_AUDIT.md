# P2I-LReg / Self-P2IR 刚性配准协议审计

## Material Passport

- **文档状态**：第一阶段审计完成；尚未实现或运行 RTOR 的 P2I-LReg 实验。
- **审计日期**：2026-09-14（Asia/Shanghai）。
- **Self-P2IR 源码快照**：官方仓库 `main`，commit [`60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079`](https://github.com/junzastar/Self-P2IR/tree/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079)（commit 时间 2025-12-04）。
- **论文来源**：[arXiv:2504.15152](https://arxiv.org/abs/2504.15152)，*Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection*, IEEE TMI 2025。
- **数据来源**：[Kaggle 官方 P2I-LReg 发布页](https://www.kaggle.com/datasets/junezhou001/p2i-lreg)，版本 1，更新时间 2025-12-03；通过官方 API 读取 21 位患者的 split 清单并对一个真实发布样本作数值检查，没有下载全量约 155 GB 数据。
- **证据等级**：`CODE` 表示发布源码直接证据；`DATA` 表示发布数据文件直接证据；`PAPER` 表示论文文字；`INFERENCE` 表示基于前三者的推断。
- **范围**：只审计 SE(3) rigid registration；SRSA、deformation pyramid、NDP/non-rigid deformation 不纳入拟实现范围。

## 1. 结论摘要

1. **Self-P2IR 的监督式刚性网络只用 `syn` 训练和评估，不将 `real` 与 `syn` 混合。** `real` 分支加载 synthetic 预训练刚性网络、冻结 `posenet`，再用 mask/CD 自监督训练非刚性部分。论文 Table I 的 RR/IR/FMR/RRE/RTE 也是在 synthetic dataset 上评估。因此公平的 rigid-only 主实验必须使用 `syn`。（`CODE` + `PAPER`）
2. **source 是患者对应的完整术前肝脏模型，target/ref 是同一患者的 synthetic 局部可见肝表面点云。** Self-P2IR 命名为 `preope_pcd_src` 与 `intra_liver_pcd_tgt`；RTOR 当前命名应为 `src_points=preoperative complete liver`、`ref_points=synthetic partial liver`。（`CODE`）
3. **GT 严格为 source→target：术前模型的 Blender world/object 坐标 → synthetic 相机坐标。** `camPose.yml` 给出 C2W，代码先将平移从 mm 除以 1000，再整体求逆为 W2C，并按 `R @ source + t` 使用。（`CODE` + `DATA`）
4. **网络输入统一使用米。** 术前 OBJ 文件名和数值为米；synthetic/real PLY 均在 adapter 中 `/1000` 从毫米变为米；RTE 输出若用于论文必须显式乘 1000 变为 mm。（`CODE` + `DATA`）
5. **论文协议与发布代码/发布 split 有多处实质不一致。** 最主要的是 120 epochs vs 配置 80（实际循环 79）、batch 2 vs 单卡 batch 1/脚本 3 GPU、synthetic 60/20/20 vs 发布清单约 94–95%/5–6% 且无 val、top-k + RANSAC-50k/LGR/weighted SVD vs 发布 tester 实际“阈值取匹配 + 无权 SVD”。这些差异不能静默抹平。（`CODE` + `DATA` + `PAPER`）
6. **当前 RTOR 核心 convention 与官方代码一致，都是 `src -> ref`。** 新 adapter 不需要反转 GT，但必须在加载时验证 `apply_transform(src, T_gt)`。（`CODE`）

## 2. Dataset

### 2.1 目录和患者组织

发布数据根目录内有 `01`–`21` 共 21 个患者目录：

```text
<root>/
├── 01/
│   ├── model/reconstructed_mesh_world_m.obj
│   ├── real/
│   │   ├── labels/<frame_id>/{label.png,img.png,label_names.txt}
│   │   ├── liverPcds/<frame_id>.ply
│   │   ├── scale/<frame_id>.txt
│   │   └── CAM_K.yml
│   ├── syn/
│   │   ├── labels/lbl<five_digit_id>.png
│   │   ├── liverPcds/ptsLiverCam<five_digit_id>.ply
│   │   ├── CAM_K.yml
│   │   └── camPose.yml
│   ├── train_syn.txt
│   ├── test_syn.txt
│   ├── train_real.txt
│   ├── test_real.txt
│   └── readme.txt
└── .../21
```

官方 README 中的 `train/test.txt` 是示意写法，**不是发布代码实际打开的文件名**。代码严格读取上述四个清单。每个 patient 的 source 都是该患者唯一的术前 OBJ；清单的每一行指定一个 synthetic 或 real target frame。

发布数据 real 清单共 344 帧，而论文称 346 keyframes，相差 2 帧。发布 synthetic 清单共 49,000 帧；患者 `01`–`06`、`21` 各 2,000 帧，`07`–`20` 各 2,500 帧，亦不同于论文“每位患者 2,500 帧”的概括。

### 2.2 各目录在 rigid-only 实验中的作用

| 路径 | 含义 | rigid synthetic 必需性 |
|---|---|---:|
| `model/reconstructed_mesh_world_m.obj` | 完整术前 CT 肝脏三角网格，world/object 坐标，米 | 必需 |
| `syn/liverPcds/*.ply` | synthetic camera 中可见的局部肝表面，文件原始数值为 mm | 必需 |
| `syn/camPose.yml` | 每帧 Blender camera-to-world 外参；构造 rigid GT | 必需 |
| `syn/labels/*.png` | synthetic 肝 mask | rigid 网络不使用其像素内容；原代码为完整模型输入结构仍加载 |
| `syn/CAM_K.yml` | synthetic 相机内参 | rigid 点云匹配和 pose GT 不需要；仅后续渲染链使用 |
| `real/liverPcds/*.ply` | 单目深度恢复并尺度校准后的术中可见肝点云 | 不用于监督 rigid Table I |
| `real/labels/`、`real/scale/`、`real/CAM_K.yml` | real 自训练、尺度一致和渲染信息 | rigid-only 不使用 |

## 3. Source cloud 与 Target cloud

### Source cloud

- 文件：`<patient>/model/reconstructed_mesh_world_m.obj`。
- 语义：从术前 CT 分割得到的**完整患者特异性肝脏模型**。
- 官方加载流程：读取三角网格 → 均匀采样 200,000 点 → Open3D voxel downsample `0.0006 m` → 每个 sample 随机截取至最多 8,192 点。
- 法向量只为后续网格/非刚性流程传播；刚性 KPFPN 输入 feature 是常数 1。

### Target / reference cloud

- 文件：`<patient>/syn/liverPcds/<frame>.ply`。
- 语义：从患者特异性 Blender synthetic 场景渲染得到、处于相机坐标系中的**局部可见肝表面**；partiality/occlusion 已固化在点云中，并非在线 crop 产生。
- 官方加载流程：Open3D `uniform_down_sample(every_k_points=8)` → 去重复点 → statistical outlier removal (`nb_neighbors=20`, `std_ratio=2.0`) → 坐标 `/1000` → 去除全零点 → 随机截取或 wrap-padding 至 8,192 点。
- 对 synthetic 分支不做轴翻转；对 real 分支额外乘 `diag(1,-1,-1)` 从 OpenCV 相机轴变为 Blender 相机轴。

RTOR adapter 对应关系应固定为：

```text
src_points = complete preoperative liver
ref_points = partial synthetic liver in camera coordinates
transform  = T_ref_from_src = T_camera_from_world
```

## 4. GT transform、坐标系和方向

### 4.1 Synthetic GT 来源

`camPose.yml` 的每帧条目提供：

```text
cam_R_c2w   # camera -> world rotation
cam_t_c2w   # camera origin in world, stored in millimetres
```

发布代码执行：

```text
t_c2w_m = cam_t_c2w / 1000
T_world_from_camera = [R_c2w, t_c2w_m]
T_gt = inverse(T_world_from_camera)
     = T_camera_from_world
target = R_gt @ source + t_gt
```

因此 `R_gt` 把 world/object 方向旋转到 synthetic camera frame；`t_gt` 是 source/world 原点在 target/camera frame 下的位置，单位 m。`camPose.yml` 是 synthetic rigid GT 的唯一直接来源；labels、CAM_K 和 real scale 文件都不是 rigid GT。

### 4.2 Real 分支没有 rigid GT

发布 dataset 对 `real` 直接令 `R=I, t=0`。这不是物理配准真值，而是占位值。real 数据经 synthetic 预训练刚性网络估计初始 pose，再用于非刚性自监督训练/测试；因此不能用 real 分支的单位矩阵计算公平的 RRE/RTE/RR。

### 4.3 数据样本数值验证

对官方 Kaggle 患者 `01` 的 synthetic test frame `ptsLiverCam01900` 进行了不使用 ICP/GT 优化的只读检查：

| 项目 | 数值 |
|---|---:|
| source OBJ 范围 | `0.232 × 0.145 × 0.133 m` |
| target PLY 原始范围 | `209 × 138 × 177`（mm） |
| target `/1000` 后范围 | `0.209 × 0.138 × 0.177 m` |
| target→source 最近邻均值，施加 GT 前 | `112.88 mm` |
| target→source 最近邻均值，施加 GT 后 | `0.66 mm` |
| 双向最近邻均值之和，施加 GT 前 | `268.27 mm` |
| 双向最近邻均值之和，施加 GT 后 | `23.74 mm` |
| `det(R_c2w)` | `1.000000019` |

该检查同时确认了单位和 `source -> target` 方向。双向值的 GT 后残差较大于单向 target→source 是完整 source 对局部 target 的正常 complete-to-partial 效应。

### 4.4 RTOR 当前 convention

当前 RTOR/P2P 数据接口把 target 放入 `ref_points`、complete liver 放入 `src_points`，loss 与 evaluator 均执行 `apply_transform(src_points, transform)` 后同 ref 比较；模型输出也声明为 src→ref。因此 P2I-LReg adapter 不应反转 GT。

## 5. Unit

| 对象 | 文件存储 | 进入 rigid 网络 |
|---|---:|---:|
| 术前 OBJ | m | m |
| synthetic/real PLY | mm | 除以 1000 后为 m |
| `cam_t_c2w` | mm | 除以 1000 后为 m |
| voxel/radius/IR/RR 阈值 | 配置以 m 表示 | m |
| RTE | 发布 tester 原始数值为 m | 论文表格要求显式转为 mm |
| RRE | 发布 tester 原始数值为 rad | 论文表格要求显式转为 degree |

禁止再做全局归一化后仍把阈值解释成米。若 RTOR 为数值稳定性增加坐标归一化，所有 GT 平移和距离阈值必须一致变换，并在输出时恢复物理单位；该操作属于实现层选择，不是 Self-P2IR released-code 数据协议。

## 6. Preprocessing

| 操作 | Released code | 说明/风险 |
|---|---|---|
| Centering | 否（rigid synthetic） | 不做独立中心化 |
| PCA scale | 否（rigid synthetic） | 论文 PCA scale 是 real 单目点云尺度一致模块，不应加入 synthetic rigid benchmark |
| Source voxel | `0.0006 m` | 只在 mesh 200k 均匀采样后使用 |
| Target voxel | 无显式 voxel | 使用 `every_k_points=8`，再去重/统计滤波 |
| KPFPN nominal voxel | `first_subsampling_dl=0.001 m` | 控制 KPConv 邻域/层级；发布 adapter 并未对两云统一先做 1 mm Open3D voxel |
| Input points | 最多/通常恰为 8,192 | source 随机截断；target 随机截断或 wrap-padding |
| Cropping | 无在线 crop | synthetic target 的局部可见性来自渲染 |
| Train random rotation | 有，仅 syn | 每个 Euler 分量从高斯采样并截断至 ±0.18 rad；随机施加到 source 或 target |
| Train random translation | 无 | camera GT 本身含平移，但没有额外平移增强 |
| Train random scale | 有，仅 syn | isotropic `U(0.8,1.25)`；并把 scale/逆 scale 乘入 `pose_R` |
| Train noise | 有 | 注释称 Gaussian，实际是逐坐标 uniform；硬编码宽度 `0.002 m`，即每坐标约 ±1 mm |
| Test augmentation | 无 | 但点截断仍使用全局 NumPy RNG |

两个必须修正或显式分轨的发布代码问题：

1. synthetic scale augmentation 会令 `pose_R` 不再属于 SO(3)，与论文所称 SE(3) rigid GT 冲突。它不能无说明地进入严格 rigid-only 主协议。
2. 若 synthetic 样本在 `0.0065 m` 半径下少于 1,000 个 input-level correspondence，`__getitem__` 即使在 test 模式也随机换成其他样本。这会造成困难样本被替换、重复样本和指标分母错误。SCI 评估 adapter 应 fail/warn 并保留 case/frame identity，不得静默换样本。

此外，YAML 中 `augment_noise: 0.005` 未被 dataset 使用；dataset 构造函数把它硬编码为 `0.002`。

## 7. Voxel size 与 Input points：paper vs code

| 项目 | Paper protocol | Released-code protocol | 差异 |
|---|---:|---:|---|
| 输入点数 | 8,192 | `max_points=8192` | 基本一致；代码 target 可能 wrap-padding |
| KPFPN voxel | 0.001 m | `first_subsampling_dl=0.001` | 配置值一致，但代码不等价于“输入前统一 1 mm voxel”；source 另有 0.6 mm voxel，target 为 every-8 |
| coarse GT match radius | 未在实验段给出 | `0.06 m` mutual NN | 发布值为 60 mm，必须原样记录，不能误当 6 mm |
| raw correspondence filter radius | 未给出 | `0.0065 m` | 只用于过滤/重采样，非 coarse loss 半径 |

## 8. Train / Val / Test split

### 8.1 Synthetic（rigid 主协议）

发布数据清单的实测计数：

| 患者 | train_syn | test_syn |
|---|---:|---:|
| `01`–`06` | 每人 1,900 | 每人 100 |
| `07`–`20` | 每人 2,350 | 每人 150 |
| `21` | 1,900 | 100 |
| **总计** | **46,200** | **2,800** |

- 每个患者 train/test 无重叠。
- 没有 `val_syn.txt`。发布 `get_datasets()` 令 `val_set = test_set`，即 validation 与 test 完全相同。
- 论文称所有 synthetic 数据随机 60% train / 20% val / 20% test；发布清单实际约 94–95% train / 5–6% test。

### 8.2 Real（不用于监督 rigid Table I）

- 固定 train patients：`03,04,05,06,07,08,09,10,11,12,14,15,16,19,20,21`，252 帧。
- 固定 test patients：`01,02,13,17,18`，92 帧。
- 没有独立 val；代码同样让 val=test。
- 论文写的是 patient-level 五折交叉验证，每折 12 train / 4 val / 5 test。发布包只给出一个 16/5 合并清单，没有五折映射，也无法从文件中恢复 16 人中哪 12 人是 train、哪 4 人是 val。

### 8.3 对复现实验的约束

rigid Table I 只能在 synthetic test 上计算。为了不利用 test 做 model selection，后续实现应把以下两种口径分开命名，不能混报：

- **released-split track**：严格使用 46,200/2,800 清单，固定训练轮数，最终 epoch 评估；不把 test 当 validation 选 checkpoint。
- **paper-targeted track**：需要作者提供原始 60/20/20 synthetic split（及若要 real 五折时的 5 份 12/4/5 patient split）。在拿到之前状态必须标为 `NOT_REPRODUCIBLE_FROM_RELEASE`，不得自行声称复现论文 split。

## 9. Training protocol

| 项目 | Paper protocol | Released train config/script | 结论 |
|---|---:|---:|---|
| rigid data | synthetic | 默认 YAML 是 `dataset: real`；`main.py` 的 real 分支加载 synthetic checkpoint 并冻结 posenet | `train.sh` 默认并非从头训练 rigid；训练 rigid 必须显式切换 `dataset: syn` |
| rigid epochs | 120 | `max_epoch: 80`，且循环 `range(1,max_epoch)` 实际只跑 79 | 不一致 |
| optimizer | SGD | SGD, momentum 0.93, weight decay 1e-6 | 一致/代码更具体 |
| initial LR | 1e-4 | 1e-4 | 一致 |
| scheduler | 未在实验段细述 | ExponentialLR, gamma 0.95/epoch | 采用 released-code 时需记录 |
| batch size | 2（两张 RTX 3090） | per-process 1；`train.sh` 启 3 GPU，名义 global batch 3 | 不一致 |
| loss weights | `Lcorr`、`Ltran` 均为 1 | `match_weight=1`, `motion_weight=0` | released code 未启用 transform/motion loss |
| match threshold ηH | 0.15 | 训练 matcher config 0.2；tester 强制 0.15 | 不一致 |

发布 synthetic 训练路径还存在可执行性缺陷：`do_valid=True` 时 validation loss 字典无 `loss` 键但训练器访问 `stats_meter['loss']`；首轮验证前后又会冻结/保持冻结 `posenet`。所以不能把未经修复的 `train.sh` 运行结果称作官方 rigid-from-scratch 复现。后续实现必须把必要 bug fix 写入 manifest，并保持所有三种方法同一训练数据和选择规则。

## 10. Correspondence definition

### 10.1 Ground-truth correspondence

训练监督不是使用 marker 或人工 landmark。collate 阶段在 KPFPN coarse level：

1. 用 `T_gt` 将 coarse source 变换到 target frame；
2. 在变换后 source 与 target 间计算 mutual nearest-neighbor；
3. 欧氏搜索半径为 `coarse_match_radius=0.06 m`；
4. 这些 coarse mutual-NN pair 构成 focal correspondence loss 的正样本。

dataset 还计算 input-level Open3D correspondence，半径 `0.0065 m`，但 rigid coarse loss 使用 collate 中重新计算的 0.06 m correspondence；前者主要作为“至少 1000 对”的样本过滤条件。

### 10.2 Predicted correspondence 与 top-k

论文 Table I 用 confidence top-k，`k ∈ {2000,1500,1000,500,250}`。发布 tester 并未实现该表协议：

- `CM.get_match(conf_matrix, thr=0.15, mutual=False)` 选择所有置信度超过阈值的 pair；
- `get_topk_match` 与 `get_match` 的函数体实际相同，也没有 `k` 参数；
- 模型内部 SoftProcrustes 会按展平置信度排序，但取的数量是 batch 中 coarse cloud 最大长度乘 `sample_rate=1.0`，不是 Table I 的固定 k；
- tester 随后又忽略模型内部的 pose，并基于阈值匹配重新估计 pose。

因此 Table I 的 top-k 结果**不能由发布 tester 原样复现**；后续共享 evaluator 必须实现确定性的 score-descending top-k，并为不足 k 的情形定义且记录规则。

## 11. Pose estimator

| Estimator | Paper | Released code |
|---|---|---|
| RANSAC-50k | Table I 默认；50,000 iterations | Open3D 函数存在，criteria 为 `(50000,1000)`，但 tester 调用被注释；注释调用阈值为 0.001 m |
| LGR | Table I 对照 | Self-P2IR 发布仓库无可调用 LGR 实现 |
| weighted SVD | Table I 对照、网络内部 pose | `SoftProcrustes` 内部确实按 confidence 加权；但 tester 的 `ransac_regist_coarse` 调 `weighted_svd(..., weights=None)`，实际为**无权 SVD** |

发布 tester 函数名 `ransac_regist_coarse` 不能作为已使用 RANSAC 的证据；实际执行路径以函数体为准。公平比较应让 Self-P2IR、GeoTransformer baseline、RTOR 输出同一格式的 `(src_corr, ref_corr, score)`，再由同一个共享 estimator 实现分别运行 RANSAC-50k、LGR、weighted SVD；禁止 ICP refinement。

## 12. Metrics

### 12.1 论文定义

- **IR**：预测 correspondence 中，经 GT 对齐后欧氏距离 `< τ1` 的比例。
- **FMR**：IR `> τ2=5%` 的点云 pair 比例。
- **RR**：估计变换误差 `< ρ` 的点云 pair 比例。
- **RRE**：预测与 GT rotation 的 geodesic error，单位 degree。
- **RTE**：预测与 GT translation 的 L2 error，单位 mm。

论文 Table I 对每个 estimator 和 top-k 报告上述五项；正文还报告不同 `ρ`，并明确提到 `ρ=5 mm` 和 synthetic overlap 分析中的 `ρ<0.01 m`。

### 12.2 发布 tester 的精确定义及问题

- **IR 主输出**：tester 调用 `inlier_thr=0.1`；先计算**平方距离**，再判断 `< 0.1²`，故主 IR 实际是 100 mm 半径。其它命名为 `0.015/0.005/...` 的分支也直接与平方距离比较而未平方阈值，命名和物理阈值不一致。
- **FMR 主输出**：`IR(main) > 0.05` 的 pair fraction；因此逻辑上的 5% 一致，但依赖上述 100 mm IR。
- **RR**：在原始 8,192 source 点上计算 `mean(||T_est(x)-T_gt(x)||)`，再分别判断 `< {0.04,0.02,0.01,0.005,0.002} m`。这是 source-surface transformation RMSE 的 L1-over-points 版本，不是仅靠 RRE/RTE 的阈值。
- **RRE**：公式正确计算 geodesic angle，但函数返回 radians，未转 degree；同时 trace clip 使用 `[-1,3]` 后再代入 `(trace-1)/2`，数值仍可落入 `[-1,1]`。
- **RTE**：直接返回 metre-space translation L2，未乘 1000；论文表头却是 mm。
- **失败样本**：匹配少于 3 对时 pose 设为单位变换且 RR=0；但 `corrs` 数组的 batch 结构可能不完整。

因此后续论文表格不能直接抄发布 tester 的打印值。必须建立共享、带单位测试的 metric 实现，并同时保留一个 `released_code_compat` 输出用于对照诊断。

### 12.3 建议的共享 evaluator 固定项（实施阶段门槛）

在不引入任何 oracle/ICP 的前提下，对三种模型统一：

```text
transform direction: src -> ref
distance unit inside evaluator: metre
reported RRE: degree
reported RTE: millimetre
top-k: 2000, 1500, 1000, 500, 250
FMR threshold: IR > 0.05
pose estimators: RANSAC-50k / LGR / weighted SVD
```

`τ1`、RANSAC inlier threshold、RR 的唯一主 `ρ` 仍需在实施前冻结：论文文字/图表与发布 tester 不能唯一导出 Table I 的全部精确参数。合理做法是从 Predator/GeoTransformer 官方 evaluator 的对应协议核对后，在 `protocol.py` 中显式版本化，不能凭结果反推或调阈值。

## 13. Paper protocol / released-code protocol / difference 总表

| 项目 | Paper protocol | Released-code/data protocol | Difference / 风险 |
|---|---|---|---|
| rigid train/eval domain | synthetic | synthetic 分支；默认 YAML 却为 real 完整框架 | rigid adapter 应只取 syn |
| source/target | complete preop / rendered partial | 同 | 一致 |
| GT | SE(3) source→target | inverse(C2W)，但 scale augmentation 可破坏 SO(3) | clean test 一致，train augmentation 冲突 |
| points | 8192 | max 8192，target 可 wrap-pad | 基本一致 |
| voxel | 1 mm | KPFPN 参数 1 mm；source 0.6 mm、target every-8 | 实际预处理不一致 |
| synthetic split | 60/20/20 | 46,200/0/2,800，val=test | 严重不一致 |
| real split | 五折 12/4/5 | 单一 16/0/5，val=test | 五折不可从发布文件恢复 |
| epochs | 120 | 配置 80，循环实际 79 | 不一致 |
| batch | 2 on 2 GPUs | 1/process，脚本 3 GPUs | 不一致 |
| LR | 1e-4 | 1e-4 | 一致 |
| optimizer | SGD | SGD, momentum .93 | 一致/代码更具体 |
| loss | Lcorr + Ltran，各权重 1 | match=1, motion=0 | 不一致 |
| top-k | 2000/1500/1000/500/250 | tester 无固定 top-k | 缺失 |
| RANSAC-50k | 默认 | 实现存在但调用被注释 | 未执行 |
| LGR | 对照 | 缺失 | 未发布 |
| weighted SVD | 对照 | tester 实际无权 SVD | 命名/执行不一致 |
| RRE/RTE unit | degree/mm | rad/m | 需显式转换 |
| test identity | 每个固定 test pair | 不足 1000 correspondence 时随机换样本 | 不可接受，需修复 |

## 14. 第一阶段 Gate 与后续实施约束

### 已确定，可以据此实现

- rigid-only 的 dataset domain 为 `syn`。
- `src=完整术前模型`，`ref=synthetic partial liver`。
- `T_gt = inverse(camPose C2W)`，严格 `src -> ref`。
- 内部物理单位为 m；输出 RTE 为 mm、RRE 为 degree。
- 不使用 real 单位矩阵作为 GT，不使用 mask、marker、GT correspondence 进入推理，不加入 ICP、test-time oracle 或 non-rigid 模块。
- 新实验可复用 RTOR `geotransformer/` 公共模块；不得修改既有 `experiments/geotransformer.p2p_liver/` 的行为。

### 尚未由官方发布材料唯一确定，实施时必须显式版本化

1. 论文 60/20/20 synthetic 原始 split 未发布；released track 应使用现有 46,200/2,800，paper-targeted track 标记为不可复现，直到作者提供 split。
2. Table I 固定 top-k 的 released evaluator 未发布；必须用同一共享实现公平重算三个模型，而不能声称逐 bit 复现原表。
3. Table I 的精确 `τ1` 与统一主 `ρ` 需要再对照其引用的 Predator/GeoTransformer 官方 benchmark 代码后冻结；Self-P2IR tester 的 100 mm IR 和混用平方阈值不可直接采信。
4. 严格 SE(3) 实验应禁用 released scale augmentation；若保留它，只能作为单独的 `released_aug_compat` 消融，因为其 GT 线性部分不是 rotation。
5. 训练轮数应至少输出两个清晰标签：`released_config_80` 与 `paper_120`。所有方法的主公平表必须采用相同 split、epoch selection 和 estimator，不能为某一方法单独挑更优口径。

## 15. 实施阶段必须加入的自动检查

后续 `P2ILRegDataset` 至少应检查：

1. `T_gt.shape == (4,4)`、最后一行为 `[0,0,0,1]`、所有值有限。
2. clean rigid track 中 `R.T @ R ≈ I`、`det(R) ≈ 1`；scale compatibility track 另行标记。
3. source/ref 尺寸、坐标范围和物理单位 sanity check，防止漏除 1000。
4. 以 asymmetric complete-to-partial 友好的 `Chamfer_ref_to_aligned_src` 为主，并附 symmetric Chamfer：大多数样本必须满足 after < before；否则 warning，超过预设失败率则 hard error。
5. 每个 sample 返回稳定的 `patient_id`、`frame_id`、`case_id`；不得在 test 失败时随机替换样本。
6. split 零交集、文件存在性、frame 与 `camPose.yml` key 一致性。
7. evaluator 单元测试固定 degree/mm、平方距离与欧氏距离转换、top-k tie-breaking、少于 3 对 correspondence 的失败行为。

## 16. 审计依据

主要源码入口：[`datasets/dataset.py`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/datasets/dataset.py)、[`datasets/dataloader.py`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/datasets/dataloader.py)、[`configs/train/main_config.yaml`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/configs/train/main_config.yaml)、[`configs/test/main_config.yaml`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/configs/test/main_config.yaml)、[`main.py`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/main.py)、[`trainer.py`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/trainer.py)、[`tester.py`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/tester.py)、[`lib/loss.py`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/lib/loss.py)、[`models/lepard/procrustes.py`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/models/lepard/procrustes.py)、[`models/lepard/matching.py`](https://github.com/junzastar/Self-P2IR/blob/60d6c1b30d57148e6bc0c97fd1fab0f68cc0f079/models/lepard/matching.py)。

本审计没有把注释掉的代码、函数名或论文概述当成实际执行路径；所有差异均保留，不以猜测补齐缺失协议。
